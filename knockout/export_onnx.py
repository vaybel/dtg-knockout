"""Export a training checkpoint to ONNX for torch-free deployment.

The deployment target only needs onnxruntime + numpy/scipy: the model becomes a .onnx graph
and the deterministic post-ops (refine/cleanup/gate) are already numpy. Includes a parity
check — the exported graph must match the torch model to float tolerance before it ships.

    python -m knockout.export_onnx --ckpt checkpoints_v9/best.pt --out weights/knockout_v9.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .model import KnockoutMatte


def export(ckpt_path: str, out_path: str, size: int = 512, opset: int = 17) -> None:
    ck = torch.load(ckpt_path, map_location="cpu")
    model = KnockoutMatte(width=ck.get("width", 32), depth=ck.get("depth", 4)).eval()
    model.load_state_dict(ck["model"])

    dummy = torch.zeros(1, 4, size, size)
    # external_data=False: a single self-contained .onnx file (no sidecar .data) — one
    # artifact to version, hash, and ship.
    torch.onnx.export(
        model, (dummy,), out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "logits": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset, external_data=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4):
        x = rng.random((1, 4, size, size), dtype=np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"input": x})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"exported {out_path}  parity max|Δlogit|={worst:.2e}")
    if worst > 1e-3:
        raise SystemExit("parity check FAILED — do not ship this export")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=512)
    args = p.parse_args()
    export(args.ckpt, args.out, args.size)


if __name__ == "__main__":
    main()
