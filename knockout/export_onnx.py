"""Export a training checkpoint to ONNX for torch-free deployment.

The deployment target only needs onnxruntime + numpy/scipy: the model becomes a .onnx graph
and the deterministic post-ops (refine/cleanup/gate) are already numpy. Includes a parity
check — the exported graph must match the torch model to float tolerance before it ships.

    python -m knockout.export_onnx --ckpt checkpoints_v9/best.pt --out weights/knockout_v9.onnx
"""

from __future__ import annotations
from pathlib import Path

import argparse

import numpy as np
import torch

from .model import KnockoutMatte


def export(ckpt_path: str, out_path: str, size: int = 512, opset: int = 17, parity_data: Path | None = None) -> None:
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

    def parity(x: np.ndarray) -> tuple[float, int]:
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"input": x})[0]
        return float(np.abs(ref - got).max()), int(((ref >= 0) != (got >= 0)).sum())

    # What ships is the binarized alpha, so the gate is on sign flips per pixel; the logit
    # bound only catches a broken graph. Random inputs exercise every op; real inputs, when a
    # MatteDataset dir is given, exercise the values the model actually sees.
    rng = np.random.default_rng(0)
    worst, flips, total = 0.0, 0, 0
    for _ in range(4):
        d, f = parity(rng.random((1, 4, size, size), dtype=np.float32))
        worst, flips, total = max(worst, d), flips + f, total + size * size
    print(f"exported {out_path}  noise: max|Δlogit|={worst:.2e} flips={flips / total:.1e}")
    if worst > 5e-3 or flips / total > 1e-4:
        raise SystemExit("parity check FAILED on random inputs — do not ship this export")
    if parity_data is not None:
        from .dataset import MatteDataset

        ds = MatteDataset(parity_data, size=size)
        worst, flips, total = 0.0, 0, 0
        for i in range(len(ds)):
            d, f = parity(ds[i][0][None].numpy())
            worst, flips, total = max(worst, d), flips + f, total + size * size
        print(f"  real inputs ({len(ds)}): max|Δlogit|={worst:.2e} flips={flips / total:.1e}")
        if flips / total > 2e-4:
            raise SystemExit("parity check FAILED on real inputs — do not ship this export")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--parity-data", type=Path, default=None,
                    help="MatteDataset dir for a real-input parity check (optional)")
    args = p.parse_args()
    export(args.ckpt, args.out, args.size, parity_data=args.parity_data)


if __name__ == "__main__":
    main()
