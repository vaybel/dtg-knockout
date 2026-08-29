"""Load and save knockout weights in the Hugging Face format (config.json + safetensors).

    from knockout.pretrained import from_pretrained
    model = from_pretrained("vaybel/dtg-knockout")      # downloads from the Hub
    model = from_pretrained("weights/v6")               # or a local dir

Weights ship as `model.safetensors` (no pickle — safe to load from untrusted sources) plus a
`config.json` carrying the architecture (width/depth). Convert a training checkpoint with:

    python -m knockout.pretrained --ckpt checkpoints_v6/best.pt --out weights/v6
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model import KnockoutMatte


def save_pretrained(model: KnockoutMatte, out_dir: str | Path, *, width: int = 32,
                    depth: int = 4, in_ch: int = 4) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(out / "model.safetensors"))
    (out / "config.json").write_text(
        json.dumps({"arch": "KnockoutMatte", "in_ch": in_ch, "width": width, "depth": depth}, indent=2)
    )
    return out


def from_pretrained(repo_or_path: str, device: str = "cpu") -> KnockoutMatte:
    """Build a KnockoutMatte and load weights from a local dir or a Hugging Face repo id."""
    p = Path(repo_or_path)
    if (p / "config.json").exists():
        cfg = json.loads((p / "config.json").read_text())
        wpath = p / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        cfg = json.loads(Path(hf_hub_download(repo_or_path, "config.json")).read_text())
        wpath = hf_hub_download(repo_or_path, "model.safetensors")
    model = KnockoutMatte(in_ch=cfg.get("in_ch", 4), width=cfg.get("width", 32), depth=cfg.get("depth", 4))
    model.load_state_dict(load_file(str(wpath)))
    return model.to(device).eval()


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Convert a training checkpoint to the pretrained format.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    w, d = ck.get("width", 32), ck.get("depth", 4)
    m = KnockoutMatte(width=w, depth=d)
    m.load_state_dict(ck["model"])
    print("saved pretrained ->", save_pretrained(m, args.out, width=w, depth=d))


if __name__ == "__main__":
    _main()
