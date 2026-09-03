"""Evaluate a checkpoint: GT scores + the prod no-GT gate over a threshold sweep.

The gate (boundary_ground_residue) is what will police the model in production, so we
report the accept-rate under it, not just IoU. Prod runs refine *then* the gate, so both
sides are reported: raw model output and the refined (prod-path) output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import MatteDataset
from .metrics import boundary_ground_residue, gt_scores, hex_to_rgb01
from .model import KnockoutMatte
from .refine import refine_alpha
from .train import pick_device

RESIDUE_ACCEPT = 0.02  # prod-style clean-contour gate


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--device", default="mps")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = KnockoutMatte(width=ckpt.get("width", 32), depth=ckpt.get("depth", 4)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    ds = MatteDataset(args.data, size=args.size)
    loader = DataLoader(ds, batch_size=8)  # unshuffled: the manifest index below relies on order
    agg = {"iou": 0.0, "over_crop": 0.0, "under_crop": 0.0, "sad": 0.0}
    agg_ref = dict.fromkeys(agg, 0.0)
    res_raw, res_ref, acc_raw, acc_ref, n = [], [], 0, 0, 0
    for bi, (x, y, _bg) in enumerate(loader):
        pred = torch.sigmoid(model(x.to(device))).cpu().numpy()
        for b in range(pred.shape[0]):
            a = (pred[b, 0] >= args.threshold).astype(np.float32)
            for k, v in gt_scores(a, y[b, 0].numpy()).items():
                agg[k] += v
            rgb01 = x[b, :3].permute(1, 2, 0).numpy()
            # reconstruct bg from the dataset item (kept in manifest order)
            bg01 = hex_to_rgb01(ds.items[bi * loader.batch_size + b]["bg_hex"])
            ar = refine_alpha(a, rgb01, bg01)
            for k, v in gt_scores(ar, y[b, 0].numpy()).items():
                agg_ref[k] += v
            r = boundary_ground_residue(a, rgb01, bg01)
            rr = boundary_ground_residue(ar, rgb01, bg01)
            res_raw.append(r)
            res_ref.append(rr)
            acc_raw += int(r <= RESIDUE_ACCEPT)
            acc_ref += int(rr <= RESIDUE_ACCEPT)
            n += 1

    def report(tag: str, scores: dict[str, float], residues: list[float], accepted: int) -> None:
        print(tag)
        for k, v in scores.items():
            print(f"  {k}: {v / n:.4f}")
        print(f"  boundary_residue mean={np.mean(residues):.4f} p90={np.percentile(residues, 90):.4f}")
        print(f"  gate accept-rate (residue<={RESIDUE_ACCEPT}): {accepted / n:.3f}")

    print(f"N={n}")
    report("raw model output:", agg, res_raw, acc_raw)
    report("refined (prod path: refine -> gate):", agg_ref, res_ref, acc_ref)


if __name__ == "__main__":
    main()
