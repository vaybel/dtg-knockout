"""MPS training loop for the known-background matting model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

from .dataset import MatteDataset
from .losses import matte_loss
from .metrics import gt_scores
from .model import KnockoutMatte, param_count


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: KnockoutMatte, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    agg = {"iou": 0.0, "over_crop": 0.0, "under_crop": 0.0, "sad": 0.0}
    n = 0
    for x, y, _bg in loader:
        pred = torch.sigmoid(model(x.to(device))).cpu().numpy()
        yy = y.numpy()
        for b in range(pred.shape[0]):
            for k, v in gt_scores(pred[b, 0], yy[b, 0]).items():
                agg[k] += v
            n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, nargs="+", required=True,
                   help="one or more manifest dirs; concatenated (synth pools + teacher pairs)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--device", default="mps")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--depth", type=int, default=4, help="U-Net levels; raise to 5 for 512px+ RF")
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip; 0 to disable")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers; 0 avoids MPS deadlock")
    p.add_argument("--max-steps", type=int, default=0, help="smoke cap; 0 = full epochs")
    p.add_argument("--init-from", type=Path, default=None,
                   help="warm-start: load model weights from this checkpoint before training")
    p.add_argument("--doctrine", action="store_true",
                   help="enable v12 visibility/fringe/decisiveness/mottle loss terms")
    p.add_argument("--lam-v", type=float, default=1.0,
                   help="visibility-weight strength; lower to reduce close-color over-crop")
    args = p.parse_args()

    device = pick_device(args.device)
    parts = [MatteDataset(d, size=args.size) for d in args.data]
    ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    print("pools: " + ", ".join(f"{d.name}={len(p)}" for d, p in zip(args.data, parts)))
    n_val = max(1, int(len(ds) * args.val_frac))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    model = KnockoutMatte(width=args.width, depth=args.depth).to(device)
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck0["model"])
        print(f"warm-started from {args.init_from} (width={ck0.get('width')}, depth={ck0.get('depth')})")
    print(f"device={device} params={param_count(model)/1e6:.2f}M train={len(train_ds)} val={len(val_ds)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_ld, desc=f"epoch {epoch}")
        for step, (x, y, bg) in enumerate(pbar):
            x, y, bg = x.to(device), y.to(device), bg.to(device)
            opt.zero_grad()
            if args.doctrine:
                loss = matte_loss(model(x), y, rgb=x[:, :3], dist=x[:, 3:4], garment=bg,
                                  lam_v=args.lam_v)
            else:
                loss = matte_loss(model(x), y)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            pbar.set_postfix(loss=float(loss.detach()))
            if args.max_steps and step + 1 >= args.max_steps:
                break
        sched.step()
        m = evaluate(model, val_ld, device)
        print(f"epoch {epoch}: iou={m['iou']:.3f} over={m['over_crop']:.3f} "
              f"under={m['under_crop']:.3f} sad={m['sad']:.2f}")
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            torch.save(
                {"model": model.state_dict(), "width": args.width, "depth": args.depth},
                args.out / "best.pt",
            )
        if args.max_steps:
            break


if __name__ == "__main__":
    main()
