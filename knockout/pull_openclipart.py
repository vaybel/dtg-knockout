"""Rasterize CC0 Openclipart SVGs into transparent-PNG foregrounds (perfect alpha labels).

Streams `nyuuzyou/openclipart` (metadata only — no 22GB download), rasterizes each inline
svg_content with cairosvg, keeps renders whose alpha fraction is in a usable band (real
transparent shape, not a full-canvas fill). Vector diversity is the proven OOD lever.

Run:
    python -m knockout.pull_openclipart --n 10000 --out data/fg_openclipart
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

import numpy as np


class _Timeout(Exception):
    pass


def _rasterize(svg: str, size: int, seconds: int) -> bytes:
    """Rasterize with a hard wall-clock cap — some Openclipart SVGs (huge path counts,
    recursive patterns) spin cairosvg forever; skip them rather than block the stream."""
    import cairosvg

    def _alarm(_sig, _frame):
        raise _Timeout

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=Path("data/fg_openclipart"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-attempts", type=int, default=60000)
    ap.add_argument("--svg-timeout", type=int, default=5, help="seconds per SVG before skip")
    args = ap.parse_args()

    from datasets import load_dataset
    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nyuuzyou/openclipart", split="train", streaming=True)
    kept = len(list(args.out.glob("ocal_*.png")))  # resume: keep already-pulled
    seen = attempts = timeouts = 0
    import io

    for row in ds:
        if kept >= args.n or attempts >= args.max_attempts:
            break
        svg = row.get("svg_content")
        if not svg or "<svg" not in svg:
            continue
        seen += 1
        if seen <= kept:  # already rasterized on a prior run — skip cheaply
            continue
        attempts += 1
        try:
            png = _rasterize(svg, args.size, args.svg_timeout)
            im = Image.open(io.BytesIO(png)).convert("RGBA")
        except _Timeout:
            timeouts += 1
            continue
        except Exception:
            continue
        frac = float((np.asarray(im.getchannel("A")) > 16).mean())
        if not (0.02 <= frac <= 0.95):
            continue
        im.save(args.out / f"ocal_{kept:06d}.png")
        kept += 1
        if kept % 250 == 0:
            print(f"{kept}/{args.n} kept ({attempts} attempts, {timeouts} timed out)", flush=True)

    print(f"DONE: {kept} foregrounds -> {args.out} ({attempts} attempts, {timeouts} timed out)")


main()
