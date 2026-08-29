"""Foreground-corpus hygiene: score out photo content and damaged-alpha labels.

Two failure classes pollute a matting corpus built from found transparent PNGs:

- photo cutouts (real photographs die-cut to a sticker) — off-domain for print graphics,
  detected by their gradient texture: photos shade continuously almost everywhere, flat
  illustration is flat with sharp edges.
- damaged labels (a previous remover's over-crop baked into the PNG's own alpha) — a
  moth-eaten silhouette teaches the model to eat art. Detected by contour raggedness
  relative to a clean shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def photo_score(im: Image.Image, size: int = 256) -> float:
    """Fraction of opaque pixels with soft (photographic) shading. Illustration ~0.0-0.2."""
    im = im.convert("RGBA")
    im.thumbnail((size, size))
    arr = np.asarray(im, np.float32) / 255.0
    a = arr[:, :, 3] > 0.5
    if a.sum() < 500:
        return 0.0
    lum = arr[:, :, :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    gy, gx = np.gradient(lum)
    g = np.hypot(gx, gy)
    soft = (g > 0.004) & (g < 0.05)
    return float(soft[a].mean())


def damage_score(im: Image.Image, size: int = 384, band_px: int = 4) -> float:
    """Fraction of the just-removed border band that was NOT the background color.

    Most remover outputs keep RGB under alpha=0. A clean knockout removed a near-uniform
    ground, so the transparent band hugging the contour matches the transparent region's
    dominant color. When a remover bit into artwork, the band still carries the art's
    colors — content continuity across the alpha edge is the damage signature.
    Returns -2.0 when RGB under transparency was zeroed/flattened (signal unavailable).
    """
    im = im.convert("RGBA")
    im.thumbnail((size, size))
    arr = np.asarray(im, np.float32) / 255.0
    a = arr[:, :, 3] > 0.5
    t = ~a
    if a.sum() < 500 or t.sum() < 500:
        return 0.0
    rgb_t = arr[t][:, :3]
    if rgb_t.std() < 0.02:  # RGB wiped under transparency — continuity signal gone
        return -2.0
    bg_est = np.median(rgb_t, axis=0)
    band = t & ndimage.binary_dilation(a, iterations=band_px)
    if band.sum() < 100:
        return 0.0
    d = np.linalg.norm(arr[band][:, :3] - bg_est, axis=1) / np.sqrt(3.0)
    return float((d > 0.12).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--metric", choices=["photo", "damage"], required=True)
    p.add_argument("--out", type=Path, required=True, help="TSV: path<TAB>score")
    args = p.parse_args()
    fn = photo_score if args.metric == "photo" else damage_score
    with args.out.open("w") as f:
        for i, path in enumerate(sorted(args.dir.rglob("*.png"))):
            try:
                s = fn(Image.open(path))
            except Exception:  # noqa: BLE001 — corpus files can be arbitrarily broken
                s = -1.0
            f.write(f"{path}\t{s:.4f}\n")
            if (i + 1) % 1000 == 0:
                print(i + 1, flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
