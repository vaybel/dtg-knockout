"""Typography foreground generator — transparent-PNG text art with perfect alpha labels.

Print typography is the biggest DTG category and the corpus's weakest class: glyph counters,
inter-letter gaps, thin script strokes, outlined varsity type, distressed slogans. Each render
is a foreground for synth_composite, with effects sampled to cover those exact structures:
outline (sticker case), drop shadow, glow, arc warp, multi-line stacks, distress erosion,
thin-stroke fonts. Alpha is exact by construction.
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = [
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    str(Path.home() / "Library/Fonts"),
]

WORDS = (
    "wild free brave happy blessed grateful chaos coffee mama papa dog cat lake beach "
    "sunset mountain vibes squad crew club team est legend queen king boss babe salty "
    "sweet lucky loved chosen kind strong fearless golden retro vintage classic original "
    "custom limited edition tour trip camp fest game day night year"
).split()
PHRASES = [
    "STAY WILD", "GOOD VIBES ONLY", "BEST DAD EVER", "DOG MOM", "LAKE LIFE",
    "EST. 2024", "GAME DAY", "BE KIND", "COFFEE FIRST", "WORLD CUP 2026",
    "I ASKED FOR HELP", "LIVE FREE", "HOME SWEET HOME", "SUNSHINE STATE OF MIND",
    "SAVE THE BEES", "ADVENTURE AWAITS", "MAMA BEAR", "PLANT LADY", "SALT AIR",
]


def _covers_latin(font: ImageFont.FreeTypeFont) -> bool:
    """Reject fonts that draw Latin as .notdef boxes: distinct letters must render distinctly."""
    masks = [np.asarray(font.getmask(c)) for c in "AGE"]
    if any(m.size == 0 or not m.any() for m in masks):
        return False
    a, g, e = (m.astype(bool) for m in masks)
    return not (a.shape == g.shape == e.shape
                and (a == g).all() and (g == e).all())


def load_fonts() -> list[str]:
    ok = []
    for d in FONT_DIRS:
        for f in glob.glob(d + "/*.tt[fc]") + glob.glob(d + "/*.otf"):
            try:
                font = ImageFont.truetype(f, 48)
                if _covers_latin(font):
                    ok.append(f)
            except OSError:
                continue
    if not ok:
        raise SystemExit("no usable fonts found")
    return sorted(ok)


SUBTEXT = [
    "(now i'm being watched)", "est. 2026", "limited edition", "* handle with care *",
    "all rights reserved", "since 1970", "made in the usa", "batch no. 042",
    "not responsible for lost items", "terms and conditions apply", "small print goes here",
    "california - nevada", "population: you", "warning: may cause joy", "tap to continue",
]


def _phrase(rng: random.Random) -> list[str]:
    if rng.random() < 0.4:
        text = rng.choice(PHRASES)
    else:
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 4)))
        text = rng.choice([text.upper(), text.title(), text.lower()])
    words = text.split()
    if len(words) >= 3 and rng.random() < 0.5:  # multi-line stack
        cut = rng.randint(1, len(words) - 1)
        return [" ".join(words[:cut]), " ".join(words[cut:])]
    return [text]


def _lines_with_scale(rng: random.Random) -> list[tuple[str, float]]:
    """(text, size-scale) per line. Hierarchy and fine-print modes make the small text the
    model kept dropping: a big headline with a tiny subtext, or a block of fine print."""
    r = rng.random()
    if r < 0.35:  # headline + tiny subtext (the parenthetical/tagline failure)
        head = [(ln, 1.0) for ln in _phrase(rng)]
        subs = [(rng.choice(SUBTEXT), rng.uniform(0.16, 0.34))
                for _ in range(rng.randint(1, 2))]
        return head + subs
    if r < 0.5:  # fine-print block: several small lines
        n = rng.randint(2, 4)
        sc = rng.uniform(0.22, 0.40)
        return [(rng.choice(SUBTEXT), sc) for _ in range(n)]
    return [(ln, 1.0) for ln in _phrase(rng)]


def _arc(img: Image.Image, strength: float) -> Image.Image:
    """Vertical arc warp: displace columns along a parabola (varsity curve)."""
    w, h = img.size
    bulge = int(abs(strength) * h * 0.4)
    out = Image.new("RGBA", (w, h + bulge), (0, 0, 0, 0))
    arr, dst = np.asarray(img), np.zeros((h + bulge, w, 4), np.uint8)
    xs = np.arange(w)
    dy = (bulge * (1 - (2 * xs / max(w - 1, 1) - 1) ** 2)).astype(int)
    if strength < 0:
        dy = bulge - dy
    for x in xs:
        dst[dy[x]:dy[x] + h, x] = arr[:, x]
    out = Image.fromarray(dst)
    return out


def render_text(rng: random.Random, fonts: list[str], canvas: int = 1024) -> Image.Image | None:
    lines = _lines_with_scale(rng)
    font_path = rng.choice(fonts)
    base = rng.randint(70, 210)
    fill = tuple(rng.randint(0, 255) for _ in range(3)) + (255,)
    stroke_fill = rng.choice([(255, 255, 255, 255), (0, 0, 0, 255),
                              tuple(rng.randint(0, 255) for _ in range(3)) + (255,)])

    pad = 80
    img = Image.new("RGBA", (canvas * 2, canvas), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = pad
    for text, scale in lines:
        size = max(14, int(base * scale))
        try:
            font = ImageFont.truetype(font_path, size)
        except OSError:
            return None
        stroke = rng.randint(3, 12) if (scale >= 0.9 and rng.random() < 0.25) else 0
        d.text((canvas, y), text, font=font, fill=fill, anchor="ma",
               stroke_width=stroke, stroke_fill=stroke_fill)
        y += int(size * 1.25)
    bbox = img.getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    if rng.random() < 0.30:  # arc warp
        img = _arc(img, rng.uniform(-0.8, 0.8))
    if rng.random() < 0.25:  # glow / soft shadow behind
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        a = img.getchannel("A").filter(ImageFilter.GaussianBlur(rng.uniform(4, 12)))
        col = tuple(rng.randint(0, 255) for _ in range(3))
        glow.paste(Image.new("RGBA", img.size, col + (255,)), (0, 0), a)
        off = rng.randint(0, 10)
        base = Image.new("RGBA", (img.width + off, img.height + off), (0, 0, 0, 0))
        base.alpha_composite(glow, (off, off))
        base.alpha_composite(img, (0, 0))
        img = base
    if rng.random() < 0.30:  # distress erosion
        arr = np.asarray(img).copy()
        noise = np.random.default_rng(rng.randint(0, 2**31)).random(arr.shape[:2])
        speck = (Image.fromarray((noise * 255).astype(np.uint8))
                 .filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0))))
        keep = np.asarray(speck, np.float32) / 255.0 > rng.uniform(0.25, 0.45)
        arr[:, :, 3] = np.where(keep, arr[:, :, 3], 0)
        img = Image.fromarray(arr)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data/fg_text"))
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    fonts = load_fonts()
    print(f"{len(fonts)} fonts")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    while written < args.n:
        img = render_text(rng, fonts)
        if img is None or img.width < 64 or img.height < 32:
            continue
        if float((np.asarray(img.getchannel("A")) > 16).mean()) < 0.01:
            continue
        img.save(args.out_dir / f"text_{written:06d}.png")
        written += 1
        if written % 500 == 0:
            print(written, flush=True)
    print(f"wrote {written} to {args.out_dir}")


if __name__ == "__main__":
    main()
