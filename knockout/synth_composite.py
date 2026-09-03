"""Turn transparent-PNG artwork into on-garment training triples (input, alpha, bg_hex).

Each foreground's own alpha IS the ground-truth matte. Composite onto a garment color
sampled from the real prod distribution, with production-matched artifacts (tint drift,
shading, soft shadow, anti-alias, noise, JPEG, disjoint multi-element layouts). The recorded
bg_hex is the *requested* color the pipeline knows; the pixels drift off it, as in prod.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .base_colors import sample_base_hex
from .metrics import hex_to_rgb01


@dataclass
class DriftConfig:
    max_tint_drift: float = 0.34
    shade_strength: float = 0.10
    shadow_prob: float = 0.5
    noise_sigma: float = 0.012
    jpeg_quality: tuple[int, int] = (72, 95)
    multi_element_prob: float = 0.35
    max_side: int = 512
    # tonal-collision op: ground sampled near the art's own color (small drift, or the
    # collision dissolves) — trains the case where the color-distance channel goes blind.
    collision_prob: float = 0.22
    collision_drift: float = 0.06
    # sticker op: white outline ring behind the art — near-iso-luminant edge on light grounds.
    outline_prob: float = 0.18
    # v12 ops
    distress_prob: float = 0.15    # grunge holes IN the art; GT carries them (intended speckle)
    badge_prob: float = 0.12       # art enclosed in a solid plate; GT keeps the whole plate
    dense_prob: float = 0.08       # 3-6 elements as one sheet (lockups, stacks)
    dense_max: int = 6
    inline_prob: float = 0.10      # collegiate lettering: ground-colored inline between fill and keyline
    pocket_prob: float = 0.10      # ground-colored pocket enclosed deep inside the art; GT keeps it
    scene_prob: float = 0.06       # art on a rectangular scene block whose interior runs near the ground; GT keeps the block


def _low_freq_shading(h: int, w: int, amp: float, rng: random.Random) -> np.ndarray:
    small = np.array([[1.0 + rng.uniform(-amp, amp) for _ in range(4)] for _ in range(4)], np.float32)
    field = Image.fromarray((small * 127.5).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(field, np.float32)[:, :, None] / 127.5


def _fit(fg: Image.Image, canvas: tuple[int, int], rng: random.Random) -> Image.Image:
    cw, ch = canvas
    scale = (rng.uniform(0.35, 0.85) * min(cw, ch)) / max(fg.size)
    return fg.resize((max(1, round(fg.width * scale)), max(1, round(fg.height * scale))), Image.LANCZOS)


def _paste(base: Image.Image, fg: Image.Image, rng: random.Random) -> None:
    base.alpha_composite(fg, (rng.randint(0, max(0, base.width - fg.width)),
                              rng.randint(0, max(0, base.height - fg.height))))


def add_outline(fg: Image.Image, rng: random.Random, bg01: np.ndarray) -> Image.Image:
    """Sticker-style ring: white (sometimes colored) fill behind a dilated alpha.

    The ring color must contrast with the ground: a white ring composited onto a white ground
    has an invisible boundary while the label says "keep the ring" — an unlearnable sample that
    teaches the model to keep arbitrary ground-colored regions hugging art. Candidates are tried
    in order and the first that clearly separates from the ground wins.
    """
    k = rng.choice([5, 9, 13, 17])
    a = fg.getchannel("A").filter(ImageFilter.MaxFilter(k))
    candidates = [(255, 255, 255), (0, 0, 0),
                  tuple(rng.randint(0, 255) for _ in range(3))]
    if rng.random() < 0.2:
        candidates.reverse()
    color = next(c for c in candidates
                 if np.linalg.norm(np.array(c, np.float32) / 255.0 - bg01) / np.sqrt(3.0) >= 0.22)
    pad = k
    base = Image.new("RGBA", (fg.width + 2 * pad, fg.height + 2 * pad), (0, 0, 0, 0))
    ring = Image.new("RGBA", fg.size, color + (255,))
    base.paste(ring, (pad, pad), a)
    base.alpha_composite(fg, (pad, pad))
    return base


def add_distress(fg: Image.Image, rng: random.Random) -> Image.Image:
    """Punch grunge holes INTO the art (noise → blur → threshold → morphology).

    The holes land in the GT alpha, so the label says drop them — teaching that a solid region
    broken up by intentional distress SHOULD be broken up. With the mottle loss (excess-vs-GT
    transitions), this is what separates intended distress from an unintended patchy cut.
    """
    a = np.asarray(fg.getchannel("A"), np.float32) / 255.0
    noise = np.random.rand(*a.shape).astype(np.float32)
    blur = np.asarray(Image.fromarray((noise * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(rng.uniform(1.0, 3.0))), np.float32) / 255.0
    keep = (blur > rng.uniform(0.42, 0.60)).astype(np.float32)
    if rng.random() < 0.5:  # speckle vs cracks
        keep = np.asarray(Image.fromarray((keep * 255).astype(np.uint8)).filter(
            ImageFilter.MinFilter(3)), np.float32) / 255.0
    a2 = (a * keep * 255).astype(np.uint8)
    out = fg.copy()
    out.putalpha(Image.fromarray(a2, "L"))
    return out


def wrap_in_badge(fg: Image.Image, rng: random.Random, bg01: np.ndarray) -> Image.Image:
    """Enclose the art in a solid plate (rounded rect or ellipse), contrast-guarded vs ground.

    GT alpha = the whole plate silhouette, so the ground-colored counters between art and plate
    edge must be KEPT — the badge/card 'keep the enclosed region' case the pool never had.
    """
    from PIL import ImageDraw

    pad = round(max(fg.size) * rng.uniform(0.10, 0.22))
    W, H = fg.width + 2 * pad, fg.height + 2 * pad
    candidates = [(255, 255, 255), (0, 0, 0), tuple(rng.randint(0, 255) for _ in range(3))]
    if rng.random() < 0.3:
        candidates.reverse()
    color = next((c for c in candidates
                  if np.linalg.norm(np.array(c, np.float32) / 255.0 - bg01) / np.sqrt(3.0) >= 0.22),
                 candidates[0])
    plate = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(plate)
    if rng.random() < 0.5:
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=round(min(W, H) * 0.12),
                            fill=color + (255,))
    else:
        d.ellipse([0, 0, W - 1, H - 1], fill=color + (255,))
    plate.alpha_composite(fg, (pad, pad))
    return plate


def wrap_in_scene_block(fg: Image.Image, rng: random.Random, bg01: np.ndarray) -> Image.Image:
    """Set the art on a hard-edged rectangular scene (a photo/poster block); GT keeps the block.

    The block is a low-frequency cloud field in two or three colours with patches that run
    within the keyer's tolerance of the ground — deliberately at the block's edge too — and a
    sprinkle of bright speckles. A bordered scene whose interior touches the ground colour is
    the class where keying the outer ground bites into the block; the rule is the same as for
    badges and pockets: everything bounded by ink is kept, so the alpha is the whole rectangle.
    """
    pad = round(max(fg.size) * rng.uniform(0.15, 0.35))
    W, H = fg.width + 2 * pad, fg.height + 2 * pad
    ground = np.asarray(bg01, np.float32)
    # two or three cloud colours; one of them is a near-ground tone so the interior meets the
    # ground somewhere, the others must contrast with it
    def far_color() -> np.ndarray:
        for _ in range(20):
            c = np.array([rng.random() for _ in range(3)], np.float32)
            if np.linalg.norm(c - ground) / np.sqrt(3.0) >= 0.25:
                return c
        return 1.0 - ground
    near = np.clip(ground + np.array([rng.uniform(-0.04, 0.04) for _ in range(3)], np.float32), 0, 1)
    colors = [far_color(), far_color() if rng.random() < 0.6 else near, near]
    rng.shuffle(colors)
    fields = [_low_freq_shading(H, W, 1.0, rng)[..., 0] for _ in colors]
    stack = np.stack(fields, 0)
    stack = stack - stack.min(axis=0, keepdims=True)
    weights = stack / (stack.sum(axis=0, keepdims=True) + 1e-6)
    scene = sum(weights[i][..., None] * colors[i][None, None, :] for i in range(len(colors)))
    # a band along one or two edges pulled toward the ground: the bite class
    band = np.zeros((H, W), np.float32)
    bw = max(2, round(min(W, H) * rng.uniform(0.04, 0.12)))
    for side in rng.sample(("top", "bottom", "left", "right"), rng.randint(1, 2)):
        if side == "top":
            band[:bw, :] = 1.0
        elif side == "bottom":
            band[-bw:, :] = 1.0
        elif side == "left":
            band[:, :bw] = 1.0
        else:
            band[:, -bw:] = 1.0
    band = band * _low_freq_shading(H, W, 1.0, rng)[..., 0].clip(0, 1)
    scene = scene * (1 - band[..., None]) + near[None, None, :] * band[..., None]
    rgb = (np.clip(scene, 0, 1) * 255).round().astype(np.uint8)
    # bright speckles (stars / grain) so the block reads as a photo, not a flat plate
    n_spk = rng.randint(0, max(1, W * H // 4000))
    ys = np.random.randint(0, H, n_spk)
    xs = np.random.randint(0, W, n_spk)
    rgb[ys, xs] = 255 if rng.random() < 0.7 else rng.randint(160, 255)
    block = Image.fromarray(np.dstack([rgb, np.full((H, W), 255, np.uint8)]), "RGBA")
    block.alpha_composite(fg, (pad, pad))
    return block


def _block_glyph(rng: random.Random, h: int) -> Image.Image:
    """One blocky letterform mask: a few bars, sometimes a counter-bearing frame."""
    w = round(h * rng.uniform(0.55, 0.85))
    m = Image.new("L", (w, h), 0)
    from PIL import ImageDraw

    d = ImageDraw.Draw(m)
    t = max(3, round(h * rng.uniform(0.16, 0.26)))  # stroke thickness
    if rng.random() < 0.35:
        # frame glyph (O/B/D-like): encloses a counter the GT drops, like a real letter
        d.rectangle([0, 0, w - 1, h - 1], outline=255, width=t)
    else:
        # bar glyph (H/I/L/T/E-like)
        d.rectangle([0, 0, t, h - 1], fill=255)
        if rng.random() < 0.7:
            d.rectangle([w - 1 - t, 0, w - 1, h - 1], fill=255)
        for yfrac in rng.sample([0.0, 0.45, 1.0], k=rng.randint(1, 2)):
            y = round(yfrac * (h - 1 - t))
            d.rectangle([0, y, w - 1, y + t], fill=255)
    return m


def make_inline_lettering(rng: random.Random, bg01: np.ndarray) -> Image.Image:
    """Collegiate lettering whose inline stroke is EXACTLY the ground color.

    Structure per word: colored glyph fill -> ground-colored inline band -> colored keyline.
    GT alpha keeps the whole outer silhouette INCLUDING glyph counters: enclosed ground
    bounded by ink is authored ink — only distress-scale speckle is dropped, and
    add_distress teaches that pole. The inline is learnable despite matching the ground because it is
    sandwiched between two ink strokes: both boundaries are observable (unlike a ring
    dissolving into open ground). The model split these bands patchily instead of
    keeping them.
    """
    h = rng.randint(48, 96)
    gap = max(2, h // 10)
    glyphs = [_block_glyph(rng, h) for _ in range(rng.randint(3, 7))]
    W = sum(g.width for g in glyphs) + gap * (len(glyphs) + 1)
    word = Image.new("L", (W, h + 2 * gap), 0)
    x = gap
    for g in glyphs:
        word.paste(g, (x, gap), g)
        x += g.width + gap
    k1 = rng.choice([4, 6, 8])    # inline width — survives the canvas-fit downscale
    k2 = rng.choice([4, 6])       # keyline width
    pad = k1 + k2 + 2
    m0 = Image.new("L", (word.width + 2 * pad, word.height + 2 * pad), 0)
    m0.paste(word, (pad, pad))
    m1 = m0.filter(ImageFilter.MaxFilter(2 * k1 + 1))
    m2 = m1.filter(ImageFilter.MaxFilter(2 * k2 + 1))
    a0, a1, a2 = (np.asarray(m, np.float32) / 255.0 > 0.5 for m in (m0, m1, m2))

    def _contrast(cands: list[tuple[int, int, int]]) -> tuple[int, int, int]:
        # every candidate can sit within tolerance of a dark or mid ground: fall back to the
        # ground's complement rather than raising out of the pool build
        fallback = tuple(int(round((1.0 - v) * 255)) for v in bg01)
        return next((c for c in cands
                     if np.linalg.norm(np.array(c, np.float32) / 255.0 - bg01) / np.sqrt(3.0) >= 0.22),
                    fallback)

    fill = _contrast([tuple(rng.randint(0, 255) for _ in range(3)), (20, 60, 90), (0, 0, 0)])
    keyline = _contrast([tuple(rng.randint(0, 255) for _ in range(3)), (200, 160, 40), (0, 0, 0)])
    ground = tuple(int(round(v * 255)) for v in bg01)
    rgb = np.zeros((*a0.shape, 3), np.uint8)
    rgb[a2 & ~a1] = keyline
    rgb[a1 & ~a0] = ground
    rgb[a0] = fill
    # counters render as ground but stay KEPT: enclosed-by-ink ground is authored ink
    from scipy import ndimage

    filled = ndimage.binary_fill_holes(a0)
    rgb[filled & ~a0] = ground
    alpha = a2 | filled
    out = Image.fromarray(rgb, "RGB").convert("RGBA")
    out.putalpha(Image.fromarray((alpha * 255).astype(np.uint8), "L"))
    return out


def add_interior_pocket(fg: Image.Image, rng: random.Random, bg01: np.ndarray) -> Image.Image:
    """Paint 1-2 ground-colored pockets deep INSIDE the art; GT keeps them (they are ink).

    The blaze/teeth/eye-white class: art-interior regions that happen to match the ground
    must survive the key. Pockets sit well inside the eroded alpha so enclosure — not
    color — is the learnable signal. Distinct from add_distress (small scattered holes the
    GT drops): pockets are few, larger, and deep interior.
    """
    from PIL import ImageDraw
    from scipy import ndimage

    a = np.asarray(fg.getchannel("A"), np.float32) / 255.0 > 0.9
    m = min(fg.size)
    r = rng.randint(max(3, m // 30), max(5, m // 14))
    # the whole pocket must sit inside solidly-opaque art with a margin — erode by the
    # pocket radius plus a buffer before picking a center
    core = ndimage.binary_erosion(a, iterations=r + max(4, m // 32), border_value=0)
    ys, xs = np.where(core)
    if ys.size < 64:
        return fg
    out = fg.copy()
    d = ImageDraw.Draw(out)
    ground = tuple(int(round(v * 255)) for v in bg01)
    for _ in range(rng.randint(1, 2)):
        i = rng.randrange(ys.size)
        cy, cx = int(ys[i]), int(xs[i])
        if rng.random() < 0.5:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ground + (255,))
        else:
            pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
            d.polygon(pts, fill=ground + (255,))
    # pockets must stay strictly interior: restore the original alpha
    out.putalpha(fg.getchannel("A"))
    return out


def dominant_color_hex(fg: Image.Image) -> str:
    small = fg.copy()
    small.thumbnail((64, 64))
    arr = np.asarray(small, np.float32)
    opaque = arr[:, :, 3] > 128
    if opaque.sum() < 10:
        return "#808080"
    med = np.median(arr[opaque][:, :3], axis=0).round().astype(int)
    return "#%02x%02x%02x" % tuple(med)


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def compose_sample(fgs: list[Image.Image], bg_hex: str, cfg: DriftConfig,
                   rng: random.Random) -> tuple[Image.Image, Image.Image, int]:
    ss, w, h = 2, cfg.max_side, cfg.max_side
    art = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    if rng.random() < cfg.dense_prob and len(fgs) >= 3:
        n = rng.randint(3, min(cfg.dense_max, len(fgs)))
    else:
        n = 1 + (1 if rng.random() < cfg.multi_element_prob and len(fgs) > 1 else 0)
    for fg in fgs[:n]:
        placed = _fit(fg, (w * ss, h * ss), rng)
        if rng.random() < 0.5:
            placed = placed.rotate(rng.uniform(-8, 8), expand=True, resample=Image.BICUBIC)
        _paste(art, placed, rng)

    art_np = np.asarray(art.resize((w, h), Image.LANCZOS), np.float32) / 255.0
    fg_rgb, fg_a = art_np[:, :, :3], art_np[:, :, 3:4]

    base = hex_to_rgb01(bg_hex)
    drift = np.array([rng.uniform(-1, 1) for _ in range(3)], np.float32)
    drift = drift / (np.linalg.norm(drift) + 1e-6) * rng.uniform(0.0, cfg.max_tint_drift)
    ground = np.clip(base + drift, 0, 1)[None, None, :]
    ground = np.clip(ground * _low_freq_shading(h, w, cfg.shade_strength, rng), 0, 1)
    ground = np.broadcast_to(ground, (h, w, 3)).copy()

    if rng.random() < cfg.shadow_prob:
        off = rng.randint(2, 8)
        sh = Image.fromarray((fg_a[:, :, 0] * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(rng.uniform(3, 9)))
        sh = np.roll(np.roll(np.asarray(sh, np.float32) / 255.0, off, 0), off, 1)
        ground *= (1.0 - 0.35 * sh[:, :, None])

    comp = np.clip(fg_rgb * fg_a + ground * (1 - fg_a)
                   + np.random.normal(0, cfg.noise_sigma, (h, w, 3)), 0, 1)
    inp = Image.fromarray((comp * 255).round().astype(np.uint8), "RGB")
    if rng.random() < 0.4:
        inp = inp.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.0)))
    inp = _jpeg(inp, rng.randint(*cfg.jpeg_quality))

    alpha = Image.fromarray(np.where(fg_a[:, :, 0] >= 0.5, 255, 0).astype(np.uint8), "L")
    return inp, alpha, n


def load_foreground(path: Path, max_side: int) -> Image.Image | None:
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:
        return None
    frac = float((np.asarray(im.getchannel("A")) > 16).mean())
    if not (0.02 <= frac <= 0.95):
        return None
    if max(im.size) > max_side * 2:
        s = max_side * 2 / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def build(fg_dir: Path, out_dir: Path, n_samples: int, seed: int, max_side: int = 512) -> int:
    cfg = DriftConfig(max_side=max_side)
    rng = random.Random(seed)
    np.random.seed(seed)
    fgs = sorted(fg_dir.rglob("*.png"))
    if not fgs:
        raise SystemExit(f"no PNG foregrounds under {fg_dir}")
    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    (out_dir / "alpha").mkdir(parents=True, exist_ok=True)
    manifest = (out_dir / "manifest.jsonl").open("w")
    written, attempts = 0, 0
    while written < n_samples and attempts < n_samples * 20:
        attempts += 1
        picks = [p for p in (load_foreground(rng.choice(fgs), cfg.max_side)
                             for _ in range(cfg.dense_max)) if p]
        if not picks:
            continue
        # Ground first, from the RAW art (a ring would pollute the dominant color), then outline
        # with a ring color guaranteed to contrast with that ground.
        collision = rng.random() < cfg.collision_prob
        if collision:
            # nudge off the exact art color: zero separation is unlearnable label noise
            base = hex_to_rgb01(dominant_color_hex(picks[0]))
            off = np.array([rng.uniform(-1, 1) for _ in range(3)], np.float32)
            off = off / (np.linalg.norm(off) + 1e-6) * rng.uniform(0.03, 0.08)
            nudged = np.clip(base + off, 0, 1)
            bg_hex = "#%02x%02x%02x" % tuple((nudged * 255).round().astype(int))
            sample_cfg = replace(cfg, max_tint_drift=cfg.collision_drift)
        else:
            bg_hex = sample_base_hex(rng)
            sample_cfg = cfg
        bg01 = hex_to_rgb01(bg_hex)
        picks = [add_distress(p, rng) if rng.random() < cfg.distress_prob else p for p in picks]
        picks = [add_interior_pocket(p, rng, bg01) if rng.random() < cfg.pocket_prob else p
                 for p in picks]
        if rng.random() < cfg.inline_prob:
            picks.append(make_inline_lettering(rng, bg01))
        ops: list[str] = []
        wrapped = []
        for p in picks:
            roll = rng.random()
            if roll < cfg.scene_prob:
                wrapped.append(wrap_in_scene_block(p, rng, bg01))
                ops.append("scene")
            elif roll < cfg.scene_prob + cfg.badge_prob:
                wrapped.append(wrap_in_badge(p, rng, bg01))
                ops.append("badge")
            elif rng.random() < cfg.outline_prob:
                wrapped.append(add_outline(p, rng, bg01))
                ops.append("outline")
            else:
                wrapped.append(p)
        picks = wrapped
        inp, alpha, n = compose_sample(picks, bg_hex, sample_cfg, rng)
        sid = f"{written:07d}"
        inp.save(out_dir / "input" / f"{sid}.png")
        alpha.save(out_dir / "alpha" / f"{sid}.png")
        manifest.write(json.dumps({"id": sid, "input": f"input/{sid}.png",
                                   "alpha": f"alpha/{sid}.png", "bg_hex": bg_hex,
                                   "n_elements": n, "collision": collision,
                                   "ops": ops}) + "\n")
        written += 1
    manifest.close()
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fg-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-side", type=int, default=512)
    args = p.parse_args()
    n = build(args.fg_dir, args.out_dir, args.n, args.seed, args.max_side)
    print(f"wrote {n} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
