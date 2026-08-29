"""Upstream gate: knockout, full-bleed, or reject?

Run this *before* the matting model. It sorts an incoming image into one of three verdicts:

  knockout    art on a removable garment-color ground -> call the knockout model
  full_bleed  design owns the canvas edge-to-edge with no removable ground -> print as-is, no model
  reject      a garment mockup (a photo of a design ON a shirt), not a flat design -> drop or review

Two axes, not three peers. First: is this a flat design or a mockup photo (reject)? Then, given a
design: is there a removable ground (knockout) or not (full_bleed)? `ground_frac` is the primary
discriminator — full_bleed *requires* little removable ground, so a die-cut subject that reaches
the frame on a large removable ground is still a knockout. When unsure, default to knockout: the
damaging error is printing a knockout's garment-color background as a solid block, so full_bleed
and reject each demand positive evidence.

The strongest signal is not in the image — it's the **product/placement type**. If the pipeline
knows the order is all-over-print, never knock out; a standard DTG placement almost always does.
Pass `product_is_aop` when you have it; this module is the fallback. Reject needs three fabric cues
to co-fire (texture + monochrome + photographic) so busy digital art is not mistaken for a garment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# Border uniform below this std => a solid ground is present (a background to remove).
BORDER_UNIFORM_STD = 0.05
# More than this fraction of the border band being artwork => the design bleeds to the edges.
EDGE_COVER_FULLBLEED = 0.12
# Full-bleed requires removable ground below this fraction. Above it there IS a background to
# remove, so the design routes knockout however its art meets the edges (a die-cut subject that
# reaches the frame is still a knockout). Validated separator on real designs: genuine full-bleed
# clusters near 0.04, edge-touching knockouts near 0.47.
GROUND_FRAC_FULLBLEED = 0.15
# A garment mockup (a photo of a design ON a shirt) is caught only when ALL THREE fabric cues
# co-fire in the corners: high-freq weave texture, a plain *colored-fabric* saturation band, and
# soft photographic shading. The saturation BAND is the key: a design on pure black or white has
# ~0 corner saturation, so an upper-only bound wrongly caught dark ornate art; a real colored
# garment (navy, olive, sage) sits in a mid band. White/black-garment mockups fall through to
# knockout — safe, the downstream gate handles them.
FABRIC_TEXTURE = 2.0        # native-res corner high-freq energy (weave)
FABRIC_SAT_MIN = 0.04       # below this = a design's own black/white ground, not colored fabric
FABRIC_SAT_MAX = 0.15       # above this = saturated art, not a plain garment
FABRIC_PHOTO_MIN = 0.45     # soft-gradient fraction above this = photographic, not flat art

# Common chroma-key knockout backdrops (green/blue screen, magenta). A design die-cut on one of
# these reaches the edges but is still a knockout, not full-bleed.
CHROMA_KEYS = [(0, 177, 64), (0, 255, 0), (0, 71, 187), (0, 0, 255), (255, 0, 255)]
CHROMA_TOL = 0.22


@dataclass
class RouteDecision:
    verdict: str          # "knockout" | "full_bleed" | "reject"
    confidence: float     # 0..1
    reason: str
    signals: dict

    @property
    def remove(self) -> bool:
        """Backward-compatible: True only for a clean knockout."""
        return self.verdict == "knockout"


def _hex_to_rgb01(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def corner_texture(im: Image.Image, frac: float = 0.14) -> float:
    """Median high-frequency energy across the 4 corner patches, at NATIVE resolution.

    Flat/digital backgrounds score ~0; knit-fabric corners score clearly higher. Measured on the
    full-res image on purpose — downsampling blurs the weave away.
    """
    g = im.convert("L")
    w, h = g.size
    cw, ch = max(8, int(w * frac)), max(8, int(h * frac))
    boxes = [(0, 0, cw, ch), (w - cw, 0, w, ch), (0, h - ch, cw, h), (w - cw, h - ch, w, h)]
    vals = []
    for b in boxes:
        c = g.crop(b)
        hp = np.asarray(c, np.float32) - np.asarray(c.filter(ImageFilter.BoxBlur(3)), np.float32)
        vals.append(float(hp.std()))
    return float(np.median(vals))


def _mockup_signals(im: Image.Image, frac: float = 0.14) -> tuple[float, float]:
    """(corner_saturation, photo_score): the two cues that separate a fabric photo from art.

    Corner saturation ~0 on a plain (gray/white/black) garment; high on colorful art. photo_score
    is the fraction of the image with soft photographic shading — high on a photo, low on flat art.
    """
    a = np.asarray(im.convert("RGB"), np.float32) / 255.0
    h, w = a.shape[:2]
    cw, ch = max(8, int(w * frac)), max(8, int(h * frac))
    corners = [a[:ch, :cw], a[:ch, w - cw:], a[h - ch:, :cw], a[h - ch:, w - cw:]]
    sat = float(np.median([(c.max(2) - c.min(2)).mean() for c in corners]))
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    gy, gx = np.gradient(lum)
    g = np.hypot(gx, gy)
    photo = float(((g > 0.004) & (g < 0.05)).mean())
    return sat, photo


def _is_chroma(rgb01: np.ndarray) -> bool:
    return any(
        float(np.linalg.norm(rgb01 - np.array(c, np.float32) / 255.0)) < CHROMA_TOL
        for c in CHROMA_KEYS
    )


def should_remove_background(
    image: Image.Image | str,
    known_bg_hex: str | None = None,
    product_is_aop: bool | None = None,
    size: int = 256,
) -> RouteDecision:
    """Route an image to knockout / full_bleed / reject.

    `product_is_aop` wins outright when known. Otherwise geometry (border uniformity, edge coverage)
    is measured on a `size`-px thumbnail, while fabric texture is measured on the full-res image.
    """
    if product_is_aop is True:
        return RouteDecision("full_bleed", 1.0, "product is all-over-print", {})
    if product_is_aop is False and known_bg_hex is None:
        # Trusts the caller outright — skips mockup detection. Pass known_bg_hex as well if the
        # input might be a garment photo rather than a flat design.
        return RouteDecision("knockout", 1.0, "product is a standard placement", {})

    im = Image.open(image) if isinstance(image, str) else image
    im = im.convert("RGB")
    tex = corner_texture(im)  # native resolution — before any downsample
    r = np.asarray(im.resize((size, size)), np.float32) / 255.0

    b = 4
    ring = np.concatenate(
        [r[:b].reshape(-1, 3), r[-b:].reshape(-1, 3), r[:, :b].reshape(-1, 3), r[:, -b:].reshape(-1, 3)]
    )
    border_rgb = np.median(ring, 0)
    border_std = float(ring.std(0).mean())

    ref = _hex_to_rgb01(known_bg_hex) if known_bg_hex else border_rgb
    dist = np.linalg.norm(r - ref[None, None], axis=-1) / np.sqrt(3.0)
    ground = dist < 0.10
    ground_frac = float(ground.mean())

    band = np.zeros((size, size), bool)
    m = max(4, size // 16)
    band[:m] = band[-m:] = band[:, :m] = band[:, -m:] = True
    edge_cover = float((~ground)[band].mean())

    signals = dict(
        corner_texture=round(tex, 2),
        border_std=round(border_std, 3),
        edge_cover=round(edge_cover, 3),
        ground_frac=round(ground_frac, 3),
    )

    # Chroma-key backdrop => knockout regardless of edge coverage (die-cut reaching the edges).
    if _is_chroma(border_rgb) and border_std < BORDER_UNIFORM_STD:
        return RouteDecision("knockout", 0.9, "chroma-key backdrop", signals)

    # Full-bleed requires little removable ground. Edge coverage and a patterned border only
    # decide full-bleed once the ground is already gone; with substantial ground present, an
    # edge-touching subject is still a knockout. ground_frac gates the verdict. A full-frame
    # scene inside a thin solid frame therefore routes knockout — the keeper strips the frame
    # ring and keeps the scene, which the downstream quality gate polices.
    low_ground = ground_frac < GROUND_FRAC_FULLBLEED
    if low_ground and border_std >= BORDER_UNIFORM_STD:
        conf = min(1.0, 0.5 + 5 * (border_std - BORDER_UNIFORM_STD))
        return RouteDecision("full_bleed", round(conf, 2), "patterned border, no removable ground", signals)
    if low_ground and edge_cover >= EDGE_COVER_FULLBLEED:
        return RouteDecision("full_bleed", 0.7, "artwork reaches the edges, no removable ground", signals)

    # Substantial removable ground (or a clean solid margin): knockout, unless it's a mockup.
    # Reject only when all three fabric cues co-fire — texture alone is busy art, not a garment.
    # A full-frame photographic scene design still trips this (it is genuinely photographic and
    # colored); reject is the safe direction (review), so that residual FP is left as-is.
    if tex >= FABRIC_TEXTURE:
        sat, photo = _mockup_signals(im)
        signals["corner_sat"] = round(sat, 3)
        signals["photo"] = round(photo, 3)
        if FABRIC_SAT_MIN <= sat <= FABRIC_SAT_MAX and photo >= FABRIC_PHOTO_MIN:
            return RouteDecision(
                "reject", round(min(1.0, 0.5 + 0.1 * (tex - FABRIC_TEXTURE)), 2),
                "monochrome, photographic, textured corners — a garment mockup, not flat art",
                signals,
            )

    # Clean inset artwork on a solid garment-color ground => knockout.
    if known_bg_hex is not None:
        bg_match = float(np.linalg.norm(border_rgb - _hex_to_rgb01(known_bg_hex)) / np.sqrt(3.0))
        signals["bg_match"] = round(bg_match, 3)
        if bg_match > 0.25:
            return RouteDecision(
                "knockout", 0.55,
                "inset on a solid ground, but its color doesn't match the known garment — review",
                signals,
            )
        return RouteDecision("knockout", 0.9, "inset artwork on the known garment-color ground", signals)
    return RouteDecision("knockout", 0.75, "inset artwork on a solid garment-color ground", signals)
