"""Morphological cleanup on the final binary alpha.

Two distinct jobs that must NOT share a threshold:

- **Despeckle** (this module, small default): drop 1-2px isolated noise and fill 1-2px pinholes
  — genuine artifacts at any resolution. Safe for small text.
- **Minimum printable feature** (~1mm: dot gain fills tiny holes, the choked white underbase
  halos on misregistration): a PRINT-resolution rule, `140 px^2` at 300 DPI. It MUST be applied
  on the full-size print file, never here — at the model's 512px working res or a ~1000px design
  res, `1mm` is only 3-12 px^2, so a print-scale threshold shreds small text and fine strokes
  into disconnected fragments. Defer it to the print-res stage.

Color-aware despeckle: an isolated speck that is *the ground color* is a leftover of the removed
background — dropping it is chromatically lossless (fabric is a perfect ink for the garment color),
so it can be cleared at a generous size. An *art-colored* speck may be intentional (distress dots,
a tittle), so it keeps the tiny 1-2px floor. Pass rgb01 + bg01 to enable this; without them the
module falls back to pure size despeckle.

Run after ``refine_alpha``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .metrics import color_distance, flat_distance

# Ink kept in a free region (flat, ground-colored, invisible on the garment) survives only at
# this model confidence. Below it, kept pixels are indecision dither — patchy ink/fabric mixes
# that print as two different materials inside one region. Deliberate collision art (white-on-white
# glyphs, black-on-black fills) sits well above it.
HYGIENE_CONF_KEEP = 0.75


def cleanup_alpha(
    alpha01: np.ndarray,
    rgb01: np.ndarray | None = None,
    bg01: np.ndarray | None = None,
    min_hole_area: int = 4,
    min_island_area: int = 4,
    ground_tol: float = 0.08,
    ground_area_frac: float = 0.0001,
    ink_hole_tol: float = 0.25,
    conf01: np.ndarray | None = None,
    garment01: np.ndarray | None = None,
) -> np.ndarray:
    """Despeckle + repair provably-wrong regions using the known ground color.

    `min_island_area` (~4px^2) removes only 1-2px noise of any color — small enough to preserve
    small-text punctuation. When rgb01+bg01 are given, two color-keyed repairs are added:

    - **remnant chips**: an isolated island up to `ground_area_frac` of the frame whose color is
      within `ground_tol` of the ground segment is leftover background — dropped. Clearing it is
      chromatically lossless on-fabric; art-colored specks are untouched. The cap is deliberately
      tiny: at any larger setting this rule eats ultra-collision text glyphs, which size+color
      cannot distinguish from chips (measured: every larger cap traded 3-4x more text over-crop
      than chip removal gained). Larger kept chips are garment-colored — sheen-only on product —
      and remain a model-quality target, not a post-op one.
    - **wrong holes**: a knockout hole can only legitimately exist where the *input* was
      ground-colored (a hole shows fabric where the render showed garment color). An enclosed
      transparent region whose input pixels average farther than `ink_hole_tol` from the ground
      sits on artwork — a hole punched through art — and is filled, whatever its size.

    Holes/regions connected to the canvas border are never filled (outside background).

    With `conf01` (the model's soft prediction) a material-hygiene pass runs first: in *free*
    regions — flat, ground-colored, and invisible on the garment — a kept pixel below
    ``HYGIENE_CONF_KEEP`` clears to fabric. Removal there is chromatically lossless, and it is
    the doctrine direction: a patchy ink/fabric mix in one region prints as two materials.
    Confident kept regions (deliberate garment-colored art) are untouched.
    """
    a = alpha01 >= 0.5
    dist = color_distance(rgb01, bg01) if rgb01 is not None and bg01 is not None else None

    if conf01 is not None and dist is not None:
        from .gate import EDGE_TAU, GROUND_TOL, VIS_TOL, _channel_grad

        g = garment01 if garment01 is not None else bg01
        flat = _channel_grad(rgb01) < EDGE_TAU
        free = flat & (dist < GROUND_TOL) & (flat_distance(rgb01, g) < VIS_TOL)
        a &= ~(free & (conf01 < HYGIENE_CONF_KEEP))

        # Confident dust: a tiny all-free island far from any substantial art is leftover
        # ground, not a collision glyph (those cluster near the rest of the design).
        # Free ⇒ clearing is lossless.
        labels, n = ndimage.label(a)
        if n > 1:
            idx = np.arange(1, n + 1)
            areas = np.bincount(labels.ravel(), minlength=n + 1)
            substantial = areas >= 0.001 * a.size
            substantial[0] = False
            if substantial[1:].any():
                dt = ndimage.distance_transform_edt(~substantial[labels])
                min_gap = ndimage.minimum(dt, labels, index=idx)
                free_frac = ndimage.mean(free, labels, index=idx)
                diag = float(np.hypot(*a.shape))
                dust = ~substantial[1:] & (min_gap > 0.12 * diag) & (free_frac > 0.9)
                if dust.any():
                    kill = np.zeros(n + 1, bool)
                    kill[1:] = dust
                    a[kill[labels]] = False

    if min_island_area > 0:
        labels, n = ndimage.label(a)
        if n:
            areas = np.bincount(labels.ravel())
            drop = areas < min_island_area
            if dist is not None:
                ground_cap = max(min_island_area, int(ground_area_frac * a.size))
                mean_d = ndimage.mean(dist, labels, index=np.arange(1, n + 1))
                ground = np.zeros(n + 1, bool)
                ground[1:] = (areas[1:] < ground_cap) & (mean_d < ground_tol)
                drop |= ground
            drop[0] = False
            a[drop[labels]] = False

    if min_hole_area > 0:
        labels, n = ndimage.label(~a)
        if n:
            border = np.zeros_like(a)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            open_ids = np.unique(labels[border])
            areas = np.bincount(labels.ravel())
            fill = areas < min_hole_area
            if dist is not None:
                mean_d = ndimage.mean(dist, labels, index=np.arange(1, n + 1))
                ink_hole = np.zeros(n + 1, bool)
                ink_hole[1:] = mean_d > ink_hole_tol
                fill |= ink_hole
            fill[0] = False
            fill[open_ids] = False
            a[fill[labels]] = True

    return a.astype(np.float32)
