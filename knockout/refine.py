"""Deterministic edge refinement on the model's alpha — the cascade's decontamination idea.

The model matts regions well but leaves a garment-colored fringe along contours (opaque
pixels whose color still sits on the ground segment). That rim prints as ink and fails the
prod boundary-residue gate. This strips exactly those pixels: opaque, near the contour, and
ground-colored — the same trade the cascade's `_remove_color_distance` decontamination makes.
No model, no retrain.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .metrics import RESIDUE_GROUND_TOL, color_distance


def _neighbors_or(mask: np.ndarray) -> np.ndarray:
    g = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                g |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return g


def _dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    for _ in range(iters):
        mask = _neighbors_or(mask)
    return mask


def _erode(mask: np.ndarray, iters: int) -> np.ndarray:
    for _ in range(iters):
        mask = ~_neighbors_or(~mask)
    return mask


def refine_alpha(
    alpha01: np.ndarray,
    rgb01: np.ndarray,
    bg01: np.ndarray,
    band_px: int = 3,
    tol: float = RESIDUE_GROUND_TOL,
    choke: int = 0,
    max_loss: float = 0.15,
) -> np.ndarray:
    """Strip the ground-colored contour rim; optional 1px DTG choke. Returns binary alpha.

    Only opaque pixels that are BOTH within `band_px` of transparency AND within `tol` of the
    ground segment are removed — genuine art that reaches its edge in a non-ground color is
    kept. `choke` erodes uniformly (DTG ink spread) and can hurt thin strokes, so default off.

    `max_loss` protects tonal-collision art: on strokes near the ground color the whole stroke
    sits inside the contour band, so an unguarded strip erases entire words the model correctly
    kept. Any connected component that would lose more than this fraction of itself is left
    untouched — a rim strip trims edges, it must never consume the structure it trims.
    """
    a = alpha01 >= 0.5
    contour = _dilate(~a, band_px) & a
    d = color_distance(rgb01, bg01)
    strip = contour & (d < tol)
    if max_loss < 1.0:
        labels, n = ndimage.label(a)
        if n:
            comp = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
            lost = np.bincount(labels[strip].ravel(), minlength=n + 1).astype(np.float64)
            protect = (lost / np.maximum(comp, 1)) > max_loss
            protect[0] = False
            strip &= ~protect[labels]
    a = a & ~strip
    if choke:
        a = _erode(a, choke)
    return a.astype(np.float32)
