"""Deterministic matting metrics — the shared color-distance signal, the prod no-GT gate,
and ground-truth scores for training eval.

The color-distance function is the exact shaded-segment model from production's
``_remove_color_distance`` so the model's conditioning channel equals the keyer's signal,
and the no-GT gate (``boundary_ground_residue``) matches what will police it in prod.
"""

from __future__ import annotations

import numpy as np

SHADOW_K_MIN = 0.78
SHADOW_K_MAX = 1.06
RESIDUE_GROUND_TOL = 0.12
# Gate tolerance is deliberately wider than the refine tolerance: refine strips contour pixels
# with dist < RESIDUE_GROUND_TOL, so a gate at the same tol would be satisfied by construction.
# The margin lets the gate still catch near-ground halo that refine intentionally kept.
GATE_GROUND_TOL = 0.16
_ROOT3 = np.sqrt(3.0)


def hex_to_rgb01(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def color_distance(rgb01: np.ndarray, bg01: np.ndarray) -> np.ndarray:
    """Normalized distance from each pixel to the shaded-ground segment {k·bg}, in [0,~1].

    rgb01: HxWx3 float. Softly shaded background reads ~0; ink reads high. This is both the
    model's conditioning channel and the keyer's decision variable.
    """
    bg_dot = float(bg01 @ bg01) or 1e-6
    k = np.clip((rgb01 @ bg01) / bg_dot, SHADOW_K_MIN, SHADOW_K_MAX)
    return np.linalg.norm(rgb01 - k[..., None] * bg01[None, None, :], axis=-1) / _ROOT3


def flat_distance(rgb01: np.ndarray, color01: np.ndarray) -> np.ndarray:
    """Plain normalized distance to a flat color — no shadow slack.

    Visibility on a printed garment: fabric is one flat color, so ink at 0.8·garment reads as a
    visible darker patch. Use this for on-garment visibility; use ``color_distance`` (shaded
    segment) only for ground-ness.
    """
    return np.linalg.norm(rgb01 - color01[None, None, :], axis=-1) / _ROOT3


def visible_bad_frac(
    pred01: np.ndarray, gt01: np.ndarray, rgb01: np.ndarray, garment01: np.ndarray,
    vis_tol: float = 0.12, edge_slack_px: int = 2,
) -> float:
    """Fraction of frame where the pred composite visibly differs from the GT composite.

    The doctrine label: a knockout is married to one garment color, so the only defect that
    exists is one the customer can see on that garment. Alpha disagreement within
    ``edge_slack_px`` of the GT contour is AA/edge jitter, invisible at print scale — ignored.
    """
    from scipy import ndimage

    p = pred01 >= 0.5
    g = gt01 >= 0.5
    disagree = p ^ g
    if not disagree.any():
        return 0.0
    contour = g ^ ndimage.binary_erosion(g, border_value=0)
    edge_zone = ndimage.binary_dilation(contour, iterations=edge_slack_px)
    vis = flat_distance(rgb01, garment01) > vis_tol
    return float((disagree & ~edge_zone & vis).mean())


def boundary_ground_residue(
    alpha01: np.ndarray, rgb01: np.ndarray, bg01: np.ndarray, band_px: int = 3,
    tol: float = GATE_GROUND_TOL,
) -> float:
    """Fraction of opaque contour pixels still carrying the ground color (no GT needed).

    The prod no-op/gate signal: a rim of garment-colored ink along the art contour. 0 = clean.
    Run it *after* refine (the prod order); `tol` stays wider than refine's so it keeps teeth.

    This gate polices *under-crop* (halo) only. Over-crop — dropped artwork — is invisible to
    it, and on tonal-collision art invisible to any color-keyed signal; budget for that tail
    via the fallback path, not this gate.
    """
    opaque = alpha01 >= 0.5
    band = ~opaque
    if not band.any() or band.all():
        return 0.0
    for _ in range(band_px):
        grown = band.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    grown |= np.roll(np.roll(band, dy, 0), dx, 1)
        band = grown
    contour = opaque & band
    n = int(contour.sum())
    if not n:
        return 0.0
    d = color_distance(rgb01, bg01)
    return float((contour & (d < tol)).sum()) / n


def gt_scores(pred01: np.ndarray, gt01: np.ndarray) -> dict[str, float]:
    """Ground-truth training metrics against the true alpha."""
    p = pred01 >= 0.5
    g = gt01 >= 0.5
    inter = float((p & g).sum())
    union = float((p | g).sum()) or 1.0
    gt_n = float(g.sum()) or 1.0
    pred_n = float(p.sum()) or 1.0
    return {
        "iou": inter / union,
        "over_crop": float((g & ~p).sum()) / gt_n,   # true fg the model dropped
        "under_crop": float((p & ~g).sum()) / pred_n,  # bg the model kept
        "sad": float(np.abs(pred01 - gt01).sum()) / (gt01.size / 1000.0),  # per-1k-px
    }
