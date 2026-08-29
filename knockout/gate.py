"""Deterministic no-GT quality gate: ship the model's cutout, or fall back to the cascade?

The gate judges in garment-composite space — the only space the customer sees. A knockout is
married to one garment color, so a defect exists only if it is visible ON that garment:
ground-colored halo, remnants, or dropped ground-colored art all print (or expose fabric)
in the garment's own color and are invisible by construction. What can hurt:

- HALO-VISIBLE (under-crop): ground-colored contour off real edges that is ALSO visibly
  off-garment — only possible when the ground drifted away from the requested garment color.
  Raw halo is kept as telemetry; it no longer benches on its own.
- OVER-CROP (dropped art): removed pixels whose color visibly differs from the garment.
  Dropping garment-colored art (white fur on a white tee) is free — fabric substitutes.
- REMNANT / PLATE (kept background): any non-tiny connected component whose own contour has
  little edge support — a floating scrap of garment texture, or the whole canvas kept. This is
  the signal that catches *textured* remnants (photo shading prints as visibly-off-flat ink)
  and it is unchanged. Scored per component so one bad scrap can't be averaged away.
- INK BUDGET: cap on flat garment-colored kept area. Color-invisible, but it prints ink; the
  cap only exists to stop pathological keep-everything alphas that merge into art components.

accept = no visible halo AND little visible ink dropped AND no unsupported component AND
gratuitous ink under budget. Thresholds are calibrated against ground truth on held-out
synthetic (see tune_gate.py) with usability defined in the same composite space, then frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .metrics import color_distance, flat_distance

# Calibrated on held-out synthetic (tune_gate.py); frozen for real-data application.
GROUND_TOL = 0.14     # contour pixel this close to the ground segment = ground-colored
EDGE_TAU = 0.045      # per-channel gradient above this = a real image edge
HALO_MAX = 0.06       # max fraction of contour that may be *visible* (off-garment) halo
VIS_TOL = 0.12        # flat distance to garment above this = customer-visible on the garment
INK_TOL = 0.30        # shaded-segment distance above this = unambiguous ink (never the ground)
DROPPED_VIS_MAX = 0.04  # max fraction of visible-ink pixels the model may drop
COMP_MIN_FRAC = 0.0015  # components smaller than this fraction of the frame are ignored
COMP_EDGE_MIN = 0.55    # a scored component needs at least this contour edge-support to be art
INK_BUDGET_MAX = 0.50   # max fraction of frame kept as flat garment-colored (gratuitous) ink
ELEMENT_MIN_FRAC = 0.0005  # visible-ink components this large are design elements in their own right
ELEMENT_DROP_MAX = 0.35    # max fraction of any single element the model may drop


@dataclass
class GateVerdict:
    accept: bool
    reason: str
    signals: dict


def _channel_grad(rgb01: np.ndarray, tol_px: int = 7) -> np.ndarray:
    grads = [np.hypot(*np.gradient(rgb01[:, :, c])) for c in range(3)]
    return ndimage.maximum_filter(np.max(grads, axis=0), size=tol_px)


def gate(
    alpha01: np.ndarray, rgb01: np.ndarray, bg01: np.ndarray,
    garment01: np.ndarray | None = None,
) -> GateVerdict:
    """bg01 is the ground actually keyed on (drift-aware); garment01 the requested garment
    color the print will live on. They usually coincide; under drift they differ and the
    visibility signals tighten against the garment, not the drifted ground."""
    if garment01 is None:
        garment01 = bg01
    a = alpha01 >= 0.5
    if not a.any():
        return GateVerdict(False, "empty cutout (total over-crop)",
                           {"halo": 0.0, "halo_visible": 0.0, "dropped_ink": 1.0,
                            "worst_elem_drop": 1.0, "bad_comp_frac": 0.0,
                            "worst_comp_support": 0.0, "kept_frac": 0.0,
                            "gratuitous_ink": 0.0, "mottle": 0.0})
    grad = _channel_grad(rgb01)
    dist = color_distance(rgb01, bg01)
    vis = flat_distance(rgb01, garment01)

    contour = a & ~ndimage.binary_erosion(a, border_value=0)
    nc = int(contour.sum())
    halo_px = contour & (dist < GROUND_TOL) & (grad < EDGE_TAU)
    # A large SOLID garment-colored region is a design fill (a wide cream stripe, a navy badge
    # ring), not retained ground — its border is the design's own edge, not a halo. Exclude
    # those borders; a thin retained-ground fringe erodes away and still counts as halo.
    kept_ground = a & (dist < GROUND_TOL) & (grad < EDGE_TAU)
    design_fill = ndimage.binary_dilation(
        ndimage.binary_erosion(kept_ground, iterations=6, border_value=0), iterations=8)
    halo_px = halo_px & ~design_fill
    halo = float(halo_px.sum()) / max(nc, 1)
    halo_visible = float((halo_px & (vis > VIS_TOL)).sum()) / max(nc, 1)

    # Ink-ness is shaded-segment distance (shaded ground is ground, not ink); visibility is
    # flat distance (it must also differ from the garment to be missed). Erode before scoring:
    # the 1-2px anti-aliased fringe around glyph and art edges is ink-colored but its removal
    # is correct edge behavior, and on large type it adds up to thousands of "dropped" pixels —
    # benching perfect cutouts. A genuinely dropped stroke or region still has interior mass.
    ink = ndimage.binary_erosion((dist > INK_TOL) & (vis > VIS_TOL),
                                 iterations=2, border_value=0)
    dropped_ink = float((ink & ~a).sum()) / max(int(ink.sum()), 1)

    # Per-element over-crop: a global fraction lets one destroyed element hide inside a dense
    # design. Every visible-ink component of element scale is scored on its own.
    worst_elem_drop = 0.0
    ilabels, ni = ndimage.label(ink)
    if ni:
        iidx = np.arange(1, ni + 1)
        iareas = np.bincount(ilabels.ravel(), minlength=ni + 1)[1:]
        dropped_areas = ndimage.sum_labels(~a, ilabels, index=iidx)
        big = iareas >= ELEMENT_MIN_FRAC * a.size
        if big.any():
            worst_elem_drop = float((dropped_areas[big] / iareas[big]).max())

    gratuitous = float((a & (vis < VIS_TOL) & (grad < EDGE_TAU)).sum()) / a.size

    # Mottle telemetry: density of alpha transitions inside the garment-toned flat zone
    # (garment-colored pixels off any real edge). High = a salt-and-pepper keep/drop mix that
    # prints ink and bare fabric interleaved in one region. It does NOT bench: intentional
    # distress texture is geometrically identical, so the defect-vs-intent split needs the
    # model, not a threshold. Logged for the learn-loop and to supervise v12 training.
    flat_ground = (dist < GROUND_TOL) & (grad < EDGE_TAU)
    zone_area = int(flat_ground.sum())
    if zone_area >= 64:
        trans = ndimage.binary_dilation(a) & ~ndimage.binary_erosion(a, border_value=0)
        mottle = float((trans & flat_ground).sum()) / zone_area
    else:
        mottle = 0.0

    labels, n = ndimage.label(a)
    min_area = COMP_MIN_FRAC * a.size
    worst_comp = 1.0
    bad_comp_frac = 0.0
    if n:
        for lab in range(1, n + 1):
            comp = labels == lab
            area = int(comp.sum())
            if area < min_area:
                continue
            cc = comp & ~ndimage.binary_erosion(comp, border_value=0)
            sup = float((grad[cc] >= EDGE_TAU).mean()) if cc.any() else 0.0
            worst_comp = min(worst_comp, sup)
            if sup < COMP_EDGE_MIN:
                bad_comp_frac += area / a.size

    signals = {"halo": round(halo, 4), "halo_visible": round(halo_visible, 4),
               "dropped_ink": round(dropped_ink, 4),
               "worst_elem_drop": round(worst_elem_drop, 4),
               "worst_comp_support": round(worst_comp, 4),
               "bad_comp_frac": round(bad_comp_frac, 4),
               "kept_frac": round(float(a.mean()), 4),
               "gratuitous_ink": round(gratuitous, 4),
               "mottle": round(mottle, 4)}

    if halo_visible > HALO_MAX:
        return GateVerdict(False, "halo: ground-colored contour visible on garment", signals)
    if dropped_ink > DROPPED_VIS_MAX:
        return GateVerdict(False, "over-crop: dropped garment-visible ink", signals)
    if worst_elem_drop > ELEMENT_DROP_MAX:
        return GateVerdict(False, "over-crop: design element destroyed", signals)
    if bad_comp_frac > 0.0:
        return GateVerdict(False, "remnant/plate: kept component without edge support", signals)
    if gratuitous > INK_BUDGET_MAX:
        return GateVerdict(False, "ink budget: gratuitous garment-colored coverage", signals)
    return GateVerdict(True, "clean", signals)
