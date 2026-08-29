"""Matting loss: boundary-weighted BCE + L1 + Laplacian-pyramid (edge sharpness).

Contour pixels are upweighted so the model learns crisp edges — the v1 gap. The band is a
morphological gradient of the target (dilate − erode), computed with max-pool so it stays on-device.

v12 doctrine terms (opt-in when rgb/dist/garment are supplied): visibility weighting (errors
visible on the garment cost more), fringe penalty (kept ground-colored contour), and
decisiveness+mottle (commit to whole-region keep/drop in the flat ground zone, penalizing only
excess transitions vs GT so intentional distress is not punished). Constants mirror gate.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# calibration constants, kept in sync with knockout/gate.py
GROUND_TOL = 0.14
EDGE_TAU = 0.045
VIS_TOL = 0.12


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0, 1)
    return t * t * (3 - 2 * t)


def _grad_mag(rgb: torch.Tensor) -> torch.Tensor:
    """Per-channel finite-difference gradient magnitude, max over channels (B,1,H,W)."""
    dx = F.pad((rgb[:, :, :, 1:] - rgb[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((rgb[:, :, 1:, :] - rgb[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return (dx + dy).amax(dim=1, keepdim=True)


def _tv(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


def _gauss_kernel(device: torch.device) -> torch.Tensor:
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device)
    k = k[:, None] * k[None, :]
    return (k / k.sum())[None, None]


def _pyramid(x: torch.Tensor, kernel: torch.Tensor, levels: int = 4) -> list[torch.Tensor]:
    out, cur = [], x
    for _ in range(levels):
        blur = F.conv2d(F.pad(cur, (2, 2, 2, 2), mode="reflect"), kernel)
        down = blur[:, :, ::2, ::2]
        up = F.interpolate(down, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        out.append(cur - up)
        cur = down
    return out


def _boundary_band(target: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Contour region of the target: dilate − erode, in [0,1]."""
    dil = F.max_pool2d(target, k, 1, k // 2)
    ero = -F.max_pool2d(-target, k, 1, k // 2)
    return (dil - ero).clamp(0, 1)


def matte_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_w: float = 4.0,
    fg_weight: float = 1.2,
    rgb: torch.Tensor | None = None,
    dist: torch.Tensor | None = None,
    garment: torch.Tensor | None = None,
    lam_v: float = 1.0,
    lam_f: float = 0.5,
    lam_d: float = 0.5,
    lam_m: float = 0.5,
) -> torch.Tensor:
    """Boundary-weighted, over-crop-penalized matte loss.

    ``fg_weight`` (BCE pos_weight) makes missing true foreground — dropping real art,
    the over-crop failure — cost more than leaving background. Moderate; too high just
    trades over-crop for halo (fg_weight=1.6 measurably dropped gate-pass, so kept near 1).

    When ``rgb`` (B3HW), ``dist`` (B1HW shaded color-distance channel), and ``garment`` (B3)
    are supplied, the v12 doctrine terms are added.
    """
    pred = torch.sigmoid(logits)
    w = 1.0 + boundary_w * _boundary_band(target)

    if rgb is not None and garment is not None:
        # visibility weight: emphasize errors that differ from the garment (visible on it),
        # de-emphasize garment-toned ones (invisible — fabric substitutes).
        flat = ((rgb - garment[:, :, None, None]) ** 2).sum(1, keepdim=True).sqrt() / (3 ** 0.5)
        w = w * (1.0 + lam_v * _smoothstep(flat / VIS_TOL))

    pos = torch.tensor(fg_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=w, pos_weight=pos)
    diff = pred - target
    l1 = (w * torch.where(diff < 0, -fg_weight * diff, diff)).mean()
    k = _gauss_kernel(logits.device)
    lap = sum(F.l1_loss(p, t) for p, t in zip(_pyramid(pred, k), _pyramid(target, k), strict=True))
    loss = bce + l1 + 0.5 * lap

    if dist is not None:
        ground = (dist < GROUND_TOL).float()
        # fringe: penalize keeping ground-colored pixels on the GT-background side of the contour
        band = _boundary_band(target)
        fringe = (band * (1 - target) * ground * pred).mean()
        loss = loss + lam_f * fringe
        if rgb is not None:
            flat_zone = ground * (_grad_mag(rgb) < EDGE_TAU).float()
            # decisiveness: no mid-confidence dither where the decision is invisible
            decis = (flat_zone * pred * (1 - pred)).mean()
            # mottle: only EXCESS transitions vs GT (intentional distress has GT transitions too)
            mottle = (flat_zone * (_tv(pred) - _tv(target)).clamp(min=0)).mean()
            loss = loss + lam_d * decis + lam_m * mottle
    return loss
