"""The remnant rule: a garment-toned island in open ground is benched; the same island enclosed
by the art is a pocket and passes; a pocket that would show on the requested garment is a plate."""

import numpy as np

from knockout.gate import gate
from knockout.metrics import hex_to_rgb01


def _scene(blob_inside: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = 256
    bg = hex_to_rgb01("#ffffff")
    rgb = np.ones((s, s, 3), np.float32)
    alpha = np.zeros((s, s), np.float32)
    ring = np.zeros((s, s), bool)
    ring[40:200, 40:200] = True
    ring[52:188, 52:188] = False
    rgb[ring] = [0.1, 0.1, 0.4]
    alpha[ring] = 1.0
    y0, x0 = (100, 100) if blob_inside else (215, 100)
    alpha[y0:y0 + 24, x0:x0 + 24] = 1.0  # a flat, ground-coloured kept blob
    return alpha, rgb, bg


def test_island_in_open_ground_is_a_remnant() -> None:
    alpha, rgb, bg = _scene(blob_inside=False)
    v = gate(alpha, rgb, bg)
    assert not v.accept and v.reason.startswith("remnant")


def test_enclosed_pocket_passes() -> None:
    alpha, rgb, bg = _scene(blob_inside=True)
    v = gate(alpha, rgb, bg)
    assert v.accept, v.reason


def test_pocket_visible_on_another_garment_stays_benched() -> None:
    alpha, rgb, bg = _scene(blob_inside=True)
    v = gate(alpha, rgb, bg, garment01=hex_to_rgb01("#141313"))
    assert not v.accept
