"""Example garment-color distribution for synthetic compositing.

A representative apparel palette weighted the way real print-on-demand catalogs skew — heavy
white, a dense cluster of near-blacks, and a color tail. Replace the weights with your own
catalog's garment-color frequencies for best in-domain results.
"""

from __future__ import annotations

import random

# (hex, weight) — sampling weights over a representative apparel palette.
BASE_COLOR_WEIGHTS: list[tuple[str, int]] = [
    ("#ffffff", 15350),
    ("#000000", 199),
    ("#0b0b0b", 180),
    ("#0c0c0c", 159),
    ("#141313", 146),
    ("#1d50a4", 98),
    ("#050c1d", 88),
    ("#f3f3f3", 79),
    ("#0e0e0e", 73),
    ("#080808", 59),
    ("#beeaff", 55),
    ("#da0a1a", 54),
    ("#651d32", 52),
    ("#2c493a", 51),
    ("#f3d4e3", 50),
    ("#7e8560", 50),
    ("#e7d3b3", 48),
    ("#fff0dd", 47),
    ("#a1c5e1", 46),
    ("#24283b", 46),
]

# White dominates raw counts but is the easy case (art rarely white); damp it hard so the model
# sees plenty of the dark/color grounds where halos actually happen. v1 ran 0.15 (~60% white);
# 0.08 brings white to ~38% to train edges on colored grounds.
_WHITE_DAMP = 0.08


def _weights() -> tuple[list[str], list[float]]:
    hexes, weights = [], []
    for h, c in BASE_COLOR_WEIGHTS:
        hexes.append(h)
        weights.append(c * _WHITE_DAMP if h == "#ffffff" else float(c))
    return hexes, weights


def sample_base_hex(rng: random.Random) -> str:
    hexes, weights = _weights()
    return rng.choices(hexes, weights=weights, k=1)[0]
