"""Ground-coloured artwork must be indistinguishable from the open ground beside it: an inline
band, a pocket or a plate interior comes out of the same render pass as the field around it. A
tint gap there is a shortcut the model learns instead of enclosure."""

import random

import numpy as np
from PIL import Image
from scipy import ndimage

from knockout.metrics import hex_to_rgb01
from knockout.synth_composite import DriftConfig, compose_sample, make_inline_lettering

_CFG = DriftConfig(max_side=512, multi_element_prob=0.0, dense_prob=0.0, inline_prob=0.0,
                   pocket_prob=0.0, distress_prob=0.0, badge_prob=0.0, scene_prob=0.0,
                   outline_prob=0.0, collision_prob=0.0)


def _band_vs_ground(seed: int, bg_hex: str) -> tuple[float, float] | None:
    """Compose one inline element twice from the same rng state — once as is, once with its
    ground-coloured band tagged magenta — so the band can be located exactly in the output.
    Band pixels ≥2.5 px from any neighbour and per-channel medians keep anti-aliasing out."""
    base = hex_to_rgb01(bg_hex)
    rng = random.Random(seed)
    fg = make_inline_lettering(rng, base, near_fill_prob=0.0)
    state = rng.getstate()
    fgn = np.asarray(fg, np.float32) / 255.0
    band0 = (fgn[:, :, 3] > 0.99) & (np.linalg.norm(fgn[:, :, :3] - base, axis=-1) / np.sqrt(3.0) < 0.01)
    tagged = np.asarray(fg).copy()
    tagged[band0] = [255, 0, 255, 255]
    r1, r2 = random.Random(), random.Random()
    r1.setstate(state); r2.setstate(state)
    inp, alpha, _ = compose_sample([fg], bg_hex, _CFG, r1)
    tinp, _, _ = compose_sample([Image.fromarray(tagged, "RGBA")], bg_hex, _CFG, r2)
    img = np.asarray(inp.convert("RGB"), np.float32) / 255.0
    timg = np.asarray(tinp.convert("RGB"), np.float32) / 255.0
    band = (timg[:, :, 0] > 0.8) & (timg[:, :, 1] < 0.25) & (timg[:, :, 2] > 0.8)
    core = ndimage.distance_transform_edt(band) >= 2.5
    kept = np.asarray(alpha) > 127
    ring = ndimage.binary_dilation(band, iterations=10) & ~ndimage.binary_dilation(kept, iterations=3)
    if core.sum() < 50 or ring.sum() < 50:
        return None
    b, r = np.median(img[core], 0), np.median(img[ring], 0)
    return float(np.linalg.norm(b - r) / np.sqrt(3.0)), float(np.linalg.norm(r - base) / np.sqrt(3.0))


def test_inline_band_matches_the_ground_beside_it() -> None:
    pairs = [p for seed in range(12) for bg in ("#ffffff", "#141313", "#9eab96")
             if (p := _band_vs_ground(seed, bg)) is not None]
    assert len(pairs) >= 8
    gaps, drifts = np.array(pairs).T
    assert drifts.max() > 0.02          # the field really drifts off the nominal colour…
    assert np.median(gaps) < 0.015      # …and the band drifts with it
