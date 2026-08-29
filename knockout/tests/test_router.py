"""Router regression guards — pin the verdicts the ground_frac fix corrected.

Run: python -m knockout.tests.test_router   (no pytest needed)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from knockout.router import should_remove_background


def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def die_cut_on_white_touching_edge() -> Image.Image:
    """A subject on a large removable white ground that reaches the frame — the case the old
    router mis-called full_bleed. Lots of white (high ground_frac) + art in the border band."""
    a = np.full((256, 256, 3), 255, np.uint8)
    a[40:230, 40:120] = (20, 30, 200)   # a solid subject, offset, reaching the left/top border
    a[:8, 40:120] = (20, 30, 200)       # bleeds into the top edge band
    return _img(a)


def aop_pattern() -> Image.Image:
    """Edge-to-edge pattern: no removable ground, patterned border -> full_bleed."""
    rng = np.random.default_rng(0)
    return _img(rng.integers(0, 255, (256, 256, 3)))


def solid_inset_art() -> Image.Image:
    """Inset art on a clean solid ground, not touching edges -> knockout."""
    a = np.full((256, 256, 3), 240, np.uint8)
    a[90:166, 90:166] = (200, 40, 40)
    return _img(a)


def check(name: str, img: Image.Image, expected: str, **kw) -> bool:
    d = should_remove_background(img, **kw)
    ok = d.verdict == expected
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {d.verdict} (want {expected})  "
          f"gf={d.signals.get('ground_frac')} ec={d.signals.get('edge_cover')} :: {d.reason[:40]}")
    return ok


def main() -> None:
    results = [
        check("die-cut on white, edge-touch -> knockout", die_cut_on_white_touching_edge(), "knockout"),
        check("aop pattern -> full_bleed", aop_pattern(), "full_bleed"),
        check("inset art on solid -> knockout", solid_inset_art(), "knockout"),
        check("product_is_aop=True -> full_bleed", solid_inset_art(), "full_bleed", product_is_aop=True),
        check("product_is_aop=False -> knockout", aop_pattern(), "knockout", product_is_aop=False),
    ]
    assert all(results), "router regression failed"
    print(f"\nall {len(results)} router guards passed")


if __name__ == "__main__":
    main()
