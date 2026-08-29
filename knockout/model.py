"""Background-conditioned matting U-Net.

Input: 4 channels = RGB + color-distance-to-known-bg. Output: 1-channel alpha logits.
Small enough to train on Apple-Silicon MPS (~2M params at width=32). rembg is U^2-Net;
this is a lean U-Net with residual encoder blocks — a faithful, correct v0. The RSU/U^2-Net
port is a drop-in upgrade once the data pipeline and eval are proven.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm(c: int) -> nn.GroupNorm:
    """GroupNorm — batch-size-independent, so training is stable at the small batches that
    high-resolution inputs force (BatchNorm degrades badly below ~8-16 per batch)."""
    g = 8
    while c % g:
        g //= 2
    return nn.GroupNorm(g, c)


def _cbr(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        _norm(cout),
        nn.ReLU(inplace=True),
    )


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.c1 = _cbr(cin, cout)
        self.c2 = _cbr(cout, cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2(self.c1(x)) + self.skip(x)


class KnockoutMatte(nn.Module):
    def __init__(self, in_ch: int = 4, width: int = 32, depth: int = 4) -> None:
        super().__init__()
        chs = [width * (2**i) for i in range(depth)]  # 32,64,128,256
        self.inc = _Block(in_ch, chs[0])
        self.downs = nn.ModuleList(_Block(chs[i], chs[i + 1]) for i in range(depth - 1))
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth - 1))
        )
        self.dec = nn.ModuleList(
            _Block(chs[i] * 2, chs[i]) for i in reversed(range(depth - 1))
        )
        self.outc = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = 2 ** (len(self.downs))
        if x.shape[-1] % f or x.shape[-2] % f:
            raise ValueError(f"input H and W must be divisible by {f}, got {tuple(x.shape[-2:])}")
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(self.pool(skips[-1])))
        h = skips[-1]
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips[:-1]), strict=True):
            h = dec(torch.cat([up(h), skip], dim=1))
        return self.outc(h)  # logits, HxW


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
