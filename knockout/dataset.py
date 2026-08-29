"""Dataset: (input RGB + color-distance channel) → alpha, from a synth_composite manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .metrics import color_distance, hex_to_rgb01


class MatteDataset(Dataset):
    def __init__(self, root: str | Path, size: int = 320) -> None:
        self.root = Path(root)
        self.size = size
        with (self.root / "manifest.jsonl").open() as f:
            self.items = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        it = self.items[i]
        s = self.size
        rgb = Image.open(self.root / it["input"]).convert("RGB").resize((s, s), Image.BILINEAR)
        alpha = Image.open(self.root / it["alpha"]).convert("L").resize((s, s), Image.NEAREST)

        rgb01 = np.asarray(rgb, dtype=np.float32) / 255.0
        bg01 = hex_to_rgb01(it["bg_hex"])
        dist = color_distance(rgb01, bg01).astype(np.float32)  # HxW conditioning channel

        x = np.concatenate([rgb01, dist[:, :, None]], axis=-1)  # HxWx4
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        y = torch.from_numpy((np.asarray(alpha, np.float32) / 255.0)[None])
        bg = torch.from_numpy(bg01.astype(np.float32))  # requested garment color (3,)
        return x, y, bg
