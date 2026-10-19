"""Dependency-light native video preprocessing."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from torchvision.transforms import functional as F


@dataclass(frozen=True)
class NativeVideoAugmentation:
    """Apply one spatial/color transform consistently to every frame and view."""

    training: bool = True
    crop_scale: float = 0.95
    width: int = 320
    height: int = 160
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.08
    frame_chunk_size: int = 0

    def __post_init__(self):
        if (isinstance(self.frame_chunk_size, bool)
                or not isinstance(self.frame_chunk_size, Integral)
                or self.frame_chunk_size < 0):
            raise ValueError("frame_chunk_size must be a non-negative integer")

    def __call__(
        self, video: np.ndarray, *, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        values = np.asarray(video)
        if values.dtype != np.uint8 or values.ndim not in {4, 5}:
            raise ValueError("video must be uint8 [T,H,W,C] or [T,V,H,W,C]")
        flat = values.reshape(-1, *values.shape[-3:])
        if self.frame_chunk_size == 0 or len(flat) <= self.frame_chunk_size:
            return self._augment(values, rng=rng)
        generator = rng or np.random.default_rng()
        initial = copy.deepcopy(generator)
        parts = []
        for start in range(0, len(flat), self.frame_chunk_size):
            # Reuse identical spatial/color choices for every chunk. Advance
            # the caller's RNG only once, exactly as the unchunked operation.
            chunk_rng = generator if start == 0 else copy.deepcopy(initial)
            parts.append(self._augment(flat[start:start + self.frame_chunk_size], rng=chunk_rng))
        return np.concatenate(parts).reshape(*values.shape[:-3], self.height, self.width, 3)

    def _augment(
        self, video: np.ndarray, *, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        values = np.asarray(video)
        if values.dtype != np.uint8 or values.ndim not in {4, 5}:
            raise ValueError("video must be uint8 [T,H,W,C] or [T,V,H,W,C]")
        original_shape = values.shape
        flat = values.reshape(-1, *values.shape[-3:])
        flat = np.ascontiguousarray(flat)
        # torch.from_numpy cannot safely represent a read-only mmap/view.
        # Keep zero-copy conversion for ordinary writable arrays.
        if not flat.flags.writeable:
            flat = flat.copy()
        tensor = torch.from_numpy(flat).permute(0, 3, 1, 2)
        generator = rng or np.random.default_rng()
        source_height, source_width = tensor.shape[-2:]
        crop_height = max(1, int(source_height * self.crop_scale))
        crop_width = max(1, int(source_width * self.crop_scale))
        if self.training:
            top = int(generator.integers(0, source_height - crop_height + 1))
            left = int(generator.integers(0, source_width - crop_width + 1))
        else:
            top = (source_height - crop_height) // 2
            left = (source_width - crop_width) // 2
        tensor = F.resized_crop(
            tensor,
            top,
            left,
            crop_height,
            crop_width,
            [self.height, self.width],
            antialias=True,
        )
        if self.training:
            tensor = tensor.to(torch.float32) / 255.0
            operations = [
                ("brightness", float(generator.uniform(1 - self.brightness, 1 + self.brightness))),
                ("contrast", float(generator.uniform(1 - self.contrast, 1 + self.contrast))),
                ("saturation", float(generator.uniform(1 - self.saturation, 1 + self.saturation))),
                ("hue", float(generator.uniform(-self.hue, self.hue))),
            ]
            generator.shuffle(operations)
            for name, factor in operations:
                tensor = getattr(F, f"adjust_{name}")(tensor, factor)
            tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
        else:
            tensor = tensor.to(torch.uint8)
        output = tensor.permute(0, 2, 3, 1).cpu().numpy()
        return output.reshape(*original_shape[:-3], self.height, self.width, 3)


__all__ = ["NativeVideoAugmentation"]
