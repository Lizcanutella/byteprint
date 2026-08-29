"""A tiny synthetic dataset, so the pipeline is runnable before you have data.

The "real" images are smooth gradients plus band-limited noise. The "fake" ones
add a periodic grid -- a caricature of the upsampling artifact that real
generators leave behind. Both classes are written as PNG at the same size: if
reals were JPEG and fakes were PNG, a classifier would reach 99% by learning
the container format, which is the single most common way this benchmark gets
accidentally faked.

This is a smoke-test fixture. Numbers from it mean the wiring works, nothing more.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

GENERATORS = ("gridnet", "ringnet")


def _base(rng: np.random.Generator, size: int) -> np.ndarray:
    """A smooth scene: low-frequency structure with a little fine grain."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    scene = 0.5 + 0.25 * np.sin(6.0 * np.pi * xx * rng.uniform(0.6, 1.4))
    scene += 0.2 * np.cos(4.0 * np.pi * yy * rng.uniform(0.6, 1.4))
    scene += 0.03 * rng.normal(size=(size, size))
    return scene


def _grid_artifact(size: int, period: int, amplitude: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return amplitude * (np.cos(2 * np.pi * xx / period) * np.cos(2 * np.pi * yy / period))


def _write(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.clip(array, 0.0, 1.0) * 255.0
    rgb = np.repeat(pixels.astype(np.uint8)[:, :, None], 3, axis=2)
    Image.fromarray(rgb).save(path)


def build(out: Path | str, *, per_class: int = 64, size: int = 96, seed: int = 0) -> Path:
    """Write train/ and test/ splits under ``out`` and return the root."""
    out = Path(out)
    rng = np.random.default_rng(seed)

    for split, count in (("train", per_class), ("test", max(2, per_class // 2))):
        for index in range(count):
            _write(_base(rng, size), out / split / "real" / f"real_{index:04d}.png")

        for gen_index, generator in enumerate(GENERATORS):
            period = 4 + 2 * gen_index
            for index in range(max(1, count // len(GENERATORS))):
                scene = _base(rng, size) + _grid_artifact(size, period, amplitude=0.035)
                _write(scene, out / split / "fake" / generator / f"{generator}_{index:04d}.png")

    return out
