"""Shared fixtures: build tiny on-disk datasets in the folder layout byteprint expects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def write_image(path: Path, size: tuple[int, int] = (64, 64), seed: int = 0) -> Path:
    """Write a small deterministic noise image. Real pixels, so loaders are exercised."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """A split directory with two real images and three fakes from two generators."""
    root = tmp_path / "train"
    write_image(root / "real" / "a.jpg", seed=1)
    write_image(root / "real" / "b.png", seed=2)
    write_image(root / "fake" / "sdxl" / "c.png", seed=3)
    write_image(root / "fake" / "sdxl" / "d.png", seed=4)
    write_image(root / "fake" / "flux" / "e.png", seed=5)
    return root
