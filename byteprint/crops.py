"""Crop selection.

Resizing an image to fit a backbone's input is a low-pass filter applied
directly to the evidence a synthetic-image detector needs. So the default
path here samples crops at *native* resolution and keeps the ones richest in
high-frequency detail, where generator artifacts concentrate.

``mode="resize"`` reproduces the naive whole-image-downscale baseline so the
difference can be measured rather than assumed.

Strategies are registered rather than hard-coded, because *where you look* is
as much a research question as what you look with:

    @register_crop_mode("my-mode")
    def _sample(image, *, crop_size, top_k, candidates, rng):
        return [...]  # uint8 HWC arrays, crop_size square

Pass ``pad=False`` if the strategy handles undersized images itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from byteprint.registry import Registry

# 3x3 discrete Laplacian; its response variance is a cheap high-frequency proxy.
_LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D array, got shape {image.shape}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError(f"expected 1, 3 or 4 channels, got {image.shape[2]}")
    return np.ascontiguousarray(image)


def _convolve2d_valid(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid-mode 2-D convolution via stride tricks, so we need no SciPy."""
    kh, kw = kernel.shape
    windows = np.lib.stride_tricks.sliding_window_view(plane, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def texture_score(crop: np.ndarray) -> float:
    """High-frequency energy of a crop: the variance of its Laplacian response.

    Flat regions score 0. Noise and fine detail score high.
    """
    crop = _as_rgb(np.asarray(crop))
    if min(crop.shape[:2]) < _LAPLACIAN.shape[0]:
        return 0.0
    luma = crop.astype(np.float64).mean(axis=2)
    return float(_convolve2d_valid(luma, _LAPLACIAN).var())


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    resized = Image.fromarray(image).resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(resized)


def _pad_to_fit(image: np.ndarray, size: int) -> np.ndarray:
    """Scale an undersized image up so a full-size crop can be taken from it."""
    height, width = image.shape[:2]
    if height >= size and width >= size:
        return image
    scale = size / min(height, width)
    target = (max(size, round(width * scale)), max(size, round(height * scale)))
    return np.asarray(Image.fromarray(image).resize(target, Image.Resampling.BICUBIC))


# -- crop strategies -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CropMode:
    """A named way of choosing which parts of an image the backbone sees."""

    name: str
    sample: Callable[..., list[np.ndarray]]
    # Whether select_crops should upscale an image too small to crop from.
    pad: bool = True


CROP_MODES: Registry[CropMode] = Registry("crop mode")

# Back-compatible alias: `list(MODES)` still yields the registered names.
MODES = CROP_MODES

DEFAULT_CROP_MODE = "texture"


def register_crop_mode(name: str, *, pad: bool = True, replace: bool = False):
    """Decorator: register a crop-sampling strategy.

    The function is called as ``sample(image, crop_size=, top_k=, candidates=,
    rng=)`` and must return uint8 HWC arrays of ``crop_size`` square.
    """

    def decorator(sample: Callable[..., list[np.ndarray]]) -> Callable[..., list[np.ndarray]]:
        CROP_MODES.register(name, CropMode(name=name, sample=sample, pad=pad), replace=replace)
        return sample

    return decorator


def _random_origins(
    image: np.ndarray, crop_size: int, count: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    height, width = image.shape[:2]
    return sorted(
        {
            (
                int(rng.integers(0, height - crop_size + 1)),
                int(rng.integers(0, width - crop_size + 1)),
            )
            for _ in range(count)
        }
    )


@register_crop_mode("texture")
def _texture(image, *, crop_size, top_k, candidates, rng) -> list[np.ndarray]:
    """Oversample, then keep the crops richest in high-frequency detail.

    Generator artifacts concentrate in texture, and a flat sky crop carries
    almost no evidence either way -- so spending the backbone's budget there
    is waste.
    """
    origins = _random_origins(image, crop_size, max(candidates, top_k), rng)
    crops = [image[t : t + crop_size, l : l + crop_size] for t, l in origins]
    crops.sort(key=texture_score, reverse=True)
    return crops[:top_k]


@register_crop_mode("random")
def _random(image, *, crop_size, top_k, candidates, rng) -> list[np.ndarray]:
    """Uniform crops -- the control condition for the texture heuristic."""
    origins = _random_origins(image, crop_size, top_k, rng)
    return [image[t : t + crop_size, l : l + crop_size] for t, l in origins][:top_k]


@register_crop_mode("center")
def _center(image, *, crop_size, top_k, candidates, rng) -> list[np.ndarray]:
    """One crop from the middle. Cheap, deterministic, and usually the subject."""
    height, width = image.shape[:2]
    top, left = (height - crop_size) // 2, (width - crop_size) // 2
    return [image[top : top + crop_size, left : left + crop_size]]


@register_crop_mode("resize", pad=False)
def _resize_whole(image, *, crop_size, top_k, candidates, rng) -> list[np.ndarray]:
    """The naive baseline: squash the whole image to fit. Low-passes the evidence."""
    return [_resize(image, crop_size)]


def select_crops(
    image: np.ndarray,
    *,
    crop_size: int,
    top_k: int,
    mode: str = DEFAULT_CROP_MODE,
    candidates: int = 32,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Return up to ``top_k`` crops of ``crop_size`` square, as uint8 HWC arrays."""
    strategy = CROP_MODES.resolve(mode)
    if crop_size < 1 or top_k < 1:
        raise ValueError("crop_size and top_k must both be >= 1")

    image = _as_rgb(np.asarray(image))
    if strategy.pad:
        image = _pad_to_fit(image, crop_size)

    return strategy.sample(
        image,
        crop_size=crop_size,
        top_k=top_k,
        candidates=candidates,
        rng=np.random.default_rng(seed),
    )
