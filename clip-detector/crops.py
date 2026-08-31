"""
Native-resolution texture-crop selection for CLIP embedding, so the
backbone sees real high-frequency detail instead of a whole image that
CLIPProcessor has already downsampled to 224x224.

Resizing a whole image to fit a backbone's input is a low-pass filter
applied directly to the evidence a forensic detector needs - generator
artifacts concentrate in high-frequency texture, and CLIP's internal
224x224 resize destroys most of it for anything larger than that. This
module samples several native-resolution 224x224 crops per image and
keeps the ones with the most high-frequency energy (Laplacian-response
variance), so the backbone gets to look at un-downsampled pixels.

Meant for GPU use: extracting/embedding several crops per image
multiplies compute several-fold over the whole-image approach used in
clip_features.py, which is why that approach was used for the
CPU-only proof-of-concept in this project. See clip_features.
embed_images_preproj_crops for the embedding side.

Usage:
    from crops import select_crops
    crops = select_crops(pil_image, crop_size=224, top_k=3)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

CROP_SIZE = 224  # CLIP ViT-B/32 and ViT-L/14 both take 224x224 input
_LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _convolve2d_valid(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    windows = np.lib.stride_tricks.sliding_window_view(plane, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def texture_score(crop_rgb: np.ndarray) -> float:
    """High-frequency energy of a crop: variance of its Laplacian response.
    Flat regions (sky, blank background) score near 0; noise/fine detail
    score high."""
    luma = np.asarray(crop_rgb, dtype=np.float64).mean(axis=2)
    if min(luma.shape) < 3:
        return 0.0
    return float(_convolve2d_valid(luma, _LAPLACIAN).var())


def _pad_to_fit(img: Image.Image, size: int) -> Image.Image:
    """Upscale an undersized image so a full-size crop can be taken."""
    w, h = img.size
    if w >= size and h >= size:
        return img
    scale = size / min(w, h)
    return img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)


def select_crops(
    img: Image.Image,
    crop_size: int = CROP_SIZE,
    top_k: int = 3,
    candidates: int = 12,
    rng: np.random.Generator | None = None,
) -> list[Image.Image]:
    """Return up to `top_k` native-resolution `crop_size`-square crops,
    oversampling `candidates` random locations and keeping the ones with
    the most high-frequency texture. Falls back to a center crop if the
    image (after upscaling to fit) has no room for multiple distinct
    positions."""
    rng = rng or np.random.default_rng()
    img = img.convert("RGB")
    img = _pad_to_fit(img, crop_size)
    w, h = img.size

    if w == crop_size and h == crop_size:
        return [img]

    n = max(candidates, top_k)
    origins = sorted({
        (int(rng.integers(0, w - crop_size + 1)), int(rng.integers(0, h - crop_size + 1)))
        for _ in range(n)
    })
    crops = [img.crop((x, y, x + crop_size, y + crop_size)) for x, y in origins]
    crops.sort(key=lambda c: texture_score(np.asarray(c)), reverse=True)
    return crops[:top_k]
