from __future__ import annotations

import numpy as np
import pytest

from byteprint.crops import select_crops, texture_score


def flat(size: int = 128, value: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def noisy(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_texture_score_is_higher_for_noise_than_for_a_flat_patch() -> None:
    assert texture_score(noisy()) > texture_score(flat())


def test_texture_score_of_a_flat_patch_is_zero() -> None:
    assert texture_score(flat()) == pytest.approx(0.0)


def test_texture_mode_returns_the_requested_number_of_crops_at_the_requested_size() -> None:
    crops = select_crops(noisy(256), crop_size=64, top_k=3, mode="texture")

    assert len(crops) == 3
    assert all(c.shape == (64, 64, 3) for c in crops)


def test_texture_mode_prefers_the_detailed_half_of_an_image() -> None:
    image = flat(256)
    image[:, 128:] = noisy(256)[:, 128:]  # right half is noise, left half is flat

    crops = select_crops(image, crop_size=32, top_k=4, mode="texture", candidates=64, seed=0)

    assert all(texture_score(c) > 0 for c in crops)


def test_center_mode_returns_exactly_one_crop_from_the_middle() -> None:
    image = flat(128)
    image[48:80, 48:80] = 255  # a marker only the center crop can see

    crops = select_crops(image, crop_size=32, top_k=5, mode="center")

    assert len(crops) == 1
    assert np.all(crops[0] == 255)


def test_resize_mode_returns_the_whole_image_scaled_to_crop_size() -> None:
    crops = select_crops(noisy(200), crop_size=64, top_k=3, mode="resize")

    assert len(crops) == 1
    assert crops[0].shape == (64, 64, 3)


def test_images_smaller_than_the_crop_are_upscaled_rather_than_dropped() -> None:
    crops = select_crops(noisy(40), crop_size=64, top_k=2, mode="texture")

    assert len(crops) >= 1
    assert all(c.shape == (64, 64, 3) for c in crops)


def test_crop_selection_is_reproducible_for_a_fixed_seed() -> None:
    image = noisy(256)

    first = select_crops(image, crop_size=64, top_k=3, mode="random", seed=7)
    second = select_crops(image, crop_size=64, top_k=3, mode="random", seed=7)

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_different_seeds_select_different_random_crops() -> None:
    image = noisy(256)

    first = select_crops(image, crop_size=64, top_k=3, mode="random", seed=1)
    second = select_crops(image, crop_size=64, top_k=3, mode="random", seed=2)

    assert not all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="quadtree"):
        select_crops(noisy(), crop_size=32, top_k=1, mode="quadtree")


def test_grayscale_images_are_promoted_to_three_channels() -> None:
    gray = np.random.default_rng(0).integers(0, 256, size=(128, 128), dtype=np.uint8)

    crops = select_crops(gray, crop_size=32, top_k=2, mode="texture")

    assert all(c.shape == (32, 32, 3) for c in crops)


# -- prefix stability ------------------------------------------------------
#
# `--crop-limit` reproduces a small-k run from a large-k cache, and that only
# works for modes that *rank a fixed candidate set*: they draw
# max(candidates, top_k) windows from a seeded generator and return the head of
# one ordering, so top_k changes what is returned and not what was considered.
# Pinned here because it is load-bearing for the pooling comparison and it is
# exactly the kind of property that rots silently.


def structured(size: int = 512, seed: int = 0) -> np.ndarray:
    """Noise with a smoother patch, so ranking has something to order."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    image[100:250, 150:300] = (image[100:250, 150:300] * 0.2).astype(np.uint8)
    return image


@pytest.mark.parametrize("mode", ["texture", "anomaly", "ela"])
def test_a_larger_top_k_extends_the_selection_rather_than_redrawing_it(mode: str) -> None:
    image = structured()

    few = select_crops(image, crop_size=64, top_k=2, mode=mode, seed=7)
    many = select_crops(image, crop_size=64, top_k=8, mode=mode, seed=7)

    assert len(few) == 2 and len(many) == 8
    assert all(np.array_equal(a, b) for a, b in zip(few, many[:2], strict=True))


def test_random_mode_does_not_have_that_property() -> None:
    # It draws exactly top_k origins, so a different top_k is a different draw.
    # Stated as a test so nobody assumes --crop-limit is mode-independent.
    image = structured()

    few = select_crops(image, crop_size=64, top_k=2, mode="random", seed=7)
    many = select_crops(image, crop_size=64, top_k=8, mode="random", seed=7)

    assert not all(np.array_equal(a, b) for a, b in zip(few, many[:2], strict=True))
