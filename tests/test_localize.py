"""Crop strategies that look for the edited region rather than the busy one.

The fixtures here plant a *locally denoised* patch: a region whose structure is
intact but whose sensor grain has been removed. That is a proxy for an inpainted
region, not a replica of one -- but it is the right proxy, because it reproduces
the property that matters. A generated region is resampled from a decoder rather
than captured through a lens, so its high-frequency statistics differ from the
rest of the frame, and they usually differ by being *cleaner*.

Which is also why `texture` is the wrong instrument for it: ranking windows by
high-frequency energy does not merely fail to find such a region, it actively
sorts it last.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from byteprint.crops import CROP_MODES, LAPLACIAN, convolve2d_valid, select_crops, texture_score
from byteprint.launder import apply as launder
from byteprint.localize import (
    anomaly_z,
    band_map,
    ela_rank_origins,
    rank_origins,
    window_fingerprints,
    window_stats,
)

BOX = (80, 80, 96, 96)  # top, left, height, width -- the planted region


def camera_like(size: int = 256, seed: int = 0) -> np.ndarray:
    """Low-frequency structure plus uniform sensor grain, as a photograph has."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    base = 110.0 + 40.0 * np.sin(xx / 23.0) + 30.0 * np.cos(yy / 17.0)
    stacked = np.stack([base, base * 0.98, base * 1.02], axis=-1)
    grain = rng.normal(0.0, 9.0, stacked.shape)
    return np.clip(stacked + grain, 0, 255).astype(np.uint8)


def with_denoised_patch(image: np.ndarray, box: tuple[int, int, int, int] = BOX) -> np.ndarray:
    """Blur one region: structure survives, grain does not. The planted 'edit'."""
    top, left, height, width = box
    out = image.copy()
    patch = Image.fromarray(out[top : top + height, left : left + width])
    out[top : top + height, left : left + width] = np.asarray(
        patch.filter(ImageFilter.GaussianBlur(radius=2.0))
    )
    return out


def grid_origins(size: int, crop_size: int, step: int) -> list[tuple[int, int]]:
    """A deterministic candidate set, so ranking is measured without sampling noise."""
    stops = range(0, size - crop_size + 1, step)
    return [(top, left) for top in stops for left in stops]


def hits(origins, crop_size: int, box: tuple[int, int, int, int] = BOX) -> int:
    """How many crops sit at least half inside the planted region."""
    top_b, left_b, height_b, width_b = box
    count = 0
    for top, left in origins:
        overlap_y = max(0, min(top + crop_size, top_b + height_b) - max(top, top_b))
        overlap_x = max(0, min(left + crop_size, left_b + width_b) - max(left, left_b))
        if overlap_y * overlap_x >= 0.5 * crop_size * crop_size:
            count += 1
    return count


# -- the load-bearing measurement -----------------------------------------


def test_anomaly_ranking_lands_on_the_edited_region_far_more_often_than_texture() -> None:
    """The whole point of the mode, measured rather than asserted."""
    crop_size, top_k = 64, 2
    origins = grid_origins(256, crop_size, step=16)

    anomaly_hits = texture_hits = 0
    for seed in range(20):
        image = with_denoised_patch(camera_like(256, seed=seed))

        ranked, used_fallback = rank_origins(image, origins, crop_size=crop_size)
        assert not used_fallback, "an image with a planted edit should not fall back"
        anomaly_hits += hits(ranked[:top_k], crop_size)

        by_texture = sorted(
            origins,
            key=lambda o: texture_score(image[o[0] : o[0] + crop_size, o[1] : o[1] + crop_size]),
            reverse=True,
        )
        texture_hits += hits(by_texture[:top_k], crop_size)

    assert anomaly_hits >= 30, f"anomaly found the region only {anomaly_hits}/40 times"
    assert texture_hits <= 5, f"texture unexpectedly found the region {texture_hits}/40 times"


def test_texture_ranking_sorts_a_denoised_region_last_rather_than_at_random() -> None:
    """Why the tampered split is hard: the instrument is anti-correlated, not blind."""
    crop_size = 64
    origins = grid_origins(256, crop_size, step=16)
    image = with_denoised_patch(camera_like(256, seed=3))

    scored = sorted(
        origins,
        key=lambda o: texture_score(image[o[0] : o[0] + crop_size, o[1] : o[1] + crop_size]),
    )
    assert hits(scored[:2], crop_size) == 2, "the least-textured windows are the edited ones"


# -- not regressing the uniform case ---------------------------------------


def test_a_uniform_image_falls_back_to_exactly_what_texture_would_have_chosen() -> None:
    """A fully synthetic image has no odd region; it must keep today's behaviour."""
    crop_size = 64
    origins = grid_origins(256, crop_size, step=32)
    image = camera_like(256, seed=11)

    ranked, used_fallback = rank_origins(image, origins, crop_size=crop_size)

    assert used_fallback
    by_texture = sorted(
        origins,
        key=lambda o: texture_score(image[o[0] : o[0] + crop_size, o[1] : o[1] + crop_size]),
        reverse=True,
    )
    assert ranked == by_texture


def test_the_fallback_fires_on_almost_every_uniform_image() -> None:
    """The floor is a measured property, not a guess: false triggers stay rare."""
    crop_size = 64
    origins = grid_origins(256, crop_size, step=16)

    fallbacks = sum(
        rank_origins(camera_like(256, seed=seed), origins, crop_size=crop_size)[1]
        for seed in range(40)
    )
    assert fallbacks >= 36, f"fell back on only {fallbacks}/40 uniform images"


# -- the pieces ------------------------------------------------------------


def test_a_denoised_window_has_a_lower_residual_fingerprint_than_a_grainy_one() -> None:
    image = with_denoised_patch(camera_like(256, seed=0))

    inside = window_fingerprints(image, [(96, 96)], crop_size=64)
    outside = window_fingerprints(image, [(8, 8)], crop_size=64)

    assert np.all(inside[0] < outside[0])


def test_the_fast_laplacian_is_the_same_kernel_the_texture_score_uses() -> None:
    """The band is computed by shifted slices for speed; it must still be LAPLACIAN."""
    window = camera_like(64, seed=2)

    bands = band_map(window)

    luma = window.astype(np.float64).mean(axis=2)
    assert np.allclose(bands[:, :, 0], convolve2d_valid(luma, LAPLACIAN))


def test_the_texture_scores_computed_alongside_the_bands_are_the_real_ones() -> None:
    """The fallback reuses this pass; if it drifted, the fallback would not be `texture`."""
    image = with_denoised_patch(camera_like(256, seed=6))
    origins = grid_origins(256, 64, step=48)

    _, textures = window_stats(image, origins, crop_size=64)

    expected = [texture_score(image[t : t + 64, l : l + 64]) for t, l in origins]
    assert np.allclose(textures, expected)


def test_the_anomaly_score_is_robust_to_a_third_of_the_windows_being_edited() -> None:
    """Median and MAD, not mean and std: the outlier must not define the baseline."""
    fingerprints = np.concatenate(
        [np.zeros((20, 4)), np.full((10, 4), 5.0)]  # 10 of 30 windows are the edit
    )

    z = anomaly_z(fingerprints)

    assert z[:20].max() < z[20:].min(), "the majority must set the baseline"


# -- contract --------------------------------------------------------------


@pytest.mark.parametrize("mode", ["anomaly", "ela"])
def test_the_mode_is_registered_and_returns_crops_of_the_requested_shape(mode: str) -> None:
    assert mode in CROP_MODES.names()

    crops = select_crops(camera_like(256), crop_size=64, top_k=3, mode=mode, seed=0)

    assert len(crops) == 3
    assert all(c.shape == (64, 64, 3) and c.dtype == np.uint8 for c in crops)


@pytest.mark.parametrize("mode", ["anomaly", "ela"])
def test_selection_is_reproducible_for_a_fixed_seed(mode: str) -> None:
    image = with_denoised_patch(camera_like(256, seed=5))

    first = select_crops(image, crop_size=64, top_k=2, mode=mode, seed=7)
    second = select_crops(image, crop_size=64, top_k=2, mode=mode, seed=7)

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("mode", ["anomaly", "ela"])
def test_grayscale_and_undersized_images_are_handled_like_every_other_mode(mode: str) -> None:
    gray = np.random.default_rng(0).integers(0, 256, size=(40, 40), dtype=np.uint8)

    crops = select_crops(gray, crop_size=64, top_k=2, mode=mode, seed=0)

    assert all(c.shape == (64, 64, 3) for c in crops)


@pytest.mark.parametrize("mode", ["anomaly", "ela"])
def test_asking_for_more_crops_than_candidates_is_not_an_error(mode: str) -> None:
    crops = select_crops(camera_like(128), crop_size=64, top_k=8, mode=mode, candidates=2, seed=0)

    assert 1 <= len(crops) <= 8


# -- how the two cues behave under the ladder ------------------------------
#
# These pin a *relationship*, not a score. The absolute hit rates come from a
# planted proxy and mean nothing about SID_Set; which instrument survives which
# rung is a property of the cues themselves and is what these guard.


def _found_by(ranker, spec: str, *, seeds: int = 8, crop_size: int = 64, top_k: int = 2) -> int:
    origins = grid_origins(256, crop_size, step=16)
    total = 0
    for seed in range(seeds):
        image = with_denoised_patch(camera_like(256, seed=seed))
        if spec != "none":
            image = launder(image, spec, seed=seed)
        total += hits(ranker(image, origins, crop_size)[:top_k], crop_size)
    return total


def _anomaly_ranker(image, origins, crop_size):
    return rank_origins(image, origins, crop_size=crop_size)[0]


def _ela_ranker(image, origins, crop_size):
    return ela_rank_origins(image, origins, crop_size=crop_size)


def _texture_ranker(image, origins, crop_size):
    return sorted(
        origins,
        key=lambda o: texture_score(image[o[0] : o[0] + crop_size, o[1] : o[1] + crop_size]),
        reverse=True,
    )


@pytest.mark.parametrize("spec", ["none", "jpeg:30", "noise:0.05"])
def test_both_localising_cues_find_the_edit_where_texture_never_does(spec: str) -> None:
    assert _found_by(_anomaly_ranker, spec) == 16
    assert _found_by(_ela_ranker, spec) == 16
    assert _found_by(_texture_ranker, spec) == 0


def test_the_two_cues_fail_on_different_rungs_rather_than_together() -> None:
    """Why `ela` stays registered: it is not a strictly worse `anomaly`.

    `blur:2.0` destroys the grain everywhere, taking with it the contrast the
    anomaly cue reads -- but the smoothed region still responds to a re-encode
    unlike its surroundings, so ELA still finds it. Complementary failure is the
    same argument the project already makes for fusing two experts.
    """
    assert _found_by(_anomaly_ranker, "blur:2.0") == 0
    assert _found_by(_ela_ranker, "blur:2.0") > 8


def test_the_cue_hands_over_to_texture_rather_than_guessing_when_grain_is_destroyed() -> None:
    """Being blind is survivable; being confidently wrong is not.

    On the rungs where the anomaly cue dies, the fallback must fire on every
    image, so the mode degrades to today's behaviour instead of returning windows
    ranked by noise.
    """
    crop_size = 64
    origins = grid_origins(256, crop_size, step=16)

    for seed in range(8):
        washed = launder(with_denoised_patch(camera_like(256, seed=seed)), "blur:2.0", seed=seed)
        ranked, used_fallback = rank_origins(washed, origins, crop_size=crop_size)

        assert used_fallback, "a dead cue must defer, not guess"
        assert ranked == _texture_ranker(washed, origins, crop_size)
