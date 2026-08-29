from __future__ import annotations

import numpy as np
import pytest

from byteprint.crops import texture_score
from byteprint.launder import OFFICIAL_LADDER, apply, ladder, sample_spec


def noisy(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def smooth(size: int = 128) -> np.ndarray:
    ramp = np.linspace(0, 255, size, dtype=np.uint8)
    return np.repeat(np.tile(ramp, (size, 1))[:, :, None], 3, axis=2)


def colourful(size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(40, 200, size=(size, size, 3), dtype=np.uint8)


def test_none_spec_returns_the_image_unchanged() -> None:
    image = noisy()

    assert np.array_equal(apply(image, "none"), image)


def test_jpeg_preserves_shape_but_alters_pixels() -> None:
    image = noisy()

    out = apply(image, "jpeg:40")

    assert out.shape == image.shape
    assert not np.array_equal(out, image)


def test_lower_jpeg_quality_distorts_the_image_more() -> None:
    image = smooth()

    mild = np.abs(apply(image, "jpeg:90").astype(int) - image.astype(int)).mean()
    harsh = np.abs(apply(image, "jpeg:20").astype(int) - image.astype(int)).mean()

    assert harsh > mild


def test_downscale_round_trips_to_the_original_shape() -> None:
    image = noisy(120)

    out = apply(image, "scale:0.5")

    assert out.shape == image.shape


def test_downscale_destroys_high_frequency_detail() -> None:
    image = noisy()

    assert texture_score(apply(image, "scale:0.5")) < texture_score(image)


def test_blur_destroys_high_frequency_detail() -> None:
    image = noisy()

    assert texture_score(apply(image, "blur:1.0")) < texture_score(image)


# -- noise: sigma is normalised to [0, 1], as in the brief -----------------


def test_noise_adds_high_frequency_energy() -> None:
    image = smooth()

    assert texture_score(apply(image, "noise:0.05", seed=0)) > texture_score(image)


def test_noise_sigma_is_a_fraction_of_full_scale() -> None:
    """sigma=0.10 must perturb by ~25.5 levels, not ~0.1."""
    image = np.full((64, 64, 3), 128, dtype=np.uint8)

    out = apply(image, "noise:0.10", seed=0).astype(float)

    assert 20.0 < out.std() < 31.0


def test_larger_noise_sigma_perturbs_more() -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)

    mild = apply(image, "noise:0.02", seed=0).astype(float).std()
    harsh = apply(image, "noise:0.10", seed=0).astype(float).std()

    assert harsh > mild


def test_noise_output_stays_in_the_uint8_range() -> None:
    out = apply(np.full((32, 32, 3), 250, dtype=np.uint8), "noise:0.5", seed=0)

    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_noise_in_0_255_units_is_rejected_rather_than_silently_reinterpreted() -> None:
    """The old ladder wrote `noise:3`. Fail loudly rather than blast the image."""
    with pytest.raises(ValueError, match="normalised"):
        apply(noisy(), "noise:3")


# -- colour jitter ---------------------------------------------------------


def test_jitter_changes_pixels_without_changing_shape() -> None:
    image = colourful()

    out = apply(image, "jitter:0.2", seed=0)

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, image)


def test_jitter_of_zero_leaves_the_image_alone() -> None:
    image = colourful()

    assert np.array_equal(apply(image, "jitter:0.0", seed=0), image)


def test_jitter_stays_within_its_stated_budget() -> None:
    """+-20% on three factors cannot move mean intensity by more than ~2x."""
    image = colourful()

    out = apply(image, "jitter:0.2", seed=4).astype(float)

    ratio = out.mean() / image.astype(float).mean()
    assert 0.5 < ratio < 2.0


def test_jitter_is_reproducible_for_a_given_seed() -> None:
    image = colourful()

    assert np.array_equal(apply(image, "jitter:0.2", seed=9), apply(image, "jitter:0.2", seed=9))


def test_jitter_differs_between_seeds() -> None:
    image = colourful()

    assert not np.array_equal(apply(image, "jitter:0.2", seed=1), apply(image, "jitter:0.2", seed=2))


def test_an_out_of_range_jitter_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="jitter"):
        apply(noisy(), "jitter:1.5")


# -- centre crop -----------------------------------------------------------


def test_crop_keeps_the_requested_fraction_of_each_side() -> None:
    image = noisy(100)

    out = apply(image, "crop:0.8")

    assert out.shape == (80, 80, 3)


def test_crop_takes_the_centre_of_the_image() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[40:60, 40:60] = 255

    out = apply(image, "crop:0.8")

    assert out[30:50, 30:50].min() == 255


def test_crop_of_one_is_a_no_op() -> None:
    image = noisy(64)

    assert np.array_equal(apply(image, "crop:1.0"), image)


def test_an_out_of_range_crop_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="crop"):
        apply(noisy(), "crop:1.4")


# -- the ladder ------------------------------------------------------------


def test_the_ladder_starts_with_the_untouched_image() -> None:
    assert ladder()[0] == "none"


def test_the_official_ladder_is_exactly_the_transforms_in_the_brief() -> None:
    """Section 5.2 is a fixed list. Drifting from it invalidates every number."""
    assert ladder() == [
        "none",
        "jpeg:90",
        "jpeg:70",
        "jpeg:50",
        "jpeg:30",
        "blur:0.5",
        "blur:1.0",
        "blur:2.0",
        "scale:0.5",
        "scale:0.25",
        "noise:0.02",
        "noise:0.05",
        "noise:0.10",
        "jitter:0.2",
        "crop:0.8",
    ]


def test_the_official_ladder_is_the_default() -> None:
    assert ladder() == list(OFFICIAL_LADDER)


def test_every_official_rung_applies_without_error() -> None:
    image = noisy(128)

    for spec in ladder():
        out = apply(image, spec, seed=0)
        assert out.dtype == np.uint8 and out.ndim == 3


def test_the_stress_ladder_chains_transforms_beyond_the_brief() -> None:
    rungs = ladder("stress")

    assert any("|" in rung for rung in rungs)
    assert set(ladder()).isdisjoint(rung for rung in rungs if "|" in rung)


def test_an_unknown_ladder_is_rejected() -> None:
    with pytest.raises(ValueError, match="kitchen-sink"):
        ladder("kitchen-sink")


# -- spec parsing ----------------------------------------------------------


def test_specs_can_be_chained_left_to_right() -> None:
    image = noisy()

    chained = apply(image, "scale:0.5|jpeg:30")

    assert chained.shape == image.shape
    assert not np.array_equal(chained, apply(image, "scale:0.5"))


def test_an_unknown_operation_is_rejected() -> None:
    with pytest.raises(ValueError, match="posterize"):
        apply(noisy(), "posterize:3")


def test_a_malformed_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="jpeg"):
        apply(noisy(), "jpeg")


def test_sampling_an_augmentation_spec_is_reproducible() -> None:
    first = [sample_spec(np.random.default_rng(3)) for _ in range(5)]
    second = [sample_spec(np.random.default_rng(3)) for _ in range(5)]

    assert first == second


def test_sampled_specs_only_use_operations_the_brief_names() -> None:
    rng = np.random.default_rng(0)
    official_ops = {"jpeg", "blur", "scale", "noise", "jitter", "crop"}

    for _ in range(200):
        spec = sample_spec(rng)
        if spec == "none":
            continue
        assert {stage.split(":")[0] for stage in spec.split("|")} <= official_ops


def test_every_sampled_spec_can_actually_be_applied() -> None:
    image = noisy(64)
    rng = np.random.default_rng(0)

    for _ in range(100):
        apply(image, sample_spec(rng), seed=0)


def test_noise_with_the_same_seed_is_reproducible() -> None:
    image = smooth()

    assert np.array_equal(apply(image, "noise:0.05", seed=11), apply(image, "noise:0.05", seed=11))
