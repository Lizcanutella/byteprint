from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from byteprint.metrics import roc_auc
from byteprint.probe import LinearProbe, ProbeConfig


def separable(n: int = 60, dim: int = 8, seed: int = 0):
    """Two Gaussian blobs offset along the first axis."""
    rng = np.random.default_rng(seed)
    real = rng.normal(0.0, 0.5, size=(n, dim))
    fake = rng.normal(0.0, 0.5, size=(n, dim))
    fake[:, 0] += 4.0
    X = np.vstack([real, fake]).astype(np.float32)
    y = np.array([0] * n + [1] * n)
    return X, y


def test_a_fitted_probe_ranks_synthetic_above_real() -> None:
    X, y = separable()

    scores = LinearProbe().fit(X, y).score(X)

    assert roc_auc(y, scores) == pytest.approx(1.0)


def test_scores_are_probabilities() -> None:
    X, y = separable()

    scores = LinearProbe().fit(X, y).score(X)

    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_an_unfitted_probe_refuses_to_score() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        LinearProbe().score(np.zeros((2, 8), dtype=np.float32))


def test_fitting_needs_both_classes() -> None:
    X, _ = separable()

    with pytest.raises(ValueError, match="both classes"):
        LinearProbe().fit(X, np.ones(len(X), dtype=int))


def test_fitting_rejects_mismatched_lengths() -> None:
    X, y = separable()

    with pytest.raises(ValueError, match="same length"):
        LinearProbe().fit(X, y[:-1])


def test_the_default_threshold_is_one_half() -> None:
    X, y = separable()

    assert LinearProbe().fit(X, y).threshold == pytest.approx(0.5)


def test_calibrating_to_a_zero_budget_produces_no_false_positives() -> None:
    X, y = separable()
    probe = LinearProbe().fit(X, y).calibrate(X, y, target_fpr=0.0)

    predictions = probe.predict(X)

    assert predictions[y == 0].sum() == 0


def test_calibration_records_the_budget_it_was_fitted_to() -> None:
    X, y = separable()

    probe = LinearProbe().fit(X, y).calibrate(X, y, target_fpr=0.01)

    assert probe.target_fpr == pytest.approx(0.01)


def test_predictions_are_zero_or_one() -> None:
    X, y = separable()

    predictions = LinearProbe().fit(X, y).predict(X)

    assert set(np.unique(predictions).tolist()) <= {0, 1}


def test_a_saved_probe_reloads_with_identical_scores(tmp_path: Path) -> None:
    X, y = separable()
    probe = LinearProbe().fit(X, y).calibrate(X, y, target_fpr=0.0)
    probe.save(tmp_path / "probe.joblib")

    reloaded = LinearProbe.load(tmp_path / "probe.joblib")

    assert np.allclose(reloaded.score(X), probe.score(X))


def test_a_saved_probe_reloads_with_its_calibrated_threshold(tmp_path: Path) -> None:
    X, y = separable()
    probe = LinearProbe().fit(X, y).calibrate(X, y, target_fpr=0.0)
    probe.save(tmp_path / "probe.joblib")

    assert LinearProbe.load(tmp_path / "probe.joblib").threshold == probe.threshold


def test_feature_standardisation_makes_the_probe_insensitive_to_input_scale() -> None:
    X, y = separable()

    plain = LinearProbe().fit(X, y).score(X)
    rescaled = LinearProbe().fit(X * 1000.0, y).score(X * 1000.0)

    assert roc_auc(y, plain) == pytest.approx(roc_auc(y, rescaled))


def test_class_imbalance_does_not_collapse_the_ranking() -> None:
    X, y = separable(n=200)
    keep = np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0][:5]])

    probe = LinearProbe().fit(X[keep], y[keep])

    assert roc_auc(y, probe.score(X)) > 0.95


def test_the_configured_regularisation_strength_is_used() -> None:
    X, y = separable()

    probe = LinearProbe(ProbeConfig(C=0.01)).fit(X, y)

    assert probe.config.C == pytest.approx(0.01)


# -- bags of crops ---------------------------------------------------------


def bags(n: int = 60, crops: int = 4, dim: int = 8, seed: int = 0):
    """Two classes of image, each an equal-sized bag of crop embeddings.

    Every crop of a fake image carries the signal, so this is the *easy*
    geometry -- the one where mean and max should agree. Localised evidence is
    built separately, below.
    """
    rng = np.random.default_rng(seed)
    total = 2 * n * crops
    X = rng.normal(0.0, 0.5, size=(total, dim))
    X[n * crops :, 0] += 4.0
    counts = np.full(2 * n, crops, dtype=np.int64)
    y = np.array([0] * n + [1] * n)
    return X.astype(np.float32), counts, y


def localised_bags(n: int = 120, crops: int = 8, dim: int = 8, seed: int = 0):
    """Fakes carry the signal in exactly one crop; the rest look authentic.

    The tampered-image geometry: a real photograph with one edited region. The
    offset and noise are chosen so mean-pooling is clearly imperfect rather
    than saturated at AUC 1.0 -- a fixture that separates perfectly under every
    pooling cannot tell them apart, which is the failure the first draft of
    this helper had.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.5, size=(2 * n * crops, dim))
    for image in range(n, 2 * n):
        X[image * crops, 0] += 2.0
    counts = np.full(2 * n, crops, dtype=np.int64)
    y = np.array([0] * n + [1] * n)
    return X.astype(np.float32), counts, y


def test_bag_scoring_returns_one_score_per_image_not_per_crop() -> None:
    X, counts, y = bags()

    probe = LinearProbe().fit_bags(X, counts, y)

    assert probe.score_bags(X, counts).shape == (len(y),)


def test_mean_pooling_over_bags_matches_pooling_the_features_by_hand() -> None:
    # The equivalence that makes the refactor a refactor: pooling in the probe
    # is the same operation the cache used to perform on the way in.
    X, counts, y = bags()
    pooled = np.stack([chunk.mean(axis=0) for chunk in np.split(X, np.cumsum(counts)[:-1])])

    probe = LinearProbe().fit_bags(X, counts, y)

    assert np.allclose(probe.score_bags(X, counts), probe.score(pooled))


def test_crop_level_training_gives_every_crop_its_images_label() -> None:
    X, counts, y = bags(n=10, crops=3)

    probe = LinearProbe(ProbeConfig(train_pooling="crop", pooling="max")).fit_bags(X, counts, y)

    # 20 images x 3 crops fitted as 60 rows, not 20.
    assert probe.n_training_rows == 60


def test_bag_level_training_fits_one_row_per_image() -> None:
    X, counts, y = bags(n=10, crops=3)

    probe = LinearProbe(ProbeConfig(train_pooling="mean", pooling="max")).fit_bags(X, counts, y)

    assert probe.n_training_rows == 20


def test_score_pooling_reduces_the_crops_own_scores_and_nothing_else() -> None:
    # The mechanism, asserted exactly, rather than "max wins" asserted vaguely.
    # If this holds, a score-space pooling is doing what its name says.
    X, counts, y = localised_bags(n=20, crops=4)
    chunks = np.split(np.arange(len(X)), np.cumsum(counts)[:-1])

    for spec, reduce in (
        ("max", np.max), ("mean-score", np.mean), ("topk:2", lambda v: np.sort(v)[-2:].mean())
    ):
        probe = LinearProbe(ProbeConfig(pooling=spec)).fit_bags(X, counts, y)
        crop_scores = probe.score(X)

        expected = [reduce(crop_scores[rows]) for rows in chunks]

        assert np.allclose(probe.score_bags(X, counts), expected), spec


def test_max_pooling_recovers_evidence_that_mean_pooling_dilutes() -> None:
    # Holding the fitted head fixed, so the only thing that moves is the
    # reduction. This is arm 6 against arm 2 of the comparison in
    # docs/design-crop-pooling.md, on a fixture built to have the geometry the
    # tampered class is believed to have.
    #
    # Note what is deliberately *not* asserted here: that crop-level training
    # beats bag-level training. That is the open question the cluster run
    # exists to answer, and on this fixture it happens to come out the other
    # way. Pinning it would be building the proxy out of the hypothesis.
    X, counts, y = localised_bags()

    mean = LinearProbe(ProbeConfig(pooling="mean")).fit_bags(X, counts, y)
    peak = LinearProbe(ProbeConfig(pooling="max")).fit_bags(X, counts, y)

    assert roc_auc(y, peak.score_bags(X, counts)) > roc_auc(y, mean.score_bags(X, counts))


def test_inference_only_pooling_leaves_the_fit_untouched() -> None:
    # Arm 6 of the comparison: same trained head as the baseline, different
    # reduction at scoring time. The head must be identical, not merely close.
    X, counts, y = localised_bags()

    baseline = LinearProbe(ProbeConfig(pooling="mean")).fit_bags(X, counts, y)
    inference = LinearProbe(ProbeConfig(pooling="max")).fit_bags(X, counts, y)

    assert np.allclose(baseline.score(X), inference.score(X))
    assert not np.allclose(baseline.score_bags(X, counts), inference.score_bags(X, counts))


def test_a_crop_limit_truncates_every_bag_before_pooling() -> None:
    X, counts, y = localised_bags(crops=4)

    full = LinearProbe(ProbeConfig(pooling="mean")).fit_bags(X, counts, y)
    limited = LinearProbe(ProbeConfig(pooling="mean", crop_limit=2)).fit_bags(X, counts, y)

    first_two = np.stack(
        [chunk[:2].mean(axis=0) for chunk in np.split(X, np.cumsum(counts)[:-1])]
    )
    assert np.allclose(limited.score_bags(X, counts), limited.score(first_two))
    assert not np.allclose(limited.score_bags(X, counts), full.score_bags(X, counts))


def test_pooling_reproduces_the_old_write_time_average_bit_for_bit() -> None:
    # The reproduction arm of the pooling comparison is a gate: it must return
    # the published numbers to four decimals, so "close enough" is not enough.
    # The cache stored float32 and averaged in float32 before writing, so
    # pooling happens in the array's own dtype and widens only afterwards.
    rng = np.random.default_rng(3)
    n_images, crops, dim = 60, 8, 16
    X = rng.normal(size=(n_images * crops, dim)).astype(np.float32)
    counts = np.full(n_images, crops, dtype=np.int64)
    y = rng.integers(0, 2, size=n_images)

    # What the previous format wrote: the mean of the first two crops, float32.
    old_rows = np.stack([X[i * crops : i * crops + 2].mean(axis=0) for i in range(n_images)])
    old = LinearProbe(ProbeConfig()).fit(old_rows, y)
    new = LinearProbe(ProbeConfig(pooling="mean", crop_limit=2)).fit_bags(X, counts, y)

    assert np.array_equal(old.score(old_rows), new.score_bags(X, counts))


def test_pooling_settings_survive_being_saved_and_reloaded(tmp_path: Path) -> None:
    # A probe that scored with max at training time must not silently score
    # with mean after a round trip -- that is the failure this guards.
    X, counts, y = localised_bags()
    probe = LinearProbe(
        ProbeConfig(train_pooling="crop", pooling="topk:2", crop_limit=3)
    ).fit_bags(X, counts, y)
    probe.save(tmp_path / "probe.joblib")

    reloaded = LinearProbe.load(tmp_path / "probe.joblib")

    assert reloaded.config.pooling == "topk:2"
    assert reloaded.config.train_pooling == "crop"
    assert reloaded.config.crop_limit == 3
    assert np.allclose(reloaded.score_bags(X, counts), probe.score_bags(X, counts))


def test_calibrating_on_bags_thresholds_the_pooled_scores() -> None:
    X, counts, y = localised_bags()

    probe = LinearProbe(ProbeConfig(train_pooling="crop", pooling="max"))
    probe.fit_bags(X, counts, y).calibrate_bags(X, counts, y, target_fpr=0.0)

    pooled = probe.score_bags(X, counts)
    assert probe.threshold > pooled[y == 0].max() or probe.threshold <= pooled.max()
    assert probe.target_fpr == pytest.approx(0.0)


def test_an_unknown_training_scheme_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="train_pooling"):
        LinearProbe(ProbeConfig(train_pooling="attention"))


def test_bags_whose_counts_do_not_match_their_labels_are_rejected() -> None:
    X, counts, y = bags(n=10)

    with pytest.raises(ValueError):
        LinearProbe().fit_bags(X, counts, y[:5])
