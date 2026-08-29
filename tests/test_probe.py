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
