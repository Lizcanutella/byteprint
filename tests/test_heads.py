from __future__ import annotations

import numpy as np
import pytest

from byteprint.heads import HEADS, build_head, register_head
from byteprint.probe import LinearProbe, ProbeConfig


def separable(n: int = 120, dim: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two overlapping Gaussians: learnable, but not trivially so."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], n // 2)
    features = rng.normal(size=(n, dim))
    features[labels == 1, 0] += 2.0
    return features, labels


def test_the_default_head_is_logistic_regression() -> None:
    assert ProbeConfig().head == "logreg"


def test_every_shipped_head_is_registered() -> None:
    assert set(HEADS.names()) >= {"logreg", "linear-svm", "mlp"}


@pytest.mark.parametrize("head", ["logreg", "linear-svm", "mlp"])
def test_every_shipped_head_trains_and_separates(head: str) -> None:
    features, labels = separable()

    probe = LinearProbe(ProbeConfig(head=head)).fit(features, labels)
    scores = probe.score(features)

    assert scores.shape == (len(labels),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


@pytest.mark.parametrize("head", ["logreg", "linear-svm", "mlp"])
def test_every_shipped_head_survives_a_round_trip(head: str, tmp_path) -> None:
    features, labels = separable()
    probe = LinearProbe(ProbeConfig(head=head)).fit(features, labels)
    probe.calibrate(features, labels, target_fpr=0.05)
    path = tmp_path / "probe.joblib"
    probe.save(path)

    reloaded = LinearProbe.load(path)

    assert reloaded.config.head == head
    assert np.allclose(reloaded.score(features), probe.score(features))
    assert reloaded.threshold == probe.threshold


def test_an_unknown_head_is_reported_with_the_alternatives() -> None:
    features, labels = separable()

    with pytest.raises(ValueError, match="logreg"):
        LinearProbe(ProbeConfig(head="transformer")).fit(features, labels)


def test_a_teammate_can_register_a_head_from_their_own_module() -> None:
    from sklearn.naive_bayes import GaussianNB

    @register_head("test-naive-bayes")
    def _build(config: ProbeConfig) -> GaussianNB:
        return GaussianNB()

    try:
        features, labels = separable()
        probe = LinearProbe(ProbeConfig(head="test-naive-bayes")).fit(features, labels)

        assert probe.score(features).shape == (len(labels),)
    finally:
        HEADS._entries.pop("test-naive-bayes", None)


def test_a_head_factory_receives_the_probe_config() -> None:
    config = ProbeConfig(head="logreg", C=0.25, seed=7)

    estimator = build_head(config)

    assert estimator.C == 0.25
    assert estimator.random_state == 7
