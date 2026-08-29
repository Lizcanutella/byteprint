from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from byteprint.cache import EmbeddingStore, ExtractConfig
from byteprint.fusion import FusedDetector, join_caches
from byteprint.metrics import roc_auc
from byteprint.probe import LinearProbe


def store(tmp_path: Path, name: str, dim: int) -> EmbeddingStore:
    config = ExtractConfig(
        backbone=name, dim=dim, crop_size=28, crops_per_image=1, crop_mode="texture", seed=0
    )
    return EmbeddingStore.open(tmp_path / name, config)


def put(s: EmbeddingStore, key: str, value: float, *, label: int = 1, gen: str = "sdxl") -> None:
    s.add(
        key,
        np.full((1, s.config.dim), value, dtype=np.float32),
        path=Path(f"/data/{key}.png"),
        label=label,
        generator=gen,
        spec="none",
    )


def correlated(n: int = 120, seed: int = 0):
    """Two weak, independent experts: each alone is mediocre, together they are better."""
    rng = np.random.default_rng(seed)
    dino = np.vstack([rng.normal(0, 1, (n, 1)), rng.normal(1.2, 1, (n, 1))])
    recon = np.vstack([rng.normal(0, 1, (n, 1)), rng.normal(-1.2, 1, (n, 1))])
    labels = np.array([0] * n + [1] * n)
    return dino.astype(np.float32), recon.astype(np.float32), labels


def fitted_probe(dino, labels) -> LinearProbe:
    return LinearProbe().fit(dino, labels)


def test_joining_keeps_only_rows_present_in_both_caches(tmp_path: Path) -> None:
    a, b = store(tmp_path, "dino", 2), store(tmp_path, "recon", 1)
    put(a, "x", 1.0), put(a, "y", 2.0)
    put(b, "x", 0.5)

    joined = join_caches(a, b)

    assert joined.n == 1


def test_joining_aligns_the_two_feature_matrices_by_key(tmp_path: Path) -> None:
    a, b = store(tmp_path, "dino", 2), store(tmp_path, "recon", 1)
    put(a, "x", 1.0), put(a, "y", 2.0)
    put(b, "y", 0.25), put(b, "x", 0.75)  # inserted in the opposite order

    joined = join_caches(a, b)

    for dino_row, recon_row in zip(joined.dino, joined.recon, strict=True):
        expected = 0.75 if dino_row[0] == 1.0 else 0.25
        assert recon_row[0] == pytest.approx(expected)


def test_joining_carries_labels_and_generators_through(tmp_path: Path) -> None:
    a, b = store(tmp_path, "dino", 2), store(tmp_path, "recon", 1)
    put(a, "r", 1.0, label=0, gen="real"), put(a, "f", 2.0, label=1, gen="flux")
    put(b, "r", 0.5, label=0, gen="real"), put(b, "f", 0.1, label=1, gen="flux")

    joined = join_caches(a, b)

    assert sorted(joined.labels.tolist()) == [0, 1]
    assert sorted(joined.generators) == ["flux", "real"]


def test_joining_disjoint_caches_yields_nothing(tmp_path: Path) -> None:
    a, b = store(tmp_path, "dino", 2), store(tmp_path, "recon", 1)
    put(a, "x", 1.0)
    put(b, "y", 0.5)

    assert join_caches(a, b).n == 0


def test_joining_is_deterministically_ordered(tmp_path: Path) -> None:
    a, b = store(tmp_path, "dino", 2), store(tmp_path, "recon", 1)
    for key in ("c", "a", "b"):
        put(a, key, 1.0), put(b, key, 0.5)

    assert join_caches(a, b).keys == sorted(join_caches(a, b).keys)


def test_a_key_labelled_differently_in_the_two_caches_is_an_error(tmp_path: Path) -> None:
    a, b = store(tmp_path, "dino", 2), store(tmp_path, "recon", 1)
    put(a, "x", 1.0, label=1)
    put(b, "x", 0.5, label=0)

    with pytest.raises(ValueError, match="disagree"):
        join_caches(a, b)


def test_fusion_beats_either_expert_on_its_own() -> None:
    dino, recon, labels = correlated()
    detector = FusedDetector(fitted_probe(dino, labels)).fit(dino, recon, labels)

    parts = detector.component_scores(dino, recon)

    fused = roc_auc(labels, parts["fused"])
    assert fused > roc_auc(labels, parts["probe"])
    assert fused > roc_auc(labels, parts["recon"])


def test_component_scores_expose_all_three_views() -> None:
    dino, recon, labels = correlated()
    detector = FusedDetector(fitted_probe(dino, labels)).fit(dino, recon, labels)

    assert set(detector.component_scores(dino, recon)) == {"probe", "recon", "fused"}


def test_the_reconstruction_view_is_the_negated_minimum_distance() -> None:
    dino, _, labels = correlated(n=10)
    recon = np.tile(np.array([[0.4, 0.1]], dtype=np.float32), (len(labels), 1))
    detector = FusedDetector(fitted_probe(dino, labels)).fit(dino, recon, labels)

    assert detector.component_scores(dino, recon)["recon"] == pytest.approx(-0.1)


def test_fused_scores_are_probabilities() -> None:
    dino, recon, labels = correlated()
    detector = FusedDetector(fitted_probe(dino, labels)).fit(dino, recon, labels)

    scores = detector.score(dino, recon)

    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_an_unfitted_detector_refuses_to_score() -> None:
    dino, recon, labels = correlated(n=5)

    with pytest.raises(RuntimeError, match="not fitted"):
        FusedDetector(fitted_probe(dino, labels)).score(dino, recon)


def test_calibrating_to_a_zero_budget_produces_no_false_positives() -> None:
    dino, recon, labels = correlated()
    detector = FusedDetector(fitted_probe(dino, labels)).fit(dino, recon, labels)
    detector.calibrate(dino, recon, labels, target_fpr=0.0)

    assert detector.predict(dino, recon)[labels == 0].sum() == 0


def test_a_saved_detector_reloads_with_identical_scores(tmp_path: Path) -> None:
    dino, recon, labels = correlated()
    detector = FusedDetector(fitted_probe(dino, labels)).fit(dino, recon, labels)
    detector.calibrate(dino, recon, labels, target_fpr=0.01)
    detector.save(tmp_path / "fused.joblib")

    reloaded = FusedDetector.load(tmp_path / "fused.joblib")

    assert np.allclose(reloaded.score(dino, recon), detector.score(dino, recon))
    assert reloaded.threshold == detector.threshold


def test_fitting_rejects_mismatched_lengths() -> None:
    dino, recon, labels = correlated()

    with pytest.raises(ValueError, match="same length"):
        FusedDetector(fitted_probe(dino, labels)).fit(dino, recon[:-1], labels)


def test_fitting_needs_both_classes() -> None:
    dino, recon, labels = correlated()

    with pytest.raises(ValueError, match="both classes"):
        FusedDetector(fitted_probe(dino, labels)).fit(dino, recon, np.ones_like(labels))
