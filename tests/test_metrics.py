from __future__ import annotations

import numpy as np
import pytest

from byteprint.metrics import evaluate, roc_auc, threshold_at_fpr, tpr_at_fpr


def test_perfect_ranking_scores_auc_one() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_exactly_inverted_ranking_scores_auc_zero() -> None:
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_all_tied_scores_give_chance_auc() -> None:
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auc_matches_the_hand_counted_pairwise_value() -> None:
    # 4 (positive, negative) pairs; the positive ranks higher in 3 of them.
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auc_requires_both_classes_to_be_present() -> None:
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([1, 1, 1], [0.2, 0.4, 0.9])


def test_a_zero_false_positive_budget_admits_no_negative() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.9, 0.5, 0.95]

    threshold = threshold_at_fpr(labels, scores, 0.0)

    assert threshold > 0.9


def test_tpr_at_a_zero_false_positive_budget() -> None:
    # Only the 0.95 positive clears the highest negative at 0.9.
    assert tpr_at_fpr([0, 0, 1, 1], [0.1, 0.9, 0.5, 0.95], 0.0) == pytest.approx(0.5)


def test_a_looser_budget_recovers_more_true_positives() -> None:
    # Spending one of two negatives lets the 0.5 positive through as well.
    assert tpr_at_fpr([0, 0, 1, 1], [0.1, 0.9, 0.5, 0.95], 0.5) == pytest.approx(1.0)


def test_perfectly_separated_scores_reach_full_tpr_at_zero_fpr() -> None:
    assert tpr_at_fpr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.0) == pytest.approx(1.0)


def test_evaluate_reports_auc_and_every_requested_fpr_target() -> None:
    report = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], fpr_targets=(0.01, 0.001))

    assert report.auc == pytest.approx(1.0)
    assert set(report.tpr_at_fpr) == {0.01, 0.001}


def test_evaluate_scores_each_generator_against_the_shared_real_pool() -> None:
    labels = [0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.9, 0.95, 0.05, 0.15]
    generators = ["real", "real", "sdxl", "sdxl", "flux", "flux"]

    report = evaluate(labels, scores, generators=generators)

    assert report.per_generator["sdxl"].auc == pytest.approx(1.0)
    assert report.per_generator["flux"].auc == pytest.approx(0.25)


def test_evaluate_counts_the_samples_behind_each_generator() -> None:
    report = evaluate(
        [0, 0, 1, 1, 1],
        [0.1, 0.2, 0.9, 0.95, 0.8],
        generators=["real", "real", "sdxl", "sdxl", "flux"],
    )

    assert report.per_generator["sdxl"].n_fake == 2
    assert report.per_generator["flux"].n_fake == 1


def test_evaluate_rejects_mismatched_input_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate([0, 1], [0.1, 0.2, 0.3])


def test_report_renders_a_table_naming_each_generator() -> None:
    report = evaluate(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.9, 0.95],
        generators=["real", "real", "sdxl", "flux"],
    )

    rendered = report.render()

    assert "sdxl" in rendered and "flux" in rendered and "AUC" in rendered


def test_numpy_inputs_are_accepted() -> None:
    auc = roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))

    assert auc == pytest.approx(1.0)
