from __future__ import annotations

import numpy as np
import pytest

from byteprint.pooling import (
    DEFAULT_POOLING,
    POOLINGS,
    bag_offsets,
    repeat_labels,
    resolve_pooling,
    segment_reduce,
    take_bags,
    truncate_bags,
)


# -- the registry ----------------------------------------------------------


def test_the_default_pooling_is_the_behaviour_this_project_already_shipped() -> None:
    assert DEFAULT_POOLING == "mean"
    assert resolve_pooling(DEFAULT_POOLING).space == "feature"


def test_mean_is_the_only_pooling_that_reduces_in_feature_space() -> None:
    spaces = {name: POOLINGS[name].space for name in POOLINGS.names()}

    assert spaces == {
        "max": "score",
        "mean": "feature",
        "mean-score": "score",
        "topk": "score",
    }


def test_topk_takes_its_k_from_the_name() -> None:
    pooling = resolve_pooling("topk:3")

    assert pooling.name == "topk:3"
    assert pooling.space == "score"


def test_an_unknown_pooling_lists_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="unknown pooling"):
        resolve_pooling("median")


def test_topk_without_a_k_is_rejected_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="topk"):
        resolve_pooling("topk")


def test_a_non_numeric_or_non_positive_k_is_rejected() -> None:
    for spec in ("topk:zero", "topk:0", "topk:-2"):
        with pytest.raises(ValueError):
            resolve_pooling(spec)


def test_a_pooling_that_takes_no_argument_refuses_one() -> None:
    with pytest.raises(ValueError, match="takes no argument"):
        resolve_pooling("max:2")


# -- ragged bags -----------------------------------------------------------


def test_offsets_are_the_running_start_of_each_bag() -> None:
    assert bag_offsets(np.array([2, 3, 1])).tolist() == [0, 2, 5]


def test_a_bag_may_hold_fewer_crops_than_its_neighbours() -> None:
    # center and resize return one crop whatever --crops says, and an image
    # smaller than the crop size can yield fewer. Ragged is the normal case.
    values = np.array([1.0, 3.0, 10.0, 100.0, 200.0, 300.0])
    counts = np.array([2, 1, 3])

    pooled = segment_reduce(values, counts, lambda v: v.mean(axis=0))

    assert pooled.tolist() == [2.0, 10.0, 200.0]


def test_reducing_a_matrix_keeps_one_row_per_bag() -> None:
    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    counts = np.array([4, 2])

    pooled = segment_reduce(values, counts, lambda v: v.mean(axis=0))

    assert pooled.shape == (2, 2)
    assert np.allclose(pooled[0], values[:4].mean(axis=0))
    assert np.allclose(pooled[1], values[4:].mean(axis=0))


def test_counts_that_do_not_account_for_every_row_are_a_bug_not_a_truncation() -> None:
    with pytest.raises(ValueError, match="rows"):
        segment_reduce(np.zeros(5), np.array([2, 2]), lambda v: v.mean(axis=0))


def test_an_empty_bag_list_reduces_to_an_empty_result() -> None:
    pooled = segment_reduce(np.zeros((0, 3)), np.array([], dtype=int), lambda v: v.mean(axis=0))

    assert pooled.shape == (0, 3)


def test_labels_are_inherited_by_every_crop_of_their_image() -> None:
    # Instance-level training: each crop is a row carrying its image's label.
    assert repeat_labels(np.array([1, 0, 1]), np.array([2, 1, 3])).tolist() == [
        1, 1, 0, 1, 1, 1
    ]


# -- truncation ------------------------------------------------------------


def test_truncating_keeps_each_bags_first_crops() -> None:
    values = np.arange(6, dtype=np.float64)
    counts = np.array([3, 3])

    kept, new_counts = truncate_bags(values, counts, 2)

    assert kept.tolist() == [0.0, 1.0, 3.0, 4.0]
    assert new_counts.tolist() == [2, 2]


def test_truncating_never_pads_a_bag_that_is_already_short() -> None:
    values = np.arange(4, dtype=np.float64)
    counts = np.array([1, 3])

    kept, new_counts = truncate_bags(values, counts, 2)

    assert kept.tolist() == [0.0, 1.0, 2.0]
    assert new_counts.tolist() == [1, 2]


def test_no_limit_leaves_the_bags_untouched() -> None:
    values = np.arange(5, dtype=np.float64)
    counts = np.array([2, 3])

    kept, new_counts = truncate_bags(values, counts, None)

    assert kept.tolist() == values.tolist()
    assert new_counts.tolist() == counts.tolist()


def test_a_limit_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        truncate_bags(np.zeros(2), np.array([2]), 0)


# -- selecting whole bags --------------------------------------------------


def test_selecting_images_keeps_each_ones_crops_together() -> None:
    # A train/calibration split indexes images. Splitting crop rows directly
    # would put one image's crops on both sides of the split.
    values = np.arange(6, dtype=np.float64)
    counts = np.array([1, 2, 3])

    taken, new_counts = take_bags(values, counts, np.array([2, 0]))

    assert taken.tolist() == [3.0, 4.0, 5.0, 0.0]
    assert new_counts.tolist() == [3, 1]


def test_a_boolean_mask_selects_bags_too() -> None:
    values = np.arange(6, dtype=np.float64)
    counts = np.array([1, 2, 3])

    taken, new_counts = take_bags(values, counts, np.array([False, True, False]))

    assert taken.tolist() == [1.0, 2.0]
    assert new_counts.tolist() == [2]


def test_selecting_no_images_yields_nothing_rather_than_failing() -> None:
    taken, new_counts = take_bags(np.arange(3.0), np.array([1, 2]), np.array([], dtype=int))

    assert taken.shape == (0,)
    assert new_counts.shape == (0,)


# -- the reductions themselves ---------------------------------------------


def bag(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def test_mean_pooling_reproduces_the_average_the_cache_used_to_store() -> None:
    # The behaviour this refactor must not change: pooling a bag in feature
    # space is exactly crop_features.mean(axis=0).
    crops = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])

    reduced = resolve_pooling("mean").reduce(crops)

    assert np.allclose(reduced, crops.mean(axis=0))
    assert np.allclose(reduced, [3.0, 6.0])


def test_max_pooling_takes_the_single_most_confident_crop() -> None:
    assert resolve_pooling("max").reduce(bag([0.1, 0.9, 0.3])) == pytest.approx(0.9)


def test_mean_score_pooling_averages_probabilities_not_features() -> None:
    assert resolve_pooling("mean-score").reduce(bag([0.1, 0.9, 0.2])) == pytest.approx(0.4)


def test_topk_averages_the_k_most_confident_crops() -> None:
    assert resolve_pooling("topk:2").reduce(bag([0.1, 0.9, 0.5, 0.3])) == pytest.approx(0.7)


def test_topk_of_one_is_the_maximum() -> None:
    values = bag([0.2, 0.7, 0.4])

    assert resolve_pooling("topk:1").reduce(values) == resolve_pooling("max").reduce(values)


def test_topk_uses_every_crop_when_a_bag_is_smaller_than_k() -> None:
    # A center-mode bag holds one crop. Asking for the top 4 of it is not an
    # error; it is the mean of what is there.
    assert resolve_pooling("topk:4").reduce(bag([0.25, 0.75])) == pytest.approx(0.5)
