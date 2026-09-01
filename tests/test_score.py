"""The competition's required interface: an image directory in, a JSON file out."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from byteprint.cache import ExtractConfig
from byteprint.crops import select_crops
from byteprint.data import scan_images
from byteprint.probe import LinearProbe, ProbeConfig
from byteprint.score import (
    ProbeScorer,
    UNSCORABLE,
    score_directory,
    write_predictions,
)

DIM = 8


class StubBackbone:
    name = "stub"
    dim = DIM

    def embed(self, crops):
        rows = [np.full(DIM, np.asarray(c, dtype=np.float64).mean() / 255.0) for c in crops]
        return np.stack(rows).astype(np.float32)


def config() -> ExtractConfig:
    return ExtractConfig(
        backbone="stub", dim=DIM, crop_size=28, crops_per_image=2,
        crop_mode="texture", seed=0,
    )


@pytest.fixture
def probe() -> LinearProbe:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 30)
    features = rng.normal(size=(60, DIM))
    features[labels == 1] += 1.5
    fitted = LinearProbe(ProbeConfig()).fit(features, labels)
    fitted.extract_config = config()
    return fitted


@pytest.fixture
def images(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    (root / "nested").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for name in ("a.png", "b.jpg"):
        Image.fromarray(
            rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ).save(root / name)
    Image.fromarray(
        rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    ).save(root / "nested" / "c.png")
    (root / "notes.txt").write_text("not an image")
    return root


def scorer(probe: LinearProbe) -> ProbeScorer:
    return ProbeScorer(backbone=StubBackbone(), probe=probe, config=config())


# -- discovery -------------------------------------------------------------


def test_scanning_finds_images_recursively(images: Path) -> None:
    found = scan_images(images)

    assert [p.name for p in found] == ["a.png", "b.jpg", "c.png"]


def test_scanning_ignores_non_image_files(images: Path) -> None:
    assert not any(p.suffix == ".txt" for p in scan_images(images))


def test_scanning_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        scan_images(tmp_path / "nope")


# -- scoring ---------------------------------------------------------------


def test_every_image_gets_exactly_one_prediction(images: Path, probe: LinearProbe) -> None:
    predictions = score_directory(images, scorer=scorer(probe))

    assert len(predictions) == 3
    assert len({p.image_path for p in predictions}) == 3


def test_predictions_are_likelihoods_between_zero_and_one(
    images: Path, probe: LinearProbe
) -> None:
    for prediction in score_directory(images, scorer=scorer(probe)):
        assert 0.0 <= prediction.pred <= 1.0


def test_predictions_are_ordered_deterministically(images: Path, probe: LinearProbe) -> None:
    first = [p.image_path for p in score_directory(images, scorer=scorer(probe))]
    second = [p.image_path for p in score_directory(images, scorer=scorer(probe))]

    assert first == second == sorted(first)


def test_scoring_is_reproducible_for_the_same_image(
    images: Path, probe: LinearProbe
) -> None:
    first = {p.image_path: p.pred for p in score_directory(images, scorer=scorer(probe))}
    second = {p.image_path: p.pred for p in score_directory(images, scorer=scorer(probe))}

    assert first == second


def test_paths_can_be_reported_relative_to_the_image_directory(
    images: Path, probe: LinearProbe
) -> None:
    predictions = score_directory(images, scorer=scorer(probe), relative=True)

    assert sorted(p.image_path for p in predictions) == ["a.png", "b.jpg", "nested/c.png"]


def test_absolute_paths_are_the_default(images: Path, probe: LinearProbe) -> None:
    predictions = score_directory(images, scorer=scorer(probe))

    assert all(Path(p.image_path).is_absolute() for p in predictions)


def test_an_empty_directory_produces_no_predictions(tmp_path: Path, probe: LinearProbe) -> None:
    (tmp_path / "empty").mkdir()

    assert score_directory(tmp_path / "empty", scorer=scorer(probe)) == []


def test_batching_does_not_change_any_score(images: Path, probe: LinearProbe) -> None:
    """Chunking is a speed optimisation; it must be invisible in the output."""
    one_at_a_time = score_directory(images, scorer=scorer(probe), chunk_size=1)
    batched = score_directory(images, scorer=scorer(probe), chunk_size=8)

    assert [p.pred for p in one_at_a_time] == [p.pred for p in batched]


# -- a corrupt file must not lose the whole run ----------------------------


def test_an_unreadable_image_still_gets_an_entry(
    images: Path, probe: LinearProbe
) -> None:
    (images / "broken.png").write_bytes(b"this is not a PNG")

    predictions = score_directory(images, scorer=scorer(probe))
    broken = [p for p in predictions if p.image_path.endswith("broken.png")]

    assert len(broken) == 1
    assert broken[0].pred == UNSCORABLE
    assert broken[0].error


def test_one_unreadable_image_does_not_stop_the_others(
    images: Path, probe: LinearProbe
) -> None:
    (images / "broken.png").write_bytes(b"nope")

    predictions = score_directory(images, scorer=scorer(probe))

    assert len([p for p in predictions if p.error is None]) == 3


def test_a_corrupt_directory_can_be_made_fatal_instead(
    images: Path, probe: LinearProbe
) -> None:
    (images / "broken.png").write_bytes(b"nope")

    with pytest.raises(OSError):
        score_directory(images, scorer=scorer(probe), strict=True)


# -- the JSON file itself --------------------------------------------------


def test_the_output_file_is_a_list_of_image_path_and_pred(
    tmp_path: Path, images: Path, probe: LinearProbe
) -> None:
    """The brief's wording, literally: a JSON file with image_path and pred per image."""
    out = tmp_path / "predictions.json"

    write_predictions(score_directory(images, scorer=scorer(probe)), out)
    payload = json.loads(out.read_text())

    assert isinstance(payload, list)
    assert len(payload) == 3
    for record in payload:
        assert set(record) == {"image_path", "pred"}
        assert isinstance(record["image_path"], str)
        assert isinstance(record["pred"], float)


def test_a_failed_image_carries_its_reason_in_the_json(
    tmp_path: Path, images: Path, probe: LinearProbe
) -> None:
    (images / "broken.png").write_bytes(b"nope")
    out = tmp_path / "predictions.json"

    write_predictions(score_directory(images, scorer=scorer(probe)), out)
    payload = json.loads(out.read_text())

    broken = [r for r in payload if r["image_path"].endswith("broken.png")]
    assert len(broken) == 1
    assert "error" in broken[0]


def test_the_output_directory_is_created_if_missing(
    tmp_path: Path, images: Path, probe: LinearProbe
) -> None:
    out = tmp_path / "deep" / "nested" / "predictions.json"

    write_predictions(score_directory(images, scorer=scorer(probe)), out)

    assert out.exists()


def test_the_json_is_valid_utf8_for_non_ascii_filenames(
    tmp_path: Path, probe: LinearProbe
) -> None:
    root = tmp_path / "imgs"
    root.mkdir()
    rng = np.random.default_rng(2)
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / "café.png")
    out = tmp_path / "p.json"

    write_predictions(score_directory(root, scorer=scorer(probe)), out)

    assert "café" in json.loads(out.read_text())[0]["image_path"]


# -- parallel loading ------------------------------------------------------
#
# This is the graded deliverable path, so the bar is higher than "it is faster":
# the predictions file must not depend on the worker count at all, and none of
# the guarantees above -- one entry per image, 0.5 on an unreadable file, strict
# still raising -- may weaken.


def many_images(root: Path, count: int = 12) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    for index in range(count):
        Image.fromarray(
            rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ).save(root / f"img_{index:03d}.png")
    return root


def test_parallel_scoring_returns_exactly_what_serial_scoring_returns(
    tmp_path: Path, probe: LinearProbe
) -> None:
    root = many_images(tmp_path / "many")

    serial = score_directory(root, scorer=scorer(probe), workers=1)
    parallel = score_directory(root, scorer=scorer(probe), workers=4)

    assert [p.image_path for p in parallel] == [p.image_path for p in serial]
    assert [p.pred for p in parallel] == [p.pred for p in serial]


def test_the_worker_count_does_not_change_a_prediction_even_across_chunks(
    tmp_path: Path, probe: LinearProbe
) -> None:
    # Crops are seeded by the image's position within its chunk, so chunking and
    # ordering have to survive the pool untouched.
    root = many_images(tmp_path / "many", 20)

    serial = score_directory(root, scorer=scorer(probe), workers=1, chunk_size=3)
    parallel = score_directory(root, scorer=scorer(probe), workers=4, chunk_size=3)

    assert [p.pred for p in parallel] == [p.pred for p in serial]


def test_parallel_scoring_still_gives_every_image_exactly_one_entry(
    images: Path, probe: LinearProbe
) -> None:
    predictions = score_directory(images, scorer=scorer(probe), workers=4)

    assert len(predictions) == 3


def test_an_unreadable_file_is_still_reported_at_maximum_uncertainty_in_parallel(
    tmp_path: Path, probe: LinearProbe
) -> None:
    root = many_images(tmp_path / "many", 6)
    (root / "truncated.png").write_bytes(b"not an image")

    predictions = score_directory(root, scorer=scorer(probe), workers=4)
    broken = [p for p in predictions if p.image_path.endswith("truncated.png")]

    assert len(broken) == 1
    assert broken[0].pred == UNSCORABLE
    assert broken[0].error


def test_strict_still_raises_on_the_first_unreadable_file_in_parallel(
    tmp_path: Path, probe: LinearProbe
) -> None:
    root = many_images(tmp_path / "many", 6)
    (root / "truncated.png").write_bytes(b"not an image")

    with pytest.raises((OSError, ValueError)):
        score_directory(root, scorer=scorer(probe), workers=4, strict=True)


def test_the_scorer_is_only_called_from_the_calling_thread(
    tmp_path: Path, probe: LinearProbe
) -> None:
    root = many_images(tmp_path / "many")
    seen: set[int] = set()

    class ThreadRecordingScorer(ProbeScorer):
        def score_images(self, images):
            seen.add(threading.get_ident())
            return super().score_images(images)

    score_directory(
        root,
        scorer=ThreadRecordingScorer(backbone=StubBackbone(), probe=probe, config=config()),
        workers=4,
    )

    assert seen == {threading.get_ident()}


def test_a_worker_count_below_one_is_refused(images: Path, probe: LinearProbe) -> None:
    with pytest.raises(ValueError, match="workers"):
        score_directory(images, scorer=scorer(probe), workers=0)


# -- parallel crop selection ----------------------------------------------
#
# Choosing crops costs more than decoding does, so the scorer parallelises its
# own crop loop. Same bar: identical numbers out.


def test_the_scorer_gives_identical_scores_whatever_its_worker_count(
    probe: LinearProbe,
) -> None:
    rng = np.random.default_rng(3)
    batch = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(8)]

    serial = ProbeScorer(backbone=StubBackbone(), probe=probe, config=config(), workers=1)
    parallel = ProbeScorer(backbone=StubBackbone(), probe=probe, config=config(), workers=4)

    assert np.array_equal(parallel.score_images(batch), serial.score_images(batch))


def test_the_scorer_refuses_a_worker_count_below_one(probe: LinearProbe) -> None:
    with pytest.raises(ValueError, match="workers"):
        ProbeScorer(backbone=StubBackbone(), probe=probe, config=config(), workers=0)


# -- the deliverable honours the probe's pooling ---------------------------


class VaryingBackbone:
    """Crops differ in *direction*, not only magnitude.

    :class:`StubBackbone` returns a constant vector per crop, which the probe's
    row-wise L2 normalisation collapses to one identical unit vector -- so
    every pooling gives the same answer and nothing about pooling can be
    measured through it. Spreading the crop statistic across two axes is enough
    to make the crops distinguishable.
    """

    name = "stub"
    dim = DIM

    def embed(self, crops):
        rows = []
        for crop in crops:
            level = float(np.asarray(crop, dtype=np.float64).mean()) / 255.0
            row = np.zeros(DIM)
            row[0] = level
            row[1] = 1.0 - level
            rows.append(row)
        return np.stack(rows).astype(np.float32)


def bag_probe(pooling: str) -> LinearProbe:
    """A probe fitted on bags, carrying `pooling` as its own setting."""
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 30)
    crops = np.full(60, 2, dtype=np.int64)
    features = np.zeros((120, DIM))
    features[:, 0] = rng.uniform(0.0, 1.0, size=120)
    features[:, 1] = 1.0 - features[:, 0]
    # Fakes carry the signal in their first crop only, as a tampered image does.
    features[np.arange(60, 120, 2), 0] += 1.5
    fitted = LinearProbe(ProbeConfig(pooling=pooling)).fit_bags(features, crops, labels)
    fitted.extract_config = config()
    return fitted


def test_the_required_json_interface_reflects_the_probes_pooling(images: Path) -> None:
    # The deliverable must not quietly mean-pool a probe that was trained and
    # published as a max-pooled detector.
    batch = [
        np.asarray(Image.open(images / name).convert("RGB")) for name in ("a.png", "b.jpg")
    ]

    pooled = ProbeScorer(
        backbone=VaryingBackbone(), probe=bag_probe("mean"), config=config()
    ).score_images(batch)
    peaked = ProbeScorer(
        backbone=VaryingBackbone(), probe=bag_probe("max"), config=config()
    ).score_images(batch)

    assert pooled.shape == peaked.shape == (2,)
    assert not np.allclose(pooled, peaked)


def test_a_max_pooled_scorer_returns_each_images_most_confident_crop(images: Path) -> None:
    probe = bag_probe("max")
    backbone = VaryingBackbone()
    scorer_ = ProbeScorer(backbone=backbone, probe=probe, config=config())
    image = np.asarray(Image.open(images / "a.png").convert("RGB"))

    crops = select_crops(image, crop_size=28, top_k=2, mode="texture", seed=0)
    embedded = backbone.embed(crops)

    assert scorer_.score_images([image])[0] == pytest.approx(probe.score(embedded).max())
