"""The competition's required interface: an image directory in, a JSON file out."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from byteprint.cache import ExtractConfig
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
