from __future__ import annotations

from pathlib import Path

import pytest

from byteprint.data import Sample, leave_one_generator_out, scan_split
from tests.conftest import write_image


def test_scan_split_labels_real_as_zero_and_fake_as_one(dataset_root: Path) -> None:
    samples = scan_split(dataset_root)

    labels = {s.path.name: s.label for s in samples}
    assert labels == {"a.jpg": 0, "b.png": 0, "c.png": 1, "d.png": 1, "e.png": 1}


def test_scan_split_reads_generator_from_subfolder(dataset_root: Path) -> None:
    samples = scan_split(dataset_root)

    generators = {s.path.name: s.generator for s in samples}
    assert generators["c.png"] == "sdxl"
    assert generators["e.png"] == "flux"


def test_real_samples_share_the_real_generator_label(dataset_root: Path) -> None:
    samples = scan_split(dataset_root)

    assert {s.generator for s in samples if s.label == 0} == {"real"}


def test_fake_images_outside_a_generator_folder_are_marked_unknown(tmp_path: Path) -> None:
    root = tmp_path / "train"
    write_image(root / "fake" / "loose.png", seed=7)

    samples = scan_split(root)

    assert [s.generator for s in samples] == ["unknown"]


def test_scan_split_ignores_non_image_files(dataset_root: Path) -> None:
    (dataset_root / "real" / "notes.txt").write_text("not an image")

    names = {s.path.name for s in scan_split(dataset_root)}

    assert "notes.txt" not in names


def test_scan_split_is_deterministically_ordered(dataset_root: Path) -> None:
    first = [s.path for s in scan_split(dataset_root)]
    second = [s.path for s in scan_split(dataset_root)]

    assert first == second == sorted(first)


def test_scan_split_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope"):
        scan_split(tmp_path / "nope")


def test_leave_one_generator_out_holds_that_generator_out_of_train(dataset_root: Path) -> None:
    samples = scan_split(dataset_root)

    train, held = leave_one_generator_out(samples, "sdxl")

    assert "sdxl" not in {s.generator for s in train}
    assert {s.generator for s in held} == {"sdxl"}


def test_leave_one_generator_out_keeps_all_real_images_in_train(dataset_root: Path) -> None:
    samples = scan_split(dataset_root)

    train, _ = leave_one_generator_out(samples, "sdxl")

    assert sum(1 for s in train if s.label == 0) == 2


def test_leave_one_generator_out_rejects_an_unknown_generator(dataset_root: Path) -> None:
    samples = scan_split(dataset_root)

    with pytest.raises(ValueError, match="midjourney"):
        leave_one_generator_out(samples, "midjourney")
