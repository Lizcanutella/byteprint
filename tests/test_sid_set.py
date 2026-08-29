"""SID_Set arrives as HuggingFace parquet shards; byteprint wants an image tree."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from byteprint.sid_set import (
    LABEL_DIRS,
    encode_png,
    select_rows,
    write_image_tree,
)


def encoded(fmt: str, seed: int = 0, size: int = 32) -> bytes:
    """A real encoded image in ``fmt``, so the decode path is genuinely exercised."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


# -- the label mapping -----------------------------------------------------


def test_the_three_sid_set_labels_map_onto_the_binary_folder_layout() -> None:
    assert LABEL_DIRS[0] == Path("real")
    assert LABEL_DIRS[1] == Path("fake") / "full_synthetic"
    assert LABEL_DIRS[2] == Path("fake") / "tampered"


def test_tampered_images_land_under_their_own_generator_directory() -> None:
    # So that `logo --held-out tampered` can answer whether a detector trained
    # only on fully synthetic images transfers to locally edited photographs.
    assert LABEL_DIRS[2].parts[0] == "fake"
    assert LABEL_DIRS[2].parts[1] == "tampered"


# -- row selection ---------------------------------------------------------


def test_selection_takes_the_requested_quota_from_each_label() -> None:
    labels = {"a.parquet": [0] * 10 + [1] * 10, "b.parquet": [2] * 10}
    chosen = select_rows(labels, per_class=4, seed=0)
    flat = [(shard, row) for shard, rows in chosen.items() for row in rows]
    assert len(flat) == 12


def test_selection_is_deterministic_for_a_seed() -> None:
    labels = {"a.parquet": [0, 1, 2] * 20}
    assert select_rows(labels, per_class=5, seed=7) == select_rows(labels, per_class=5, seed=7)


def test_a_different_seed_selects_different_rows() -> None:
    labels = {"a.parquet": [0] * 100}
    assert select_rows(labels, per_class=5, seed=1) != select_rows(labels, per_class=5, seed=2)


def test_selection_spans_shards_rather_than_draining_the_first_one() -> None:
    # Shards are written in dataset order, so taking the first N rows would bias
    # the sample towards whatever the corpus happens to start with.
    labels = {f"{i}.parquet": [0] * 50 for i in range(10)}
    chosen = select_rows(labels, per_class=30, seed=0)
    assert len(chosen) > 1


def test_selection_returns_sorted_row_indices_for_a_sequential_read() -> None:
    labels = {"a.parquet": [0] * 100}
    rows = select_rows(labels, per_class=20, seed=0)["a.parquet"]
    assert rows == sorted(rows)


def test_a_quota_larger_than_the_corpus_takes_everything_available() -> None:
    labels = {"a.parquet": [0] * 3 + [1] * 40}
    chosen = select_rows(labels, per_class=10, seed=0)
    rows = [row for rows in chosen.values() for row in rows]
    assert len(rows) == 13


def test_labels_outside_the_three_sid_set_classes_are_ignored() -> None:
    labels = {"a.parquet": [0, 1, 2, 7]}
    chosen = select_rows(labels, per_class=10, seed=0)
    assert sum(len(rows) for rows in chosen.values()) == 3


# -- writing the tree ------------------------------------------------------


def test_every_class_is_re_encoded_to_one_container_format(tmp_path: Path) -> None:
    # Reals-as-JPEG against fakes-as-PNG yields a 99% container classifier. The
    # pixels keep whatever compression history they arrived with; the container
    # must not differ by class.
    records = [
        (0, "real_a", encoded("JPEG", seed=1)),
        (1, "synth_a", encoded("PNG", seed=2)),
        (2, "tamper_a", encoded("PNG", seed=3)),
    ]
    write_image_tree(records, tmp_path)
    written = sorted(p.suffix for p in tmp_path.rglob("*") if p.is_file())
    assert written == [".png", ".png", ".png"]


def test_each_image_lands_in_the_directory_its_label_names(tmp_path: Path) -> None:
    records = [
        (0, "real_a", encoded("JPEG", seed=1)),
        (1, "synth_a", encoded("PNG", seed=2)),
        (2, "tamper_a", encoded("PNG", seed=3)),
    ]
    write_image_tree(records, tmp_path)
    assert (tmp_path / "real" / "real_a.png").exists()
    assert (tmp_path / "fake" / "full_synthetic" / "synth_a.png").exists()
    assert (tmp_path / "fake" / "tampered" / "tamper_a.png").exists()


def test_the_source_format_mix_is_reported_so_the_shortcut_can_be_audited(
    tmp_path: Path,
) -> None:
    records = [
        (0, "real_a", encoded("JPEG", seed=1)),
        (0, "real_b", encoded("JPEG", seed=2)),
        (1, "synth_a", encoded("PNG", seed=3)),
    ]
    stats = write_image_tree(records, tmp_path)
    assert stats.source_formats == {(0, "JPEG"): 2, (1, "PNG"): 1}


def test_the_written_pixels_survive_the_round_trip(tmp_path: Path) -> None:
    payload = encoded("PNG", seed=11)
    write_image_tree([(1, "synth_a", payload)], tmp_path)
    original = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    stored = np.asarray(Image.open(tmp_path / "fake" / "full_synthetic" / "synth_a.png"))
    assert np.array_equal(original, stored)


def test_a_corrupt_record_is_counted_rather_than_killing_the_materialisation(
    tmp_path: Path,
) -> None:
    records = [(0, "good", encoded("PNG", seed=1)), (0, "bad", b"not an image")]
    stats = write_image_tree(records, tmp_path)
    assert stats.written == 1
    assert stats.failed == 1


def test_an_unknown_label_is_refused_rather_than_silently_filed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="label 7"):
        write_image_tree([(7, "mystery", encoded("PNG"))], tmp_path)


def test_counts_are_reported_per_label(tmp_path: Path) -> None:
    records = [
        (0, "r1", encoded("PNG", seed=1)),
        (0, "r2", encoded("PNG", seed=2)),
        (1, "s1", encoded("PNG", seed=3)),
    ]
    stats = write_image_tree(records, tmp_path)
    assert stats.by_label == {0: 2, 1: 1}


def test_encoding_to_png_normalises_mode_to_rgb() -> None:
    grey = Image.new("L", (16, 16), color=128)
    buffer = io.BytesIO()
    grey.save(buffer, format="PNG")
    result = Image.open(io.BytesIO(encode_png(buffer.getvalue())))
    assert result.mode == "RGB"
