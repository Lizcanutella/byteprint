"""Re-encoding a split so compression history cannot be the classifier.

SID_Set's reals are 100% JPEG-family and its fully-synthetic images 100% PNG.
Materialisation already equalises the *container* by writing every class as
PNG, but the reals still carry JPEG quantisation artifacts in their pixels and
the synthetics do not. This module is the control that puts both classes
through the same encoder before extraction.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from byteprint.recompress import (
    RecompressStats,
    encode_image,
    parse_encoding,
    recompress_split,
)


def write_image(path: Path, fmt: str, seed: int = 0, size: int = 64) -> Path:
    """A real encoded image on disk, so the decode path is genuinely exercised."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format=fmt)
    return path


def a_split(root: Path) -> Path:
    """A minimal real/ + fake/<generator>/ tree."""
    write_image(root / "real" / "r0.png", "PNG", seed=0)
    write_image(root / "real" / "r1.png", "PNG", seed=1)
    write_image(root / "fake" / "full_synthetic" / "f0.png", "PNG", seed=2)
    write_image(root / "fake" / "tampered" / "t0.png", "PNG", seed=3)
    return root


# -- the encoding spec -----------------------------------------------------


def test_a_quality_spec_names_the_format_and_its_quality() -> None:
    assert parse_encoding("jpeg:95") == ("JPEG", 95)


def test_png_needs_no_quality() -> None:
    assert parse_encoding("png") == ("PNG", None)


def test_an_unknown_format_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="unknown encoding"):
        parse_encoding("webp:80")


def test_a_jpeg_spec_without_a_quality_is_refused() -> None:
    # Silently defaulting the quality would make two runs incomparable.
    with pytest.raises(ValueError, match="quality"):
        parse_encoding("jpeg")


def test_a_quality_outside_the_jpeg_range_is_refused() -> None:
    with pytest.raises(ValueError, match="quality"):
        parse_encoding("jpeg:101")


# -- encoding one image ----------------------------------------------------


def test_encoding_returns_bytes_in_the_requested_container() -> None:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(buffer, format="PNG")
    encoded, suffix = encode_image(buffer.getvalue(), "jpeg:95")
    assert suffix == ".jpg"
    with Image.open(io.BytesIO(encoded)) as handle:
        assert handle.format == "JPEG"


def test_encoding_to_png_keeps_the_pixels_exactly() -> None:
    array = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded, suffix = encode_image(buffer.getvalue(), "png")
    assert suffix == ".png"
    with Image.open(io.BytesIO(encoded)) as handle:
        assert np.array_equal(np.asarray(handle.convert("RGB")), array)


def test_a_jpeg_re_encode_actually_changes_the_pixels() -> None:
    # The whole point of the control: both classes take the same lossy damage.
    array = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded, _ = encode_image(buffer.getvalue(), "jpeg:95")
    with Image.open(io.BytesIO(encoded)) as handle:
        assert not np.array_equal(np.asarray(handle.convert("RGB")), array)


def test_a_greyscale_source_is_normalised_to_rgb() -> None:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(buffer, format="PNG")
    encoded, _ = encode_image(buffer.getvalue(), "jpeg:95")
    with Image.open(io.BytesIO(encoded)) as handle:
        assert handle.convert("RGB").size == (16, 16)


# -- walking a split -------------------------------------------------------


def test_every_image_in_the_split_is_re_encoded(tmp_path: Path) -> None:
    stats = recompress_split(a_split(tmp_path / "src"), tmp_path / "dst", encoding="jpeg:95")
    assert stats.written == 4
    assert sorted(p.name for p in (tmp_path / "dst").rglob("*.jpg")) == [
        "f0.jpg",
        "r0.jpg",
        "r1.jpg",
        "t0.jpg",
    ]


def test_the_label_tree_is_preserved_so_the_split_still_reads(tmp_path: Path) -> None:
    from byteprint.data import scan_split

    recompress_split(a_split(tmp_path / "src"), tmp_path / "dst", encoding="jpeg:95")
    samples = scan_split(tmp_path / "dst")
    assert len(samples) == 4
    assert sorted({s.generator for s in samples}) == ["full_synthetic", "real", "tampered"]


def test_both_classes_come_out_in_the_same_container(tmp_path: Path) -> None:
    # This is the control's entire purpose, so it is asserted directly.
    recompress_split(a_split(tmp_path / "src"), tmp_path / "dst", encoding="jpeg:95")
    formats = set()
    for path in (tmp_path / "dst").rglob("*"):
        if path.is_file():
            with Image.open(path) as handle:
                formats.add(handle.format)
    assert formats == {"JPEG"}


def test_the_same_image_set_is_carried_over_so_runs_stay_comparable(tmp_path: Path) -> None:
    # A control that also changed which images are in the split would confound
    # the very thing it exists to isolate.
    recompress_split(a_split(tmp_path / "src"), tmp_path / "dst", encoding="jpeg:95")

    def stems(root: Path) -> list[str]:
        return sorted(p.stem for p in root.rglob("*") if p.is_file())

    assert stems(tmp_path / "dst") == stems(tmp_path / "src")


def test_an_unreadable_image_is_counted_rather_than_killing_the_run(tmp_path: Path) -> None:
    src = a_split(tmp_path / "src")
    (src / "real" / "truncated.png").write_bytes(b"not an image")
    stats = recompress_split(src, tmp_path / "dst", encoding="jpeg:95")
    assert stats.written == 4
    assert stats.failed == 1


def test_a_rerun_skips_what_it_already_wrote(tmp_path: Path) -> None:
    # Recompressing 16k images should resume after a timeout, not repeat.
    src, dst = a_split(tmp_path / "src"), tmp_path / "dst"
    recompress_split(src, dst, encoding="jpeg:95")
    stats = recompress_split(src, dst, encoding="jpeg:95")
    assert stats.written == 0
    assert stats.skipped == 4


def test_the_source_container_mix_is_reported_for_the_audit(tmp_path: Path) -> None:
    src = tmp_path / "src"
    write_image(src / "real" / "r0.jpg", "JPEG", seed=0)
    write_image(src / "fake" / "full_synthetic" / "f0.png", "PNG", seed=1)
    stats = recompress_split(src, tmp_path / "dst", encoding="jpeg:95")
    assert stats.source_formats[("real", "JPEG")] == 1
    assert stats.source_formats[("full_synthetic", "PNG")] == 1


def test_the_stats_render_as_a_readable_summary(tmp_path: Path) -> None:
    stats = recompress_split(a_split(tmp_path / "src"), tmp_path / "dst", encoding="jpeg:95")
    rendered = stats.render()
    assert "4" in rendered and "jpeg:95" in rendered


def test_workers_do_not_change_the_result(tmp_path: Path) -> None:
    src = a_split(tmp_path / "src")
    recompress_split(src, tmp_path / "one", encoding="jpeg:95", workers=1)
    recompress_split(src, tmp_path / "four", encoding="jpeg:95", workers=4)
    for single in (tmp_path / "one").rglob("*"):
        if single.is_file():
            twin = tmp_path / "four" / single.relative_to(tmp_path / "one")
            assert single.read_bytes() == twin.read_bytes()


def test_a_missing_source_split_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        recompress_split(tmp_path / "nope", tmp_path / "dst", encoding="jpeg:95")


def test_stats_start_empty() -> None:
    assert RecompressStats().written == 0
