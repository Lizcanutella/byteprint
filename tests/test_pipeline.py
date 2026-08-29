from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from byteprint.cache import EmbeddingStore, ExtractConfig
from byteprint.data import scan_split
from byteprint.pipeline import extract
from tests.conftest import write_image

DIM = 4


class CountingBackbone:
    """Fake backbone: records every batch of crops it is asked to embed."""

    name = "counting"
    dim = DIM

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed(self, crops) -> np.ndarray:
        self.calls.append(len(crops))
        return np.stack(
            [np.full(DIM, float(np.asarray(c).mean()), dtype=np.float32) for c in crops]
        )


def make_store(tmp_path: Path, **overrides) -> EmbeddingStore:
    settings = dict(
        backbone="counting",
        dim=DIM,
        crop_size=28,
        crops_per_image=2,
        crop_mode="texture",
        seed=0,
    )
    settings.update(overrides)
    return EmbeddingStore.open(tmp_path / "cache", ExtractConfig(**settings))


def test_extraction_stores_one_row_per_image(dataset_root: Path, tmp_path: Path) -> None:
    store = make_store(tmp_path)

    extract(scan_split(dataset_root), backbone=CountingBackbone(), store=store)

    assert len(store) == 5


def test_extraction_reports_what_it_added(dataset_root: Path, tmp_path: Path) -> None:
    stats = extract(scan_split(dataset_root), backbone=CountingBackbone(), store=make_store(tmp_path))

    assert stats.added == 5
    assert stats.skipped == 0
    assert stats.failed == 0


def test_each_image_is_embedded_with_the_configured_number_of_crops(
    dataset_root: Path, tmp_path: Path
) -> None:
    backbone = CountingBackbone()

    extract(scan_split(dataset_root), backbone=backbone, store=make_store(tmp_path))

    assert backbone.calls == [2, 2, 2, 2, 2]


def test_a_second_run_skips_everything_already_cached(dataset_root: Path, tmp_path: Path) -> None:
    samples = scan_split(dataset_root)
    store = make_store(tmp_path)
    extract(samples, backbone=CountingBackbone(), store=store)
    store.flush()

    backbone = CountingBackbone()
    stats = extract(samples, backbone=backbone, store=make_store(tmp_path))

    assert stats.skipped == 5
    assert backbone.calls == []


def test_labels_and_generators_survive_into_the_store(dataset_root: Path, tmp_path: Path) -> None:
    store = make_store(tmp_path)

    extract(scan_split(dataset_root), backbone=CountingBackbone(), store=store)

    assert store.labels().tolist() == [1, 1, 1, 0, 0]  # scan order: fake/ sorts before real/
    assert sorted(set(store.generators())) == ["flux", "real", "sdxl"]


def test_each_laundering_spec_adds_its_own_row(dataset_root: Path, tmp_path: Path) -> None:
    store = make_store(tmp_path)

    extract(
        scan_split(dataset_root),
        backbone=CountingBackbone(),
        store=store,
        specs=("none", "jpeg:40"),
    )

    assert len(store) == 10
    assert sorted(set(store.specs())) == ["jpeg:40", "none"]


def test_augmentation_draws_a_spec_per_extra_view(dataset_root: Path, tmp_path: Path) -> None:
    store = make_store(tmp_path)

    extract(scan_split(dataset_root), backbone=CountingBackbone(), store=store, augment=3)

    assert len(store) == 5 * 3


def test_augmentation_never_repeats_a_spec_for_the_same_image(
    dataset_root: Path, tmp_path: Path
) -> None:
    store = make_store(tmp_path)

    extract(scan_split(dataset_root), backbone=CountingBackbone(), store=store, augment=4)

    per_image: dict[str, list[str]] = {}
    for path, spec in zip(store.paths(), store.specs(), strict=True):
        per_image.setdefault(path, []).append(spec)
    assert all(len(specs) == len(set(specs)) for specs in per_image.values())


def test_augmented_views_are_reproducible(dataset_root: Path, tmp_path: Path) -> None:
    samples = scan_split(dataset_root)

    first = make_store(tmp_path / "a")
    extract(samples, backbone=CountingBackbone(), store=first, augment=2, seed=5)
    second = make_store(tmp_path / "b")
    extract(samples, backbone=CountingBackbone(), store=second, augment=2, seed=5)

    assert first.specs() == second.specs()


def test_an_unreadable_image_is_counted_as_failed_not_fatal(
    dataset_root: Path, tmp_path: Path
) -> None:
    (dataset_root / "fake" / "sdxl" / "broken.png").write_bytes(b"not a png")

    stats = extract(scan_split(dataset_root), backbone=CountingBackbone(), store=make_store(tmp_path))

    assert stats.added == 5
    assert stats.failed == 1


def test_a_corrupt_image_can_be_made_fatal_on_request(dataset_root: Path, tmp_path: Path) -> None:
    (dataset_root / "fake" / "sdxl" / "broken.png").write_bytes(b"not a png")

    with pytest.raises(OSError):
        extract(
            scan_split(dataset_root),
            backbone=CountingBackbone(),
            store=make_store(tmp_path),
            skip_errors=False,
        )


def test_the_crop_mode_from_the_store_config_is_honoured(dataset_root: Path, tmp_path: Path) -> None:
    backbone = CountingBackbone()
    store = make_store(tmp_path, crop_mode="resize")

    extract(scan_split(dataset_root), backbone=backbone, store=store)

    # resize mode collapses to a single whole-image view regardless of crops_per_image
    assert backbone.calls == [1, 1, 1, 1, 1]


def test_a_backbone_whose_width_disagrees_with_the_cache_is_rejected(
    dataset_root: Path, tmp_path: Path
) -> None:
    store = make_store(tmp_path)
    backbone = CountingBackbone()
    backbone.dim = 99

    with pytest.raises(ValueError, match="width"):
        extract(scan_split(dataset_root), backbone=backbone, store=store)
