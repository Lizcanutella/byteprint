from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from byteprint.cache import ExtractConfig, EmbeddingStore, StaleCacheError, key_for
from tests.conftest import write_image


def config(**overrides) -> ExtractConfig:
    base = dict(backbone="stub", dim=4, crop_size=28, crops_per_image=2, crop_mode="texture", seed=0)
    return ExtractConfig(**{**base, **overrides})


def add_row(store: EmbeddingStore, name: str, value: float, *, label: int = 1) -> None:
    store.add(
        name,
        np.full((2, 4), value, dtype=np.float32),
        path=Path(f"/data/{name}.png"),
        label=label,
        generator="sdxl" if label else "real",
        spec="none",
    )


def test_a_saved_store_round_trips_its_vectors(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "a", 1.0)
    store.flush()

    reopened = EmbeddingStore.open(tmp_path / "cache", config())

    assert np.allclose(reopened.matrix(), np.full((1, 4), 1.0))


def test_crop_embeddings_are_mean_pooled_into_one_row_per_image(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    store.add(
        "a",
        np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
        path=Path("/data/a.png"),
        label=1,
        generator="sdxl",
        spec="none",
    )

    assert np.allclose(store.matrix(), np.full((1, 4), 1.0))


def test_labels_and_generators_stay_aligned_with_the_matrix(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "r", 0.0, label=0)
    add_row(store, "f", 1.0, label=1)

    assert store.labels().tolist() == [0, 1]
    assert store.generators() == ["real", "sdxl"]


def test_reopening_resumes_so_finished_work_is_skipped(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "a", 1.0)
    store.flush()

    reopened = EmbeddingStore.open(tmp_path / "cache", config())

    assert reopened.has("a")
    assert not reopened.has("b")


def test_appending_to_a_resumed_store_keeps_the_earlier_rows(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "a", 1.0)
    store.flush()

    reopened = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(reopened, "b", 2.0)
    reopened.flush()

    assert EmbeddingStore.open(tmp_path / "cache", config()).matrix().shape == (2, 4)


def test_a_changed_extraction_config_invalidates_the_cache(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "a", 1.0)
    store.flush()

    with pytest.raises(StaleCacheError, match="crop_size"):
        EmbeddingStore.open(tmp_path / "cache", config(crop_size=42))


def test_a_stale_cache_can_be_rebuilt_on_request(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "a", 1.0)
    store.flush()

    rebuilt = EmbeddingStore.open(tmp_path / "cache", config(crop_size=42), rebuild=True)

    assert rebuilt.matrix().shape == (0, 4)


def test_adding_a_vector_of_the_wrong_width_is_rejected(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())

    with pytest.raises(ValueError, match="width 4"):
        store.add(
            "a",
            np.zeros((2, 9), dtype=np.float32),
            path=Path("/data/a.png"),
            label=1,
            generator="sdxl",
            spec="none",
        )


def test_the_same_key_is_never_stored_twice(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "a", 1.0)

    with pytest.raises(ValueError, match="already"):
        add_row(store, "a", 5.0)


def test_the_cache_key_tracks_file_contents(tmp_path: Path) -> None:
    image = write_image(tmp_path / "x.png", seed=1)
    before = key_for(image, "none")

    write_image(tmp_path / "x.png", size=(96, 96), seed=2)

    assert key_for(image, "none") != before


def test_the_cache_key_distinguishes_laundering_specs(tmp_path: Path) -> None:
    image = write_image(tmp_path / "x.png", seed=1)

    assert key_for(image, "none") != key_for(image, "jpeg:40")


def test_an_empty_store_reports_an_empty_matrix_of_the_right_width(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())

    assert store.matrix().shape == (0, 4)
    assert store.labels().shape == (0,)
