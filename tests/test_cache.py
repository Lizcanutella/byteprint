from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from byteprint.cache import (
    SCHEMA_VERSION,
    EmbeddingStore,
    ExtractConfig,
    StaleCacheError,
    key_for,
    read_config,
)
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


# -- schema 2: the crop rows survive into the cache ------------------------


def write_schema_one_cache(root: Path) -> Path:
    """A cache as the previous format wrote it: one pooled row, no schema key."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "backbone": "stub", "dim": 4, "crop_size": 28,
                "crops_per_image": 2, "crop_mode": "texture", "seed": 0,
            }
        )
    )
    np.save(root / "features.npy", np.full((1, 4), 1.0, dtype=np.float32))
    (root / "records.jsonl").write_text(
        json.dumps(
            {"key": "a", "path": "/data/a.png", "label": 1,
             "generator": "sdxl", "spec": "none"}
        )
        + "\n"
    )
    return root


def test_the_individual_crop_rows_survive_rather_than_only_their_mean(tmp_path: Path) -> None:
    # The whole point of schema 2: pooling is decided by whoever reads the
    # cache, so the reader has to be left something to decide between.
    store = EmbeddingStore.open(tmp_path / "cache", config())
    store.add(
        "a",
        np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
        path=Path("/data/a.png"),
        label=1,
        generator="sdxl",
        spec="none",
    )

    assert store.crop_matrix().tolist() == [[0.0] * 4, [2.0] * 4]
    assert store.crop_counts().tolist() == [2]


def test_crop_rows_round_trip_through_disk(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    store.add(
        "a",
        np.array([[1.0, 1.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]], dtype=np.float32),
        path=Path("/data/a.png"),
        label=1,
        generator="sdxl",
        spec="none",
    )
    store.flush()

    reopened = EmbeddingStore.open(tmp_path / "cache", config())

    assert np.allclose(reopened.crop_matrix(), store.crop_matrix())
    assert reopened.crop_counts().tolist() == [2]
    assert np.allclose(reopened.matrix(), np.full((1, 4), 3.0))


def test_bags_of_different_sizes_stay_separable(tmp_path: Path) -> None:
    # `center` and `resize` return one crop whatever --crops says, so a cache
    # holding bags of one size is an assumption, not a guarantee.
    store = EmbeddingStore.open(tmp_path / "cache", config())
    store.add(
        "one",
        np.array([[4.0, 4.0, 4.0, 4.0]], dtype=np.float32),
        path=Path("/data/one.png"), label=1, generator="sdxl", spec="none",
    )
    add_row(store, "two", 1.0)
    store.flush()

    reopened = EmbeddingStore.open(tmp_path / "cache", config())

    assert reopened.crop_counts().tolist() == [1, 2]
    assert reopened.crop_matrix().shape == (3, 4)
    assert np.allclose(reopened.matrix(), np.array([[4.0] * 4, [1.0] * 4]))


def test_a_schema_one_cache_is_refused_with_the_fix_in_the_message(tmp_path: Path) -> None:
    root = write_schema_one_cache(tmp_path / "cache")

    with pytest.raises(StaleCacheError, match="schema 1"):
        EmbeddingStore.open(root, config())


def test_a_schema_one_cache_can_be_discarded_on_request(tmp_path: Path) -> None:
    root = write_schema_one_cache(tmp_path / "cache")

    rebuilt = EmbeddingStore.open(root, config(), rebuild=True)

    assert len(rebuilt) == 0
    assert rebuilt.crop_matrix().shape == (0, 4)


def test_the_written_config_records_its_schema(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "a", 1.0)
    store.flush()

    written = json.loads((tmp_path / "cache" / "config.json").read_text())

    assert written["schema"] == SCHEMA_VERSION


def test_reading_a_config_recovers_the_extraction_settings_without_the_schema(
    tmp_path: Path,
) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())
    add_row(store, "a", 1.0)
    store.flush()

    assert read_config(tmp_path / "cache") == config()


def test_an_empty_store_reports_empty_crop_rows(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "cache", config())

    assert store.crop_matrix().shape == (0, 4)
    assert store.crop_counts().tolist() == []
