"""The depth analysis: cache reading, the split it inherits, and the table.

An hour of GPU time produces a cache that is useless if the script meant to read
it disagrees about the column layout or the calibration holdout. All of that is
checkable on a synthetic cache in milliseconds, so it is checked here rather
than discovered afterwards.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from byteprint.cli import _split_indices
from byteprint_depth import N_BLOCKS, block_slice

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_depth.py"
WIDTH = 4


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_depth", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = load_module()


def write_cache(root: Path, n: int = 40, *, width: int = WIDTH, rows: int | None = None) -> Path:
    """A cache in the on-disk layout `EmbeddingStore.flush` writes.

    Block 0 is pure signal -- it separates the classes exactly -- and every other
    block is noise. A reader that mixes the blocks up cannot produce a perfect
    AUC on block 0 and chance elsewhere, so the test discriminates.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    labels = np.array([i % 2 for i in range(n)], dtype=np.int64)

    features = rng.normal(size=(n, N_BLOCKS * width)).astype(np.float32)
    features[:, block_slice(0, width=width)] = labels[:, None] * 20.0

    np.save(root / "features.npy", features[:rows] if rows else features)
    (root / "config.json").write_text(
        json.dumps({"backbone": "siglip2_depth_hf", "dim": N_BLOCKS * width})
    )
    with (root / "records.jsonl").open("w") as handle:
        for i, label in enumerate(labels):
            handle.write(
                json.dumps(
                    {
                        "key": f"k{i}",
                        "path": f"/img/{i}.png",
                        "label": int(label),
                        "generator": "real" if label == 0 else "full_synthetic",
                        "spec": "none" if i % 4 < 2 else "jpeg:70",
                    }
                )
                + "\n"
            )
    return root


class Args:
    width = WIDTH
    head = "logreg"
    C = 1.0
    calib_fraction = 0.2
    target_fpr = 0.01
    seed = 0


# -- the split is inherited, not reinvented -------------------------------


@pytest.mark.parametrize("n", [10, 40, 1000, 48000])
def test_the_split_matches_the_one_byteprint_train_uses(n: int) -> None:
    # A different calibration holdout makes the pooler row incomparable to the
    # published number, which is the only reason that row exists.
    mine = analyze.split_indices(n, 0.2, 0)
    theirs = _split_indices(n, 0.2, 0)
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


# -- reading the cache ----------------------------------------------------


def test_a_cache_is_read_without_materialising_the_whole_matrix(tmp_path: Path) -> None:
    cache = analyze.Cache(write_cache(tmp_path / "c"))
    assert isinstance(cache.features, np.memmap)
    assert len(cache.labels) == 40
    assert set(cache.generators.tolist()) == {"real", "full_synthetic"}


def test_a_truncated_cache_is_refused_rather_than_silently_misaligned(tmp_path: Path) -> None:
    # A job killed between writing features and writing records would otherwise
    # zip labels onto the wrong rows.
    with pytest.raises(ValueError, match="truncated"):
        analyze.Cache(write_cache(tmp_path / "c", rows=30))


def test_a_block_comes_back_with_its_own_columns(tmp_path: Path) -> None:
    cache = analyze.Cache(write_cache(tmp_path / "c"))
    block = cache.block(0, WIDTH)
    assert block.shape == (40, WIDTH)
    assert np.allclose(block[cache.labels == 1], 20.0)
    assert np.allclose(block[cache.labels == 0], 0.0)


def test_concatenating_blocks_preserves_the_order_asked_for(tmp_path: Path) -> None:
    cache = analyze.Cache(write_cache(tmp_path / "c"))
    both = cache.blocks([2, 0], WIDTH)
    assert both.shape == (40, 2 * WIDTH)
    assert np.array_equal(both[:, WIDTH:], cache.block(0, WIDTH))


# -- fitting --------------------------------------------------------------


def test_the_informative_block_separates_and_a_noise_block_does_not(tmp_path: Path) -> None:
    cache = analyze.Cache(write_cache(tmp_path / "c"))

    signal = analyze.fit_and_score(cache, cache, [0], Args())
    noise = analyze.fit_and_score(cache, cache, [3], Args())

    assert signal["auc"] == pytest.approx(1.0)
    assert signal["auc"] > noise["auc"]


def test_every_rung_present_in_the_cache_is_scored(tmp_path: Path) -> None:
    cache = analyze.Cache(write_cache(tmp_path / "c"))
    row = analyze.fit_and_score(cache, cache, [0], Args())
    assert set(row["per_rung"]) == {"none", "jpeg:70"}
    assert row["worst_rung"] in {"none", "jpeg:70"}


def test_the_per_generator_split_names_the_fake_class(tmp_path: Path) -> None:
    cache = analyze.Cache(write_cache(tmp_path / "c"))
    row = analyze.fit_and_score(cache, cache, [0], Args())
    assert "full_synthetic" in row["per_generator"]


# -- naming and rendering -------------------------------------------------


def test_the_last_block_is_named_for_the_pooler_not_a_layer() -> None:
    names = analyze.block_names(27)
    assert len(names) == N_BLOCKS
    assert names[-1] == "pooler"
    assert names[0] == "layer 1"


def test_the_table_has_a_row_per_tap_and_a_column_per_generator(tmp_path: Path) -> None:
    cache = analyze.Cache(write_cache(tmp_path / "c"))
    rows = {name: analyze.fit_and_score(cache, cache, [i], Args()) for i, name in
            enumerate(["layer 1", "layer 3"])}
    table = analyze.render(rows, ["full_synthetic"])
    assert table.count("\n") == 3  # header, rule, two taps
    assert "full_synthetic" in table.splitlines()[0]


def test_the_per_rung_table_lists_every_rung_once(tmp_path: Path) -> None:
    cache = analyze.Cache(write_cache(tmp_path / "c"))
    rows = {"layer 1": analyze.fit_and_score(cache, cache, [0], Args())}
    header = analyze.render_per_rung(rows).splitlines()[0]
    assert header.count("jpeg:70") == 1 and header.count("none") == 1


# -- schema 2: one row per crop -------------------------------------------
#
# The crop-pooling work moved the crop reduction out of the cache, so an
# extraction now writes one row per *crop* and `n_crops` per record, while this
# script was written against schema 1's one-row-per-image. Reading a schema-2
# cache as though it were schema 1 would misalign every label against its
# features, so the reader has to pool the crops itself -- and pooling them on
# read is what makes crop count a knob this analysis can sweep for free.


def write_crop_cache(
    root: Path, n: int = 40, *, crops: int = 4, width: int = WIDTH
) -> Path:
    """A schema-2 cache: ``crops`` rows per image, and ``n_crops`` per record.

    Block 0 separates the classes exactly *in the mean over an image's crops*,
    while each individual crop row is that mean plus a large zero-sum wobble.
    A reader that forgets to pool, or that pools the wrong rows together,
    cannot recover the clean separation.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    labels = np.array([i % 2 for i in range(n)], dtype=np.int64)

    features = rng.normal(size=(n * crops, N_BLOCKS * width)).astype(np.float32)
    signal = np.repeat(labels * 20.0, crops)[:, None]
    wobble = np.tile(np.arange(crops) - (crops - 1) / 2.0, n)[:, None] * 100.0
    features[:, block_slice(0, width=width)] = signal + wobble

    np.save(root / "features.npy", features)
    (root / "config.json").write_text(
        json.dumps({"backbone": "siglip2_depth_hf", "dim": N_BLOCKS * width, "schema": 2})
    )
    with (root / "records.jsonl").open("w") as handle:
        for i, label in enumerate(labels):
            handle.write(
                json.dumps(
                    {
                        "key": f"k{i}",
                        "path": f"/img/{i}.png",
                        "label": int(label),
                        "generator": "real" if label == 0 else "full_synthetic",
                        "spec": "none" if i % 4 < 2 else "jpeg:70",
                        "n_crops": crops,
                    }
                )
                + "\n"
            )
    return root


def test_a_per_crop_cache_yields_one_pooled_row_per_image(tmp_path: Path) -> None:
    cache = analyze.Cache(write_crop_cache(tmp_path / "c", n=8, crops=4))
    assert len(cache.labels) == 8
    assert cache.block(0, WIDTH).shape == (8, WIDTH)


def test_the_pooled_row_is_the_mean_of_that_images_crops(tmp_path: Path) -> None:
    root = write_crop_cache(tmp_path / "c", n=6, crops=4)
    raw = np.load(root / "features.npy")
    block = analyze.Cache(root).block(1, WIDTH)
    for i in range(6):
        expected = raw[i * 4 : (i + 1) * 4, block_slice(1, width=WIDTH)].mean(axis=0)
        assert np.allclose(block[i], expected, atol=1e-5)


def test_a_crop_limit_keeps_each_images_first_crops(tmp_path: Path) -> None:
    """``texture`` is prefix-stable, so the first two of eight crops are the two
    a ``--crops 2`` extraction would have chosen. That is what lets one
    extraction answer the crop-count question, and it needs the *first* rows."""
    root = write_crop_cache(tmp_path / "c", n=6, crops=4)
    raw = np.load(root / "features.npy")
    block = analyze.Cache(root, crop_limit=2).block(1, WIDTH)
    for i in range(6):
        expected = raw[i * 4 : i * 4 + 2, block_slice(1, width=WIDTH)].mean(axis=0)
        assert np.allclose(block[i], expected, atol=1e-5)


def test_a_crop_limit_beyond_the_bag_keeps_the_whole_bag(tmp_path: Path) -> None:
    root = write_crop_cache(tmp_path / "c", n=6, crops=4)
    assert np.allclose(
        analyze.Cache(root, crop_limit=99).block(1, WIDTH),
        analyze.Cache(root).block(1, WIDTH),
    )


def test_pooling_crops_commutes_with_slicing_columns(tmp_path: Path) -> None:
    """The identity every row of the depth table rests on.

    A block read back after pooling must equal what a single-tap extraction
    would have cached, which holds only because a mean over rows and a slice
    over columns commute. Pinned here for schema 2 as it was for schema 1.
    """
    root = write_crop_cache(tmp_path / "c", n=6, crops=4)
    raw = np.load(root / "features.npy")
    pooled_then_sliced = analyze.Cache(root).block(2, WIDTH)
    sliced = raw[:, block_slice(2, width=WIDTH)]
    sliced_then_pooled = np.stack(
        [sliced[i * 4 : (i + 1) * 4].mean(axis=0) for i in range(6)]
    )
    assert np.allclose(pooled_then_sliced, sliced_then_pooled, atol=1e-6)


def test_a_crop_cache_whose_counts_do_not_cover_its_rows_is_refused(tmp_path: Path) -> None:
    root = write_crop_cache(tmp_path / "c", n=6, crops=4)
    features = np.load(root / "features.npy")
    np.save(root / "features.npy", features[:-3])
    with pytest.raises(ValueError, match="n_crops"):
        analyze.Cache(root)


def test_the_signal_block_still_separates_after_pooling(tmp_path: Path) -> None:
    """End to end: the wobble is larger than the signal on any single crop, so
    the separation exists only in the pooled rows."""
    cache = analyze.Cache(write_crop_cache(tmp_path / "c", n=40, crops=4))
    assert analyze.fit_and_score(cache, cache, [0], Args())["auc"] == 1.0
