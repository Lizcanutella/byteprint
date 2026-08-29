"""SID_Set (HuggingFace parquet shards) -> the split directory byteprint reads.

SID_Set ships as 283 parquet shards with an ``image`` column of encoded bytes
and a three-way ``label``. Two things have to happen before the engine can see
it, and both are decisions rather than plumbing:

**The three classes collapse to two.** Label 1 (fully synthetic) is
unambiguously AIGC. Label 2 (a real photograph with an AI-edited region) is the
open question -- it plausibly counts, but the choice changes what the probe
learns. So tampered images are written under ``fake/tampered/`` as their own
*generator*: they train as AIGC by default, and ``byteprint logo --held-out
tampered`` answers what a detector trained only on fully synthetic images does
when it meets one. The decision becomes a measurement instead of an assumption.

**Every class is re-encoded to one container.** SID_Set's reals come from
OpenImages as JPEG while its synthetics are PNG. Writing the original bytes
would hand a classifier a 99%-accurate shortcut that has nothing to do with
synthesis. Re-encoding all three classes to PNG equalises the container while
preserving the pixels exactly -- a JPEG's compression artifacts survive into the
PNG, which is real forensic evidence; its *file extension* is not.

Selection is two-pass: read only the ``label`` column across every shard
(columnar, so this is cheap), sample row indices per class, then read back only
the rows that were chosen. Reservoir-sampling the decoded images instead would
mean holding gigabytes of pixels in memory to throw most of them away.
"""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

#: SID_Set's ``label`` column, mapped onto byteprint's ``real/`` + ``fake/<gen>/``
#: layout. Label 2 keeps its own generator directory on purpose -- see above.
LABEL_DIRS: dict[int, Path] = {
    0: Path("real"),
    1: Path("fake") / "full_synthetic",
    2: Path("fake") / "tampered",
}


@dataclass(slots=True)
class MaterializeStats:
    """What a materialisation run actually wrote, and what it refused."""

    written: int = 0
    failed: int = 0
    by_label: dict[int, int] = field(default_factory=dict)
    #: ``(label, original container format) -> count``. Kept because an
    #: imbalance here is exactly the shortcut re-encoding exists to remove;
    #: reporting it lets a reviewer confirm the removal was needed.
    source_formats: dict[tuple[int, str], int] = field(default_factory=dict)

    def render(self) -> str:
        per_label = ", ".join(f"label {k}: {v}" for k, v in sorted(self.by_label.items()))
        formats = ", ".join(
            f"label {label}/{fmt}: {count}"
            for (label, fmt), count in sorted(self.source_formats.items())
        )
        return (
            f"wrote {self.written} images ({per_label or 'none'}), {self.failed} failed\n"
            f"  source containers: {formats or 'none'}"
        )


def quota_for(per_class: int | Mapping[int, int]) -> dict[int, int]:
    """Normalise a quota into one count per label.

    A bare integer means the same count for each of the three classes, which
    collapses to *twice* as many fakes as reals once labels 1 and 2 are both
    AIGC. A per-label mapping (``{0: 8000, 1: 4000, 2: 4000}``) restores the
    binary balance -- and the real class is the one that sets how finely a
    threshold at 1% FPR can be placed, so it is the one worth spending on.
    """
    quota = (
        {label: int(per_class) for label in LABEL_DIRS}
        if isinstance(per_class, int)
        else {int(k): int(v) for k, v in per_class.items()}
    )
    unknown = set(quota) - set(LABEL_DIRS)
    if unknown:
        raise ValueError(f"quota names labels outside SID_Set: {sorted(unknown)}")
    if any(count <= 0 for count in quota.values()):
        raise ValueError(f"every quota must be positive, got {quota}")
    return quota


def select_rows(
    labels_by_shard: Mapping[str, Sequence[int]],
    per_class: int | Mapping[int, int],
    seed: int = 0,
) -> dict[str, list[int]]:
    """Choose row indices for each SID_Set label, up to that label's quota.

    Sampling is uniform over the whole corpus rather than shard-by-shard: the
    shards are written in dataset order, so taking the first N rows would bias
    the sample towards whatever the corpus happens to begin with. Returned row
    indices are sorted per shard so the second pass is a sequential read.
    """
    quota = quota_for(per_class)

    candidates: dict[int, list[tuple[str, int]]] = {label: [] for label in quota}
    for shard in sorted(labels_by_shard):
        for row, label in enumerate(labels_by_shard[shard]):
            if label in candidates:  # anything outside 0/1/2 is not ours to file
                candidates[label].append((shard, row))

    rng = np.random.default_rng(seed)
    chosen: dict[str, list[int]] = {}
    for label in sorted(candidates):
        pool = candidates[label]
        if not pool:
            continue
        take = min(quota[label], len(pool))
        for index in rng.permutation(len(pool))[:take]:
            shard, row = pool[int(index)]
            chosen.setdefault(shard, []).append(row)

    return {shard: sorted(rows) for shard, rows in sorted(chosen.items())}


def encode_png(payload: bytes) -> bytes:
    """Decode an image of any container and re-encode it as RGB PNG, losslessly."""
    with Image.open(io.BytesIO(payload)) as handle:
        rgb = handle.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
    return buffer.getvalue()


def _source_format(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as handle:
        return handle.format or "unknown"


def write_image_tree(
    records: Iterable[tuple[int, str, bytes]], out_root: Path | str
) -> MaterializeStats:
    """Write ``(label, image id, encoded bytes)`` records as a byteprint split.

    An unreadable record is counted and skipped -- a 210k-image corpus will
    contain a few, and dying on one after an hour of decoding is not a useful
    failure. An *unknown label* is different: it means the mapping above is
    wrong, and filing those images anywhere would silently corrupt the labels.
    """
    out_root = Path(out_root)
    by_label: Counter[int] = Counter()
    formats: Counter[tuple[int, str]] = Counter()
    written = failed = 0

    for label, image_id, payload in records:
        if label not in LABEL_DIRS:
            raise ValueError(
                f"label {label} is not one of SID_Set's classes {sorted(LABEL_DIRS)}"
            )

        try:
            source = _source_format(payload)
            encoded = encode_png(payload)
        except (UnidentifiedImageError, OSError, ValueError):
            failed += 1
            continue

        destination = out_root / LABEL_DIRS[label] / f"{image_id}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)

        written += 1
        by_label[label] += 1
        formats[(label, source)] += 1

    return MaterializeStats(
        written=written,
        failed=failed,
        by_label=dict(by_label),
        source_formats=dict(formats),
    )


# -- the parquet pass ------------------------------------------------------
#
# pyarrow is only needed to *build* a dataset, never to train or score on one,
# so it stays an optional dependency imported at call time.


def read_labels(shard_paths: Sequence[Path]) -> dict[str, list[int]]:
    """Read only the ``label`` column from every shard. Columnar, so this is cheap."""
    import pyarrow.parquet as pq

    labels: dict[str, list[int]] = {}
    for path in shard_paths:
        table = pq.read_table(path, columns=["label"])
        labels[str(path)] = table.column("label").to_pylist()
    return labels


def read_selected(
    shard_paths: Sequence[Path], selection: Mapping[str, Sequence[int]]
) -> Iterable[tuple[int, str, bytes]]:
    """Yield ``(label, image id, bytes)`` for the selected rows, shard by shard."""
    import pyarrow.parquet as pq

    for path in shard_paths:
        rows = selection.get(str(path))
        if not rows:
            continue
        table = pq.read_table(path, columns=["img_id", "image", "label"]).take(list(rows))
        for image_id, image, label in zip(
            table.column("img_id").to_pylist(),
            table.column("image").to_pylist(),
            table.column("label").to_pylist(),
        ):
            # HuggingFace stores an image column as {"bytes": ..., "path": ...}.
            payload = image["bytes"] if isinstance(image, dict) else image
            yield int(label), str(image_id), payload


def materialize(
    shard_paths: Sequence[Path],
    out_root: Path | str,
    *,
    per_class: int | Mapping[int, int],
    seed: int = 0,
) -> MaterializeStats:
    """Two-pass materialisation of a balanced subset into ``out_root``."""
    labels = read_labels(shard_paths)
    selection = select_rows(labels, per_class=per_class, seed=seed)
    return write_image_tree(read_selected(shard_paths, selection), out_root)
