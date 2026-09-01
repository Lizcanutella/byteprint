#!/usr/bin/env python
"""Fit one probe per tapped depth and print the accuracy side of the frontier.

The expensive part already happened: `byteprint extract --backbone
siglip2_depth_hf` wrote a cache whose every row holds eleven depths plus the
tower's pooled output, all from one forward pass. Reading a single depth back
out is a column slice, so the whole depth curve costs a few minutes of CPU and
no GPU at all.

    python scripts/analyze_depth.py \
        --train-cache runs/cache/sid_train_siglip2_depth_hf \
        --ladder-cache runs/cache/sid_ladder_siglip2_depth_hf \
        --out runs/depth_frontier.md

The fit/calibrate split, the head and the target FPR mirror `byteprint train`
exactly -- same helper, same seed, same 20% holdout -- so the pooler row is
directly comparable to the published SigLIP2 numbers rather than merely similar
to them. That row is the control: it should reproduce AUC 0.9497 and TPR@1%FPR
0.5854. If it does not, the plumbing is wrong and no other row means anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from byteprint.metrics import evaluate
from byteprint.pooling import resolve_pooling, segment_reduce, truncate_bags
from byteprint.probe import LinearProbe, ProbeConfig
from byteprint_depth import N_BLOCKS, SIGLIP2_SO400M_WIDTH, block_slice, tap_layers

# The reduction the published runs used, taken from the registry rather than
# spelled `values.mean(axis=0)` again, so "mean" means one thing in this project.
MEAN = resolve_pooling("mean").reduce

# so400m's depth. Only used to name the rows; the blocks themselves are whatever
# the extraction wrote, and --num-layers overrides it.
SO400M_LAYERS = 27


class Cache:
    """A read-only view of an embedding cache that never holds the whole matrix.

    `EmbeddingStore.open` materialises the features twice -- once as a list of
    rows, once re-stacked -- which for a twelve-block cache is several gigabytes
    of nothing useful. Every access here is one column block at a time, off a
    memory map.

    Both cache schemas are read. Schema 1 stored one mean-pooled row per image;
    schema 2 stores one row per *crop* plus ``n_crops`` per record, because the
    crop-pooling work moved that reduction out of the cache. Pooling on read is
    not merely compatibility: with ``crop_limit`` it makes crop count a knob
    this analysis can sweep, so one 8-crop extraction answers the depth
    question at every crop count instead of one.
    """

    def __init__(self, root: Path, crop_limit: int | None = None) -> None:
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text())
        self.records = [
            json.loads(line)
            for line in (self.root / "records.jsonl").read_text().splitlines()
            if line
        ]
        self.features = np.load(self.root / "features.npy", mmap_mode="r")
        self.crop_limit = crop_limit

        # Schema 1 rows are already per image, which is `n_crops` of 1 as far as
        # the pooling below is concerned -- so there is one code path, not two.
        self.counts = np.asarray(
            [r.get("n_crops", 1) for r in self.records], dtype=np.int64
        )
        if int(self.counts.sum()) != len(self.features):
            raise ValueError(
                f"{self.root} holds {len(self.features)} rows but its records' "
                f"n_crops sum to {int(self.counts.sum())} over {len(self.records)} "
                "images; the cache is truncated"
            )

        self.labels = np.asarray([r["label"] for r in self.records], dtype=np.int64)
        self.generators = np.asarray([r["generator"] for r in self.records])
        self.specs = np.asarray([r["spec"] for r in self.records])

    def block(self, position: int, width: int) -> np.ndarray:
        """One block's columns, mean-pooled over each image's crops.

        Slicing the columns first and pooling after is the cheap order -- it
        touches one twelfth of the bytes -- and it is only legitimate because a
        mean over rows commutes with a slice over columns. That identity is what
        makes the block read back here exactly what a single-tap extraction
        would have cached, and `test_analyze_depth.py` pins it.
        """
        columns = np.ascontiguousarray(self.features[:, block_slice(position, width=width)])
        if self.crop_limit is None and bool((self.counts == 1).all()):
            return columns
        values, counts = truncate_bags(columns, self.counts, self.crop_limit)
        return segment_reduce(values, counts, MEAN).astype(np.float32)

    def blocks(self, positions: list[int], width: int) -> np.ndarray:
        return np.concatenate([self.block(p, width) for p in positions], axis=1)


def split_indices(n: int, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Byte-identical to `byteprint.cli._split_indices`.

    Copied rather than imported because it is private, and a comparison against
    the published numbers is worthless if the calibration holdout differs.
    """
    order = np.random.default_rng(seed).permutation(n)
    cut = max(1, int(round(n * holdout)))
    return order[cut:], order[:cut]


def block_names(num_layers: int) -> list[str]:
    return [f"layer {layer}" for layer in tap_layers(num_layers)] + ["pooler"]


def fit_and_score(train: "Cache", ladder: "Cache", positions: list[int], args) -> dict:
    """Train on the train cache's blocks, evaluate over the whole ladder."""
    features = train.blocks(positions, args.width)
    fit_idx, calib_idx = split_indices(len(train.labels), args.calib_fraction, args.seed)

    probe = LinearProbe(ProbeConfig(head=args.head, C=args.C, seed=args.seed))
    probe.fit(features[fit_idx], train.labels[fit_idx])
    probe.calibrate(
        features[calib_idx], train.labels[calib_idx], target_fpr=args.target_fpr
    )
    del features

    scores = probe.score(ladder.blocks(positions, args.width))
    report = evaluate(
        ladder.labels,
        scores,
        generators=ladder.generators.tolist(),
        fpr_targets=(0.01, 0.001),
    )

    per_rung = {}
    for spec in sorted(set(ladder.specs.tolist())):
        mask = ladder.specs == spec
        if len(np.unique(ladder.labels[mask])) < 2:
            continue
        per_rung[spec] = float(evaluate(ladder.labels[mask], scores[mask]).auc)

    return {
        "auc": report.auc,
        "tpr01": report.tpr_at_fpr[0.01],
        "tpr001": report.tpr_at_fpr[0.001],
        "per_generator": {n: s.auc for n, s in report.per_generator.items()},
        "per_rung": per_rung,
        "worst_rung": min(per_rung, key=per_rung.get) if per_rung else "",
        "worst_auc": min(per_rung.values()) if per_rung else float("nan"),
    }


def leave_one_generator_out(train: "Cache", positions: list[int], args) -> float:
    """Mean AUC on a manipulation type held out of training.

    The project's weakest number and the one the backbone sweep moved most, so
    it is worth having per tap rather than only for the winner. Mirrors
    `byteprint logo`: hold out one fake generator entirely, train on the rest,
    score the held-out fakes against all the reals.
    """
    features = train.blocks(positions, args.width)
    held_out = sorted(set(train.generators[train.labels == 1].tolist()))

    aucs = []
    for name in held_out:
        fit_mask = train.generators != name
        test_mask = (train.generators == name) | (train.labels == 0)
        if len(np.unique(train.labels[fit_mask])) < 2:
            continue
        probe = LinearProbe(ProbeConfig(head=args.head, C=args.C, seed=args.seed))
        probe.fit(features[fit_mask], train.labels[fit_mask])
        aucs.append(
            evaluate(train.labels[test_mask], probe.score(features[test_mask])).auc
        )
    del features
    return float(np.mean(aucs)) if aucs else float("nan")


def render(rows: dict[str, dict], generators: list[str]) -> str:
    header = "| tap | AUC | TPR@1%FPR | TPR@0.1%FPR | "
    header += " | ".join(generators) + " | worst rung |"
    if any("logo" in row for row in rows.values()):
        header += " LOGO |"
    lines = [header, "|---|---|---|---|" + "---|" * (len(generators) + 1)]
    if any("logo" in row for row in rows.values()):
        lines[1] += "---|"
    for name, row in rows.items():
        cells = [name, f"{row['auc']:.4f}", f"{row['tpr01']:.4f}", f"{row['tpr001']:.4f}"]
        cells += [f"{row['per_generator'].get(g, float('nan')):.4f}" for g in generators]
        cells.append(f"{row['worst_rung']} {row['worst_auc']:.4f}")
        if "logo" in row:
            cells.append(f"{row['logo']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_per_rung(rows: dict[str, dict]) -> str:
    rungs = sorted({spec for row in rows.values() for spec in row["per_rung"]})
    lines = ["| tap | " + " | ".join(rungs) + " |", "|---|" + "---|" * len(rungs)]
    for name, row in rows.items():
        cells = [f"{row['per_rung'].get(r, float('nan')):.4f}" for r in rungs]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--ladder-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="write the tables here too")
    parser.add_argument("--width", type=int, default=SIGLIP2_SO400M_WIDTH)
    parser.add_argument("--num-layers", type=int, default=SO400M_LAYERS)
    parser.add_argument("--head", default="logreg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--blocks", default="", help="comma-separated block positions; default all"
    )
    parser.add_argument(
        "--crop-limit", type=int, default=None,
        help="pool only each image's first N crops. `texture` is prefix-stable, "
             "so N of an 8-crop cache is what a `--crops N` extraction would "
             "have cached -- this is the crop-count axis, for free (default: all)",
    )
    parser.add_argument(
        "--logo",
        action="store_true",
        help="also run leave-one-generator-out per tap (two extra fits each)",
    )
    args = parser.parse_args()

    train = Cache(args.train_cache, crop_limit=args.crop_limit)
    ladder = Cache(args.ladder_cache, crop_limit=args.crop_limit)
    expected = N_BLOCKS * args.width
    if train.features.shape[1] != expected:
        raise SystemExit(
            f"expected {expected} columns ({N_BLOCKS} blocks x {args.width}), got "
            f"{train.features.shape[1]} -- was this cache built by siglip2_depth_hf?"
        )

    names = block_names(args.num_layers)
    positions = (
        [int(p) for p in args.blocks.split(",")] if args.blocks else list(range(N_BLOCKS))
    )
    generators = sorted({r["generator"] for r in ladder.records if r["label"] != 0})

    limit = "all" if args.crop_limit is None else args.crop_limit
    print(f"train {len(train.labels)} images, ladder {len(ladder.labels)} images")
    print(f"crops pooled per image: {limit} (cache holds up to {int(train.counts.max())})")
    print(f"{len(positions)} taps: {', '.join(names[p] for p in positions)}\n")

    rows: dict[str, dict] = {}
    for position in positions:
        row = fit_and_score(train, ladder, [position], args)
        if args.logo:
            row["logo"] = leave_one_generator_out(train, [position], args)
        rows[names[position]] = row
        print(
            f"  {names[position]:<10} AUC {row['auc']:.4f}  TPR@1% {row['tpr01']:.4f}  "
            f"worst {row['worst_rung']} {row['worst_auc']:.4f}"
            + (f"  LOGO {row['logo']:.4f}" if args.logo else ""),
            flush=True,
        )

    # The single best mean-pooled tap, concatenated with the tower's own pooled
    # output. If depth and pooling carry different information this beats both;
    # if it merely matches the better one, they do not.
    single = {n: r for n, r in rows.items() if n != "pooler"}
    if single and "pooler" in rows:
        best = max(single, key=lambda n: single[n]["auc"])
        pair = [names.index(best), N_BLOCKS - 1]
        combo = fit_and_score(train, ladder, pair, args)
        if args.logo:
            combo["logo"] = leave_one_generator_out(train, pair, args)
        rows[f"{best} + pooler"] = combo
        print(f"\n  {best} + pooler   AUC {combo['auc']:.4f}  TPR@1% {combo['tpr01']:.4f}")

    body = (
        "## The depth frontier — accuracy by tap\n\n"
        + render(rows, generators)
        + "\n\n## Per rung of the official §5.2 ladder\n\n"
        + render_per_rung(rows)
        + "\n"
    )
    print("\n" + body)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body)
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
