#!/usr/bin/env python
"""Build a byteprint split directory from staged SID_Set parquet shards.

    python scripts/materialize_sid_set.py SID_SET_DIR OUT_DIR \
        --split train --per-class 4000 --seed 0

Writes ``OUT_DIR/{real,fake/full_synthetic,fake/tampered}/`` as PNG. See
``byteprint/sid_set.py`` for why the container is normalised and why tampered
images keep their own generator directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from byteprint.sid_set import materialize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="staged SID_Set snapshot (contains data/)")
    parser.add_argument("out", type=Path, help="split directory to write")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shards", type=int, default=0, help="use only the first N shards (0 = all)"
    )
    args = parser.parse_args(argv)

    shard_dir = args.root / "data"
    shards = sorted(shard_dir.glob(f"{args.split}-*.parquet"))
    if not shards:
        parser.error(f"no {args.split} shards under {shard_dir}")
    if args.shards:
        shards = shards[: args.shards]

    print(f"{len(shards)} shards -> {args.out}", flush=True)
    stats = materialize(shards, args.out, per_class=args.per_class, seed=args.seed)
    print(stats.render(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
