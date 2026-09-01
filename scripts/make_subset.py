#!/usr/bin/env python
"""Build a symlinked subset of a materialised split, without touching the split.

A run that has to fit a fixed wall-clock budget can afford fewer *images* far
more comfortably than fewer laundered views per image, so this takes a stride
through each label directory rather than reducing ``--augment``.

Symlinks, not copies, for two reasons. The split directories are guarded by a
``.materialised`` marker because cache keys include mtime, so rewriting them
would silently invalidate every cache on disk; a symlink tree writes no image
file and changes no mtime. And 8,000 PNGs is real disk that buys nothing.

    python scripts/make_subset.py data/sid_train data/sid_train_half --stride 2

Selection is a stride through the sorted filenames, which is deterministic and
reproducible from the arguments alone. SID_Set names files by content hash, so
sorted order carries no class, generator or difficulty structure and a stride is
an unbiased sample of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def label_dirs(root: Path) -> list[Path]:
    """Every directory that directly holds images, relative to ``root``."""
    found = {
        path.parent.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    return sorted(found)


def build(source: Path, dest: Path, stride: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel in label_dirs(source):
        names = sorted(
            p.name
            for p in (source / rel).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        kept = names[::stride]
        out = dest / rel
        out.mkdir(parents=True, exist_ok=True)
        for name in kept:
            link = out / name
            if not link.is_symlink() and not link.exists():
                link.symlink_to((source / rel / name).resolve())
        counts[str(rel)] = len(kept)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--stride", type=int, default=2, help="keep every Nth image")
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit(f"stride must be at least 1, got {args.stride}")
    if not args.source.is_dir():
        raise SystemExit(f"no such split directory: {args.source}")

    counts = build(args.source, args.dest, args.stride)
    if not counts:
        raise SystemExit(f"{args.source} holds no images")
    for rel, n in counts.items():
        print(f"  {rel}: {n}")
    print(f"total {sum(counts.values())} symlinks under {args.dest}")

    # The extraction path itself does not read this marker -- the job scripts do,
    # to decide whether to materialise. Writing it here keeps a subset from ever
    # being mistaken for an unbuilt split and re-materialised over.
    (args.dest / ".materialised").touch()


if __name__ == "__main__":
    main()
