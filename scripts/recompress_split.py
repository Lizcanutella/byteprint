#!/usr/bin/env python
"""Re-encode a split through one encoder, so compression history cannot classify.

    python scripts/recompress_split.py data/sid_train data/sid_train_jpeg95 \
        --encoding jpeg:95 --workers 8

The JPEG-95 control for the SID_Set run: SID_Set's reals are 100% JPEG-family
and its fully-synthetic images 100% PNG, so a probe can score well on
compression history alone. Putting both classes through the same encoder
removes that shortcut. See ``byteprint/recompress.py`` for what the control
does and does not establish.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from byteprint.recompress import DEFAULT_ENCODING, recompress_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="split directory to read (real/ + fake/)")
    parser.add_argument("dst", type=Path, help="split directory to write")
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help=f"'png' or 'jpeg:<quality>' (default {DEFAULT_ENCODING})",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="threads encoding in parallel"
    )
    args = parser.parse_args(argv)

    print(f"{args.src} -> {args.dst} as {args.encoding}", flush=True)
    stats = recompress_split(
        args.src, args.dst, encoding=args.encoding, workers=args.workers
    )
    print(stats.render(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
