#!/usr/bin/env python3
"""Score a directory of images, writing the competition's JSON output.

The deliverable interface from section 5.5 of the brief, as a standalone script:

    python scripts/score_directory.py IMAGE_DIR --probe runs/probe.joblib \
                                      --out predictions.json

Writes a JSON list of ``{"image_path": ..., "pred": ...}``, where ``pred`` is
the likelihood the image is AI-generated. Identical to ``byteprint score``,
which this simply forwards to -- there is one implementation, not two.
"""

from __future__ import annotations

import sys

from byteprint.cli import main

if __name__ == "__main__":
    sys.exit(main(["score", *sys.argv[1:]]))
