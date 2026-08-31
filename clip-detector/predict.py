"""
Required hackathon deliverable script: run the trained detector on every
image in a directory and write predictions as JSON.

Usage:
    python predict.py --input_dir DIR --output out.json

Output: a JSON list of {"image_path": <str>, "pred": <float in [0,1]>}
per image, where `pred` is the probability the image is AI-generated.
Pass --include_domain to add a "routed_domain" field (which of the 5
robustness-grid domains - clean/jpeg/spatial/noise/colorjitter - the
domain classifier routed the image to) for manual/debugging use; the
required deliverable shape ({"image_path", "pred"}) is unchanged
without that flag.

Pipeline (see production_pipeline.py): a cheap classical-feature domain
classifier detects which robustness-grid domain (if any) the image looks
like it's in, then routes to a domain-specialized logistic-regression
head trained on CLIP pre-projection embeddings extended with a
jpeg_q50 reactivity-delta feature (see README "Reactivity-delta
feature"). Final production numbers: mean AUROC 0.997 and mean accuracy
97.6% across the full 16-cell robustness grid, generator-diagnostic
pooled AUROC 0.977 - see README "Results" for the full table and
model/model_meta.json for the complete version history.
"""

import argparse
import json
import os

from clip_features import load_image_capped
from production_pipeline import predict_proba

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(input_dir):
    paths = []
    for f in sorted(os.listdir(input_dir)):
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS:
            paths.append(os.path.join(input_dir, f))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory of images to classify.")
    parser.add_argument("--output", required=True, help="Path to write the output JSON to.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--include_domain", action="store_true",
                         help="Add a routed_domain field per image (debugging aid, not part of the required output shape).")
    args = parser.parse_args()

    paths = list_images(args.input_dir)
    print(f"Found {len(paths)} images in {args.input_dir}")

    results = []
    for i in range(0, len(paths), args.batch_size):
        batch_paths = paths[i:i + args.batch_size]
        imgs, valid_paths = [], []
        for p in batch_paths:
            try:
                imgs.append(load_image_capped(p))
                valid_paths.append(p)
            except Exception as e:
                print(f"  WARNING: could not open {p}: {e}")
        if not imgs:
            continue
        probs, groups = predict_proba(imgs, batch_size=args.batch_size)
        for p, prob, group in zip(valid_paths, probs, groups):
            record = {"image_path": p, "pred": float(prob)}
            if args.include_domain:
                record["routed_domain"] = str(group)
            results.append(record)
        print(f"  processed {min(i + args.batch_size, len(paths))}/{len(paths)}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
