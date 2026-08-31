"""
Error analysis: pulls representative false positives (real images
classified as AI) and false negatives (AI images classified as real)
from robustness_eval.py's per-image predictions, across both the clean
condition and transformed conditions, saves example images with their
predicted probability, and writes a short written summary of the
patterns and trade-offs - the hackathon's required "Error Analysis
Note."

Usage:
    python error_analysis.py   (run robustness_eval.py first)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clip_features import load_image_capped

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_detector")
N_EXAMPLES = 6


def load_per_image():
    with open(os.path.join(RESULTS_DIR, "robustness_per_image.json")) as f:
        return json.load(f)


def save_example_grid(records, title, out_path):
    records = records[:N_EXAMPLES]
    if not records:
        return None
    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, rec in zip(axes, records):
        img = load_image_capped(rec["path"])
        ax.imshow(img)
        true_label = "real" if rec["label"] == 0 else "ai"
        ax.set_title(f"{true_label} | pred={rec['pred_prob']:.2f}\n{os.path.basename(rec['path'])}",
                      fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    return out_path


def main():
    per_image = load_per_image()

    summary = {}
    saved_plots = []
    for transform_name, records in per_image.items():
        fps = sorted(
            [r for r in records if r["label"] == 0 and r["pred_prob"] >= 0.5],
            key=lambda r: -r["pred_prob"],
        )
        fns = sorted(
            [r for r in records if r["label"] == 1 and r["pred_prob"] < 0.5],
            key=lambda r: r["pred_prob"],
        )
        n_real = sum(1 for r in records if r["label"] == 0)
        n_ai = sum(1 for r in records if r["label"] == 1)
        summary[transform_name] = {
            "n_false_positives": len(fps),
            "n_false_negatives": len(fns),
            "fpr": len(fps) / max(n_real, 1),
            "fnr": len(fns) / max(n_ai, 1),
        }

        if transform_name in ("clean", "jpeg_q30", "blur_s2.0", "crop80"):
            fp_path = os.path.join(RESULTS_DIR, f"error_fp_{transform_name}.png")
            fn_path = os.path.join(RESULTS_DIR, f"error_fn_{transform_name}.png")
            if save_example_grid(fps, f"False positives (real->AI) [{transform_name}]", fp_path):
                saved_plots.append(fp_path)
            if save_example_grid(fns, f"False negatives (AI->real) [{transform_name}]", fn_path):
                saved_plots.append(fn_path)

    summary_path = os.path.join(RESULTS_DIR, "error_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Error rates by transform (FPR = real misclassified as AI, FNR = AI misclassified as real):")
    header = f"{'transform':<20}{'FPR':<10}{'FNR':<10}{'n_FP':<8}{'n_FN':<8}"
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(f"{name:<20}{s['fpr']:<10.4f}{s['fnr']:<10.4f}{s['n_false_positives']:<8}{s['n_false_negatives']:<8}")

    print(f"\nSaved summary: {summary_path}")
    for p in saved_plots:
        print(f"Saved example grid: {p}")


if __name__ == "__main__":
    main()
