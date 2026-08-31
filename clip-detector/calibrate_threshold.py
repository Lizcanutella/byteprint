"""
Calibrates a decision threshold at a target false-positive rate, using
BYTEPRINT's own threshold_at_fpr methodology (byteprint/metrics.py) -
the fixed 0.5 cutoff used everywhere else in this project ignores that
score distributions shift between domains/generators, and on a real
platform authentic images vastly outnumber synthetic ones, so accuracy
at 0.5 flatters a detector that would be unusable in practice.

Uses the POOLED predictions from the full 16-cell robustness grid
(results_detector/robustness_per_image.json, 7088 images, 2640 real)
rather than a small dedicated calibration split, since a target FPR of
1% needs enough real images to give a stable quantile threshold - 2640
pooled real images (vs 443 in a single clean-only split) makes the
1%-FPR threshold meaningfully less noisy. This reuses the same held-out
pool used for reporting the headline numbers (not the training data),
consistent with how this project's other diagnostics calibrate.

Also recomputes accuracy/FPR/FNR at the calibrated threshold for the
full 16-cell table, alongside the existing fixed-0.5 numbers, so the
trade-off is visible rather than hidden.

Usage:
    python calibrate_threshold.py
"""

import json

import numpy as np


def threshold_at_fpr(y, s, target_fpr):
    negatives = np.sort(s[y == 0])
    allowed = int(np.floor(target_fpr * negatives.size + 1e-9))
    if allowed >= negatives.size:
        return -np.inf
    cutoff = negatives[negatives.size - allowed - 1]
    return float(np.nextafter(cutoff, np.inf))


def metrics_at_threshold(y, s, threshold):
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    return {"accuracy": acc, "fpr": fpr, "fnr": fnr, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    d = json.load(open("results_detector/robustness_per_image.json"))

    all_labels, all_scores = [], []
    for cell, records in d.items():
        for r in records:
            all_labels.append(r["label"])
            all_scores.append(r["pred_prob"])
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    target_fprs = [0.01, 0.001]
    thresholds = {tf: threshold_at_fpr(all_labels, all_scores, tf) for tf in target_fprs}
    print("Calibrated thresholds (pooled across all 16 cells, 2640 real images):")
    for tf, t in thresholds.items():
        print(f"  target_fpr={tf}: threshold={t:.6f}")

    chosen_fpr = 0.01
    chosen_threshold = thresholds[chosen_fpr]

    print(f"\nRecomputing per-cell metrics at the calibrated threshold ({chosen_threshold:.6f}, "
          f"target_fpr={chosen_fpr}) vs. the original fixed 0.5:")
    per_cell = {}
    for cell, records in d.items():
        y = np.array([r["label"] for r in records])
        s = np.array([r["pred_prob"] for r in records])
        m_fixed = metrics_at_threshold(y, s, 0.5)
        m_calib = metrics_at_threshold(y, s, chosen_threshold)
        per_cell[cell] = {"fixed_0.5": m_fixed, f"calibrated_{chosen_fpr}fpr": m_calib}
        print(f"  {cell:<20} fixed_0.5: acc={m_fixed['accuracy']:.4f} fpr={m_fixed['fpr']:.4f} fnr={m_fixed['fnr']:.4f}"
              f"   |   calibrated: acc={m_calib['accuracy']:.4f} fpr={m_calib['fpr']:.4f} fnr={m_calib['fnr']:.4f}")

    mean_acc_fixed = float(np.mean([v["fixed_0.5"]["accuracy"] for v in per_cell.values()]))
    mean_acc_calib = float(np.mean([v[f"calibrated_{chosen_fpr}fpr"]["accuracy"] for v in per_cell.values()]))
    mean_fpr_calib = float(np.mean([v[f"calibrated_{chosen_fpr}fpr"]["fpr"] for v in per_cell.values()]))
    mean_fnr_calib = float(np.mean([v[f"calibrated_{chosen_fpr}fpr"]["fnr"] for v in per_cell.values()]))
    print(f"\nMean accuracy: fixed_0.5={mean_acc_fixed:.4f}  calibrated={mean_acc_calib:.4f}")
    print(f"Mean FPR at calibrated threshold: {mean_fpr_calib:.4f} (target was {chosen_fpr})")
    print(f"Mean FNR at calibrated threshold: {mean_fnr_calib:.4f}")

    calibration = {
        "method": "threshold_at_fpr, same methodology as byteprint/metrics.py",
        "calibration_data": "pooled predictions across all 16 robustness-grid cells "
                             "(results_detector/robustness_per_image.json), 7088 images, 2640 real",
        "thresholds": thresholds,
        "recommended_threshold": chosen_threshold,
        "recommended_target_fpr": chosen_fpr,
        "mean_accuracy_at_recommended_threshold": mean_acc_calib,
        "mean_accuracy_at_fixed_0.5": mean_acc_fixed,
        "mean_fpr_at_recommended_threshold": mean_fpr_calib,
        "mean_fnr_at_recommended_threshold": mean_fnr_calib,
    }
    with open("model/calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    with open("results_detector/robustness_table_calibrated.json", "w") as f:
        json.dump(per_cell, f, indent=2)
    print("\nSaved model/calibration.json and results_detector/robustness_table_calibrated.json")


if __name__ == "__main__":
    main()
