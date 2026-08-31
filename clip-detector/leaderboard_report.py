"""
Render results/leaderboard.json as a sorted comparison table across every
signal tried so far, and flag cross-dataset consistency: for a
(signal, feature) that has been run on more than one profile, do the
profiles agree on the DIRECTION of the effect (same sign of
raw_auroc - 0.5)? A signal whose apparent direction flips between
datasets is dataset-specific noise, not a real effect - this is a
cross-dataset analogue of the leakage/shortcut checks already run per
profile.

Usage:
    python leaderboard_report.py
"""

import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEADERBOARD_PATH = os.path.join(BASE_DIR, "results", "leaderboard.json")


def load_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH):
        return []
    with open(LEADERBOARD_PATH) as f:
        return json.load(f)


def main():
    rows = load_leaderboard()
    if not rows:
        print(f"No leaderboard entries yet at {LEADERBOARD_PATH}.")
        return

    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["signal"], r["feature"])].append(r)

    print("=" * 110)
    print("SIGNAL LEADERBOARD")
    print("=" * 110)
    header = (f"{'signal':<22}{'feature':<28}{'profile':<12}{'testA':<8}{'testB':<8}"
              f"{'leak':<6}{'shortcut':<10}{'n':<6}")
    print(header)
    print("-" * len(header))

    # sort groups by best effective_auroc_testA seen (descending), most
    # promising signals first.
    def best_score(key):
        return max(r["effective_auroc_testA"] for r in by_key[key])

    for key in sorted(by_key.keys(), key=best_score, reverse=True):
        signal, feature = key
        group = by_key[key]
        for r in group:
            testB = f"{r['effective_auroc_testB']:.4f}" if r.get("effective_auroc_testB") is not None else "-"
            shortcut = r.get("shortcut_dominates")
            shortcut_str = "n/a" if shortcut is None else ("YES" if shortcut else "ok")
            print(f"{signal:<22}{feature:<28}{r['profile']:<12}{r['effective_auroc_testA']:<8.4f}"
                  f"{testB:<8}{('YES' if r['leakage_flagged'] else 'ok'):<6}"
                  f"{shortcut_str:<10}"
                  f"{r['n_real'] + r['n_ai']:<6}")

        # cross-dataset consistency, only meaningful with >=2 profiles
        if len(group) >= 2:
            signs = [1 if (r["raw_auroc_testA"] - 0.5) >= 0 else -1 for r in group]
            consistent = len(set(signs)) == 1
            min_eff = min(r["effective_auroc_testA"] for r in group)
            verdict = "CONSISTENT direction across profiles" if consistent else "DIRECTION FLIPS across profiles - likely dataset-specific artifact"
            print(f"    -> cross-dataset check ({len(group)} profiles): {verdict}"
                  f" (weakest profile eff.AUROC={min_eff:.4f})")
        print()

    # Overall "survivors": clears the bar on every axis.
    print("=" * 110)
    print("SURVIVORS (Test A eff.AUROC >= 0.65, not leakage-flagged, shortcut-ablation")
    print("           explicitly run and NOT dominated, direction-consistent across profiles")
    print("           where checked). Entries with shortcut_dominates=n/a (not verified) can")
    print("           never survive - that ablation must be run, not assumed passing.")
    print("=" * 110)
    any_survivor = False
    for key in sorted(by_key.keys(), key=best_score, reverse=True):
        signal, feature = key
        group = by_key[key]
        ok = all(r["effective_auroc_testA"] >= 0.65 and not r["leakage_flagged"]
                 and r.get("shortcut_dominates") is False for r in group)
        if len(group) >= 2:
            signs = [1 if (r["raw_auroc_testA"] - 0.5) >= 0 else -1 for r in group]
            ok = ok and len(set(signs)) == 1
        if ok:
            any_survivor = True
            profiles = ", ".join(f"{r['profile']}={r['effective_auroc_testA']:.4f}" for r in group)
            print(f"  {signal}/{feature}: {profiles}")
    if not any_survivor:
        print("  (none yet)")
    print("=" * 110)


if __name__ == "__main__":
    main()
