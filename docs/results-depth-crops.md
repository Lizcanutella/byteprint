# Depth × crops — the two gains are real, and they are largely the same gain

Two runs beat the shipped SigLIP2 baseline on axes that looked independent, and
neither tested the other's:

| | pooled AUC | TPR@1%FPR |
|---|---|---|
| shipped: pooler, 2 crops | 0.9497 | 0.5854 |
| [crop count](results-crop-pooling.md): pooler, **8 crops** | 0.9688 | 0.6995 |
| [read depth](results-depth-frontier.md): **layer 12 + pooler**, 2 crops | 0.9717 | 0.6922 |

The tempting arithmetic is 0.9497 + 0.019 + 0.022 ≈ 0.99. **It does not work,
and this run is why.** Roughly a quarter of the combined gain is not there, and
the half that goes missing is the one you would have quoted.

## The 2×2

One 8-crop extraction with all twelve depth blocks contains every cell, because
depth taps come free from one forward pass and crop count is a *read-time* knob
(`--crop-limit`; `texture` is prefix-stable, so an image's first two of eight
crops are the two a `--crops 2` extraction would have chosen). Every cell shares
one train split, one seed, one calibration holdout and one ladder — so the
interaction is measured, not inferred across runs.

**Pooled AUC over the full §5.2 ladder:**

| read | 2 crops | 8 crops | gain from crops |
|---|---|---|---|
| pooler | 0.9423 | 0.9608 | **+0.0185** |
| layer 12 + pooler | 0.9611 | **0.9708** | +0.0097 |
| **gain from depth** | **+0.0188** | +0.0100 | |

Read the margins. Each gain is worth about **+0.019 on its own and about +0.010
once the other is already there.** Additivity predicts 0.9796; the measured cell
is 0.9708. The interaction term is **−0.0088**, and the combination delivers
0.0285 of an expected 0.0373 — **76% of the additive prediction**.

The symmetry is the informative part: depth keeps 53% of its value in the
presence of extra crops, crop count keeps 52% of its value in the presence of
depth. Neither is subordinate to the other. That is the signature of two methods
recovering **overlapping evidence**, not of one being a better version of the
other.

**At the operating point the redundancy is worse, not better:**

| TPR@1%FPR | 2 crops | 8 crops | gain from crops |
|---|---|---|---|
| pooler | 0.5217 | 0.6613 | +0.1396 |
| layer 12 + pooler | 0.5917 | 0.6767 | +0.0850 |
| **gain from depth** | **+0.0700** | **+0.0154** | |

Depth is worth +0.0700 at 1% FPR when you have two crops and **+0.0154** when
you have eight — it loses **78%** of its value. Anyone quoting the depth study's
"+18% relative at 1% FPR" for an 8-crop detector would be overstating it by
roughly a factor of four.

## Every read, at both crop counts

Half-scale train split (8,000 images), full-scale ladder (1,600 test images ×
15 rungs = 24,000 views).

### 8 crops

| tap | AUC | TPR@1%FPR | TPR@0.1%FPR | full-syn | tampered | worst rung | LOGO |
|---|---|---|---|---|---|---|---|
| layer 9 | 0.9677 | 0.6434 | 0.3397 | 0.9858 | 0.9495 | blur:2.0 0.9062 | 0.6614 |
| **layer 12** | **0.9712** | **0.6842** | 0.3890 | 0.9837 | **0.9586** | noise:0.10 0.9215 | 0.6573 |
| pooler | 0.9608 | 0.6613 | **0.4258** | 0.9814 | 0.9403 | noise:0.10 0.8897 | **0.7767** |
| layer 12 + pooler | 0.9708 | 0.6767 | 0.4024 | 0.9830 | **0.9586** | noise:0.10 0.9207 | 0.7341 |

### 2 crops

| tap | AUC | TPR@1%FPR | TPR@0.1%FPR | full-syn | tampered | worst rung | LOGO |
|---|---|---|---|---|---|---|---|
| layer 9 | 0.9554 | 0.5387 | 0.1453 | 0.9788 | 0.9319 | blur:2.0 0.8738 | 0.6769 |
| layer 12 | 0.9562 | 0.5257 | 0.2889 | 0.9766 | 0.9359 | noise:0.10 0.8934 | 0.6514 |
| pooler | 0.9423 | 0.5217 | 0.2657 | 0.9765 | 0.9081 | noise:0.10 0.8502 | 0.7185 |
| **layer 12 + pooler** | **0.9611** | **0.5917** | 0.2889 | **0.9813** | **0.9408** | noise:0.10 0.8977 | **0.7277** |

## Three things that fall out of it

### 1. Fusing depths stops paying once you have crops

At 2 crops, `layer 12 + pooler` (0.9611) beats layer 12 alone (0.9562) by
+0.0049 — the depth-diversity result the frontier run reported. At 8 crops that
reverses to −0.0004: **layer 12 alone (0.9712) and layer 12 + pooler (0.9708)
are tied**, and the extra 1,152 columns buy nothing. Whatever the pooled output
was contributing on top of a mid-depth tap, eight crops already supply it.

This is worth stating plainly because the *simpler and cheaper* detector wins
here. The best in-distribution read in this run is a single mid-depth tap.

### 2. At the strictest threshold, adding depth actively hurts

TPR@0.1%FPR at 8 crops: the pooler alone scores **0.4258**, and adding layer 12
drops it to 0.4024. At 2 crops the same move was neutral-to-positive
(0.2657 → 0.2889). The most-conservative operating point is the one place where
the shipped read is still the best thing we have at 8 crops, and it is the
operating point a deployment with a low false-positive budget would actually
use.

### 3. The pooler's transfer advantage is real, persistent, and grows

LOGO mean, unseen manipulation type:

| | 2 crops | 8 crops |
|---|---|---|
| pooler | 0.7185 | **0.7767** |
| layer 12 + pooler | 0.7277 | 0.7341 |
| layer 12 | 0.6514 | 0.6573 |

**0.7767 is the highest transfer number this project has recorded**, and it
belongs to the read with the *worst* in-distribution AUC of the four. That
trade-off was visible in the depth frontier and is now confirmed at a second
crop count, with the gap widening rather than closing. Attention pooling is
buying generalisation and paying for it in in-distribution accuracy. Nothing
here explains why, and it is the most interesting open question in the project.

## The gate, and what this run cannot say

**The train split is half scale.** The materialised splits are guarded — their
cache keys include mtime, and rewriting them would invalidate every cache on
disk including the published baselines — so the train side ran on a strided
symlink subset of 8,000 of the 16,000 images (`scripts/make_subset.py`). That
was the only way to fit both extractions inside the hour this run was given.

The cost of that is measurable and consistent, which is what makes the run
usable:

| | this run (8k train) | published (16k train) | penalty |
|---|---|---|---|
| pooler, 2 crops | 0.9423 | 0.9497 | −0.0074 |
| pooler, 8 crops | 0.9608 | 0.9688 | −0.0080 |

Halving the training set costs a flat **~0.0077 AUC** at both crop counts. Every
comparison in this document is *within* the run, where that penalty applies
equally to all four cells and cancels. **No number here should be quoted against
a published one.** A full-scale layer-12-at-8-crops read would plausibly land
near 0.979, but that is an extrapolation from a two-point calibration and it has
not been measured — it is the obvious next run, and it costs one GPU-hour.

**Not established.** That the interaction stays at −0.0088 at full train scale.
That 8 crops is the right number — the curve beyond it is still unmeasured. That
any of this holds off SID_Set, or against an unseen *generator* rather than an
unseen manipulation type. One seed, one split; the 0.0004 between layer 12 and
layer 12 + pooler at 8 crops is well inside noise and those two are tied, not
ranked.

## Cost

| | |
|---|---|
| Compute | 2 GPUs of a SLURM cluster, run concurrently |
| Wall clock | **1h00m** end to end (03:08:36 → 04:08:40) |
| — ladder extraction, 24,000 views | 50m42s |
| — train extraction, 24,000 views | 51m21s (concurrent) |
| — eight probes + LOGO, CPU only | 7m06s |
| GPU time | ~1h42m |
| Failures | 0 of 48,000 views |
| Cache | 9.9 GB ladder + 8.0 GB train (192,000 crop rows × 13,824 columns each) |

The two extractions share no state and write different caches, so they were
submitted as separate jobs on separate GPUs; that is the only reason this fit
the budget. The analysis is 7 minutes because all eight probes are column slices
of one extraction — the 2×2 cost one extraction, not four.

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
"$BYTEPRINT_ROOT"/.venv/bin/python src/scripts/make_subset.py \
    data/sid_train data/sid_train_half --stride 2

BYTEPRINT_STAGE=ladder sbatch --nodelist="$BYTEPRINT_GPU_NODE" \
    --export=ALL,BYTEPRINT_SRC="$BYTEPRINT_ROOT"/src src/scripts/run_depth_crops.sbatch
BYTEPRINT_STAGE=train  sbatch --nodelist="$BYTEPRINT_GPU_NODE" \
    --export=ALL,BYTEPRINT_SRC="$BYTEPRINT_ROOT"/src src/scripts/run_depth_crops.sbatch
# once both land
BYTEPRINT_STAGE=analyze sbatch --nodelist="$BYTEPRINT_GPU_NODE" \
    --export=ALL,BYTEPRINT_SRC="$BYTEPRINT_ROOT"/src src/scripts/run_depth_crops.sbatch
```

Set `BYTEPRINT_TRAIN_DIR=data/sid_train` for the full-scale version, and expect
the train extraction to take roughly twice as long — which puts it outside a
one-hour budget and is exactly the trade this run made.
