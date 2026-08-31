# Pooling crop evidence — the hypothesis is refuted, and the free control is the result

`docs/results-crop-localisation.md` closed by arguing that pooling was the
binding constraint on tampered detection: crop embeddings were mean-pooled into
one cached row per image, so a localised signal was averaged against authentic
content before the head ever saw it. Better crop *placement* had not plateaued —
it had regressed, from two independent cues at once — which is stronger evidence
than a plateau, and it pointed here.

**Max and top-k pooling over crop scores do not beat mean pooling.** On both
backbones, at matched crop count, every score-space arm sits below the control on
pooled AUC, on tampered AUC, and on both operating points. The prediction written
down before the run was wrong.

**What did move the numbers was crop count**, an ablation that only existed
because it was free. On SigLIP2, going from 2 crops to 8 with the pooling
untouched lifts pooled AUC 0.9496 → **0.9688**, TPR@1%FPR 0.5829 → **0.6995**,
and TPR@0.1%FPR 0.2581 → **0.4626**. That is the largest single improvement this
project has recorded, and it has nothing to do with pooling.

## The gate

`texture` crop selection ranks a fixed candidate set and returns its head, so it
is prefix-stable in `top_k`: the first two of an eight-crop selection are the
same pixels a `--crops 2` run would have chosen. Arm 1 exploits that to
reproduce the published baseline from the new cache, and it ran first because
nothing below it counts if it fails.

On `dinov2_large_hf` it reproduces **exactly** — 0.9025 pooled, 0.3362
TPR@1%FPR, 0.1325 TPR@0.1%FPR, 0.8513 tampered, 0.9537 full-synthetic, 0.6457
LOGO mean, and all fifteen rungs to four decimals. The cache restructuring
changed nothing, and prefix stability holds on real images and not merely on the
fixture where it was first checked.

On `siglip2_so400m_hf` it lands **very close but not exactly**: 0.9496 against
the published 0.9497 pooled, 0.7206 against 0.7208 transfer, and 0.5829 against
0.5854 TPR@1%FPR. The AUC gap of 0.0001 is float noise. The TPR gap of 0.0025 is
larger because a TPR at a fixed FPR is far more fragile than an AUC: the
threshold is pinned by the 120th-highest-scoring real image out of 12,000, so a
perturbation too small to move the ranking can still nudge the threshold and
carry ~30 fakes across it. Recorded rather than smoothed over — the gate was
defined on the dinov2 numbers, which did reproduce exactly, and this one is a
secondary check that came within float noise on AUC.

## The arms

Seven arms per backbone, all reading **one** extraction. `--crop-limit` is "-"
where every crop is used.

### `dinov2_large_hf`

| arm | fit / pool | crops | pooled | TPR@1% | TPR@0.1% | tampered | full-syn | LOGO |
|---|---|---|---|---|---|---|---|---|
| 1 | mean / mean | 2 | 0.9025 | 0.3362 | 0.1325 | 0.8513 | 0.9537 | 0.6457 |
| 2 | mean / mean | 8 | **0.9324** | **0.4652** | 0.1141 | **0.8974** | 0.9674 | 0.6475 |
| 3 | crop / mean-score | 8 | 0.9306 | 0.4601 | 0.1071 | 0.8848 | **0.9763** | 0.6806 |
| 4 | crop / max | 8 | 0.9282 | 0.4069 | 0.0893 | 0.8871 | 0.9694 | 0.6791 |
| 5 | crop / topk:2 | 8 | 0.9313 | 0.4063 | 0.0612 | 0.8904 | 0.9722 | **0.6807** |
| 6 | mean / max | 8 | 0.9223 | 0.3569 | 0.1199 | 0.8839 | 0.9607 | 0.6683 |
| 7 | mean / topk:2 | 8 | 0.9265 | 0.3761 | 0.1139 | 0.8877 | 0.9652 | 0.6705 |

### `siglip2_so400m_hf`

| arm | fit / pool | crops | pooled | TPR@1% | TPR@0.1% | tampered | full-syn | LOGO |
|---|---|---|---|---|---|---|---|---|
| 1 | mean / mean | 2 | 0.9496 | 0.5829 | 0.2581 | 0.9176 | 0.9817 | 0.7206 |
| 2 | mean / mean | 8 | **0.9688** | **0.6995** | **0.4626** | **0.9494** | 0.9882 | **0.7642** |
| 3 | crop / mean-score | 8 | 0.9650 | 0.6392 | 0.3588 | 0.9397 | 0.9902 | 0.7457 |
| 4 | crop / max | 8 | 0.9618 | 0.6483 | 0.3193 | 0.9336 | 0.9900 | 0.7506 |
| 5 | crop / topk:2 | 8 | 0.9644 | 0.6608 | 0.3282 | 0.9382 | **0.9906** | 0.7510 |
| 6 | mean / max | 8 | 0.9613 | 0.6275 | 0.3203 | 0.9345 | 0.9880 | 0.7608 |
| 7 | mean / topk:2 | 8 | 0.9640 | 0.6411 | 0.3611 | 0.9389 | 0.9891 | 0.7620 |

## Reading it

### Pooling is refuted, on both backbones

Arm 2 is the control every score-space arm must beat, and none of them does. The
best tampered AUC among the score-space arms is 0.8904 (dinov2) and 0.9397
(SigLIP2), against controls of 0.8974 and 0.9494. The gap is small but it is
consistently in the *wrong* direction, on 12 of 12 backbone × arm comparisons for
tampered AUC.

The damage is worst where this project says it matters most. Max and top-k gut
the very-low-FPR operating point: dinov2 arm 5 scores 0.0612 at 0.1% FPR against
the control's 0.1141, and SigLIP2 arm 4 scores 0.3193 against 0.4626. There is a
mechanism for that. A max takes each image's single most extreme crop, and for an
authentic photograph the most extreme crop is the noisiest one — so max pooling
manufactures exactly the confident false positives that a 0.1% FPR budget cannot
absorb. Averaging suppresses that tail; taking a maximum selects for it.

### Crop count carries the gain, and only the free control could show it

Arms 1 and 2 differ in nothing but how many of the same cached crops are pooled.

| | dinov2_large | siglip2_so400m |
|---|---|---|
| pooled AUC | 0.9025 → **0.9324** | 0.9496 → **0.9688** |
| TPR @ 1% FPR | 0.3362 → **0.4652** | 0.5829 → **0.6995** |
| TPR @ 0.1% FPR | 0.1325 → 0.1141 | 0.2581 → **0.4626** |
| tampered | 0.8513 → **0.8974** | 0.9176 → **0.9494** |

This is the result of the run, and it was very nearly not measured. `--crop-limit`
was added as a correctness control for a cache refactor, not as an experiment;
it became the experiment. Note also that it is *not* uniformly good — dinov2's
TPR@0.1%FPR falls, the one place more crops cost something.

The honest reading of the whole table is therefore: **had arm 2 not been on the
sheet, arms 4 and 5 would have looked like a win.** Against the published
baseline, dinov2 arm 5 shows tampered 0.8513 → 0.8904 and SigLIP2 arm 5 shows
0.9176 → 0.9382, and both would have been reported as the pooling fix working.
The control that separates them exists only because prefix stability made it
cost nothing, which is the methodological point worth carrying forward.

### The label-noise prediction is confirmed by its signature

The spec predicted that crop-level training would carry real label noise, because
a tampered image's crops are mostly authentic content wearing a synthetic label,
while a fully-synthetic image's crops are all correctly labelled.

That is exactly the shape the data has, on both backbones. Crop-level fitting
produces the **best full-synthetic AUC in each table** — 0.9763 (dinov2, arm 3)
and 0.9906 (SigLIP2, arm 5), both above their controls — while producing a
**worse tampered AUC** than the same control. The cost lands precisely on the
class whose crop labels are wrong, and the benefit precisely on the class whose
crop labels are right. Eight times the training rows helps when the rows are
correct and hurts when they are not.

This is the one part of the pre-registered reasoning that survived intact, and it
survived as a mechanism rather than as a number.

### The transfer result does not replicate — and that is why the second backbone ran

On dinov2, crop-level training looked like a clear win for unseen-generator
transfer: LOGO 0.6475 → ~0.680 across all three crop-fit arms, decomposing
cleanly into +0.021 from score pooling alone (arms 2 → 6 share an identical
fitted head and differ only in the eval reduction) and a further +0.011 from
crop-level training (arms 6 → 4, reduction held at max). Consistent across three
arms, with a clean decomposition. It was the obvious silver lining.

**SigLIP2 reverses it.** There the control has the *best* LOGO of any arm
(0.7642), and every crop-fit arm is worse (0.7457–0.7510). The effect is not a
property of pooling or of crop-level training; it is a property of dinov2 in this
configuration.

Reported at length because the failure mode is the interesting part. One backbone
gave three mutually-consistent arms and a tidy additive decomposition, which is
precisely the kind of evidence that feels conclusive. It was not. The second
backbone cost one GPU slot and no wall clock, and it is the only reason this
document does not contain a confident, wrong paragraph about pooling improving
generalisation.

## Cost

| | |
|---|---|
| Compute | 2 GPUs of a SLURM cluster, run concurrently |
| Wall clock | **2h43m** (dinov2 2h33m, SigLIP2 2h43m) |
| GPU time | ~5h16m |
| Failures | 0 of 48,000 train views and 24,000 ladder views, per backbone |
| Cache | 1.5 GB train + 757 MB ladder per backbone, at 8 crops |

Cheaper in wall clock than the 3h45m crop-mode comparison it follows, despite
running seven arms per backbone against that run's three. The reason is the
schema change: pooling is now decided at train time, so all seven arms read one
extraction and each additional arm costs a logistic regression rather than an
hour of GPU. That is the practical argument for the refactor, and it stands
whether or not the hypothesis that motivated it survived.

The storage cost is real and should be stated: schema 2 holds one row per crop,
so a cache is `crops_per_image` times larger. At 8 crops that is 2.25 GB per
backbone per split-pair, against roughly 280 MB before.

**Scoring at 8 crops costs 4× the backbone forward of scoring at 2.** The
accuracy gain is large enough to be worth it, but it is not free, and any
deployment claim has to carry it.

## What this establishes, and what it does not

**Established.** Max, top-k and mean-score pooling over crop scores do not
improve tampered AUC over mean pooling at matched crop count, on two backbones,
and they measurably damage the low-FPR operating point. Crop count from 2 to 8 is
a large and consistent improvement on both. Crop-level training helps the class
whose crop labels are correct and hurts the class whose labels are not, exactly
as predicted.

**Not established.** That 8 is the right number of crops — 2 and 8 are the only
points measured, the curve between and beyond them is unknown, and the one metric
that *fell* on dinov2 (TPR@0.1%FPR) suggests it is not monotone in every respect.
That any of this transfers off SID_Set. That a learned attention pooling, rather
than a fixed reduction, would fail the same way; every pooling tested here is
parameter-free, and the label-noise result hints that the useful thing to learn
might be which crops to *trust* rather than which to take the max over.

**What is now the leading explanation for the tampered gap.** Neither crop
placement nor crop pooling. Two rounds of the same shape have now come back
negative, and both times the mechanism was plausible and the fixture supportive.
The thing that has actually moved tampered detection twice is *seeing more of the
image* — which is a coverage story, not a localisation story, and it is the
opposite of the intuition both rounds were built on.

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_pooling.sbatch
BYTEPRINT_BACKBONE=siglip2_so400m_hf sbatch --nodelist="$BYTEPRINT_GPU_NODE" \
    src/scripts/run_pooling.sbatch
```

Read arm 1 before anything else: it is a gate, not a result. Then read arm 2
before arms 3–7, because the pooled AUC of a score-space arm means nothing
against the published baseline and everything against the control at the same
crop count.
