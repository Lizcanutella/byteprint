# First real run — SID_Set, DINOv2-L probe

The first BYTEPRINT result on real data. Everything before this was the
synthetic fixture, which only ever proved the wiring worked.

## Setup

| | |
|---|---|
| Backbone | `dinov2_large_hf` (ViT-L/14, ~0.3B params — frozen, never fine-tuned) |
| Crops | 2 per image, 224px, `texture` mode, native resolution |
| Head | logistic regression, calibrated at 1% FPR |
| Train | 16,000 SID_Set images (8,000 real / 4,000 full-synthetic / 4,000 tampered), `--augment 3` |
| Test | 1,600 held-out images from SID_Set's own validation shards, × 15 §5.2 rungs = 24,000 views |
| Compute | one 48 GB GPU on a SLURM cluster, **4h 40m** wall clock, 0 failures |

Train and test are disjoint. Neither touches the competition's demonstration
set (COCO val2017 + DALL·E Advanced), which is never used for training.

## The headline: the ladder is flat

Pooled over all fifteen rungs: **AUC 0.9025**, TPR@1%FPR 0.3362, TPR@0.1%FPR
0.1325.

| rung | AUC | TPR@1%FPR |
|---|---|---|
| `none` (clean) | 0.9112 | 0.4100 |
| `jpeg:90` | 0.9175 | 0.4188 |
| `jpeg:70` | 0.9183 | 0.3875 |
| `jpeg:50` | 0.9038 | 0.3937 |
| `jpeg:30` | 0.8966 | 0.3137 |
| `blur:0.5` | 0.9176 | 0.3862 |
| `blur:1.0` | **0.9191** | 0.3750 |
| `blur:2.0` | 0.8889 | 0.3675 |
| `scale:0.5` | 0.9084 | 0.3312 |
| `scale:0.25` | 0.8899 | 0.3325 |
| `noise:0.02` | 0.9023 | 0.3362 |
| `noise:0.05` | 0.8897 | 0.3200 |
| `noise:0.10` | **0.8553** | 0.2550 |
| `jitter:0.2` | 0.9120 | 0.4138 |
| `crop:0.8` | 0.9126 | 0.3450 |

The whole ladder spans **0.064 AUC**, from 0.8553 to 0.9191. Robustness is the
graded axis, and this is the result to point at: heavy Gaussian noise is the
worst case and it costs six points, not the detector.

Two things are worth noticing.

**Clean is not the best rung.** `blur:1.0`, `jpeg:70`, `blur:0.5` and `jpeg:90`
all edge past it. That is a consequence of `--augment 3`: augmentation
*replaces* the spec list rather than adding to it, so the probe was trained
almost entirely on laundered views and barely saw a clean image. Training on the
damage moved the whole operating range onto laundered data, which is where
deployed images live.

**Compare against the fixture.** On the synthetic fixture, `blur:2.0` and
`scale:0.25` fell *below chance* (0.355 and 0.410) because the planted signal was
pure high-frequency texture. On real data those same rungs hold 0.889 and 0.890.
The fixture's failure mode was an artifact of the fixture.

## The honest weaknesses

**The operating point is mediocre.** AUC 0.90 sounds respectable; TPR@1%FPR of
0.336 is what a deployment would actually feel. At a threshold strict enough to
wrongly flag 1 authentic image in 100, two thirds of AI-generated images get
through. At 0.1% FPR it is 0.133. This is exactly why the convention here is to
report TPR at a fixed low FPR alongside AUC — an average over a ranking metric
hides it.

**Tampered images are much harder than fully synthetic ones.**

| class | AUC | n |
|---|---|---|
| full synthetic | 0.9537 | 6,000 |
| tampered | 0.8513 | 6,000 |

Plausible mechanism, not yet verified: only a *region* of a tampered image is
generated, and `texture` crop selection picks the most high-frequency windows in
the frame, which has no particular reason to be the edited region. A crop
strategy that had some notion of *where* to look would be the obvious follow-up.

**Generalisation to an unseen manipulation type is the real gap.**

| held out | AUC | TPR@1%FPR |
|---|---|---|
| `full_synthetic` | 0.6744 | 0.0563 |
| `tampered` | 0.6169 | 0.0829 |
| **mean** | **0.6457** | |

Train on one kind of manipulation, test on the other, and the detector falls
from ~0.90 to ~0.65. Stated precisely: SID_Set gives two *manipulation types*,
not two generators, so this measures transfer to an unseen kind of editing
rather than to an unseen diffusion model. It is still the number that should
temper any claim about generalisation.

## An open question these results do not settle

The materialisation audit, at full scale:

| label | source containers |
|---|---|
| 0 real | 7,993 JPEG + 7 MPO (**100% JPEG-family**) |
| 1 full synthetic | 4,000 PNG (**100% PNG**) |
| 2 tampered | 3,382 JPEG + 618 PNG |

Real and fully-synthetic are perfectly separated by file format. Every class is
re-encoded to PNG before extraction, so the *container* cannot be the
classifier — but the reals still carry JPEG compression history in their pixels
and the synthetics do not, and that is a property of how the dataset was built
rather than of how the images were made.

The flat JPEG rungs are *suggestive* that this is not the dominant signal — if
it were, compressing both classes ought to hurt more than it does — but they do
not settle it, because the probe was trained on JPEG'd views of both classes and
had every opportunity to learn features that survive it. The clean control is to
re-encode both classes through JPEG-95 before extraction and retrain. Until that
runs, **treat 0.9025 as an upper bound.**

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_sid_set.sbatch
```

Stage timings, for sizing a follow-up: materialisation 1h40m (16,000 images at
~2.7/s), train extraction 2h04m (48,000 views at ~6.4/s), ladder extraction and
reporting 56m (24,000 views at ~7.1/s). This run predates `--workers`; with
`--workers 4` the two extraction stages should come down by roughly half.
