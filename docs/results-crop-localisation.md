# Crop modes that know where to look — the fixture measurement

The SID_Set run leaves a specific gap: fully-synthetic images score AUC 0.9537,
tampered ones 0.8513. The hypothesis was the crop strategy. `texture` keeps the
windows richest in high-frequency detail, and in a tampered image — a real
photograph with an AI-edited region — the edited region has no reason to be one
of them.

Two crop modes were built to test that hypothesis. On a planted fixture they work
exactly as intended — one of them finds the edited region 40 times out of 40
where `texture` finds it 0 times out of 40.

**On SID_Set they make the tampered number worse.** The hypothesis is refuted,
and the fixture result did not transfer. [Skip to the run](#the-run-on-sid_set-the-hypothesis-is-refuted);
the fixture sections below are kept because how a convincing proxy misled us is
the more useful half of this document.

## The fixture

Each image is low-frequency structure plus uniform sensor grain. The planted
"edit" is one 96×96 region blurred in place: its structure survives, its grain
does not.

That is a proxy for an inpainted region rather than a replica of one, chosen
because it reproduces the property that matters. A region resampled by a decoder
did not pass through a lens, so its high-frequency statistics differ from the
rest of the frame — and they usually differ by being *cleaner*. Everything below
inherits that assumption, which is the main reason these numbers cannot stand in
for a run on real tampered images.

A "hit" is a returned crop sitting at least half inside the planted region.
Two crops per image, 64px, from a fixed grid of candidate windows so that
ranking is measured without sampling noise on top of it.

## `texture` is anti-correlated, not blind

| mode | hits |
|---|---|
| `anomaly` | **40 / 40** |
| `texture` | **0 / 40** |

Zero out of forty is the result worth pausing on. `texture` is not failing to
find the edited region by chance — it is reliably ranking it *last*, because the
region is smoother than the frame around it and smoothness is exactly what the
heuristic sorts to the bottom. That is a mechanism for the 0.851 number, and it
is the reason a better ranking was worth building before anything more expensive.

## The hand-over threshold sits in an empty gap

`anomaly` must not buy the tampered number by spending the full-synthetic one. A
uniformly generated image has no odd region, so its ranking would be noise; when
no window stands out the mode defers to `texture` instead.

| | maximum deviation from the image's own median (MAD units) |
|---|---|
| images carrying a planted edit | min **66.3**, median 67.0 |
| uniform images | max **1.19**, median 0.92 |
| `Z_FLOOR` | **6.0** |

An order of magnitude clear on both sides. The threshold is not tuned onto a
slope, and the fallback fires on 40/40 uniform images. A test pins that when it
fires, the returned ordering is byte-identical to what `texture` mode would have
produced — the fallback is the old behaviour, not an approximation of it.

## Across the §5.2 ladder

Twelve images per rung, two crops each, so 24 hits is the ceiling.

| rung | `anomaly` | `ela` | `texture` | fallback fired |
|---|---|---|---|---|
| `none` | 24 | 24 | 0 | 0/12 |
| `jpeg:90` | 24 | **0** | 0 | 0/12 |
| `jpeg:70` | 24 | 24 | 0 | 0/12 |
| `jpeg:50` | 24 | 24 | 0 | 0/12 |
| `jpeg:30` | 24 | 24 | 0 | 0/12 |
| `blur:0.5` | 24 | 24 | 0 | 0/12 |
| `blur:1.0` | 24 | 24 | 0 | 0/12 |
| `blur:2.0` | **0** | 18 | 0 | **12/12** |
| `scale:0.5` | 24 | 24 | 0 | 0/12 |
| `scale:0.25` | **1** | 24 | 1 | **12/12** |
| `noise:0.02` | 24 | 24 | 0 | 0/12 |
| `noise:0.05` | 24 | 24 | 0 | 0/12 |
| `noise:0.10` | 8 | 24 | 0 | 8/12 |
| `jitter:0.2` | 24 | 24 | 0 | 0/12 |
| `crop:0.8` | 24 | 24 | 0 | 0/12 |

`texture` never finds the region on any rung. `anomaly` holds the ceiling on
twelve of fifteen and goes blind on the two that destroy the grain everywhere —
which is the honest limit of a cue built on grain contrast. Where it goes blind
it goes blind *safely*: the fallback fires on every image, so it degrades to
today's behaviour rather than returning windows ranked by noise. `noise:0.10` is
the partial case, falling back on two thirds of images.

## The control corrected the claim it was built to confirm

`ela` — classic error-level analysis, ranking windows by their response to a
fresh JPEG encode — was registered as the absolute-cue control, on the
expectation that it would collapse under recompression while a within-image
contrast survived. The reasoning was that `jpeg:30` gives the whole frame one
fresh compression history, erasing what ELA reads, while a contrast between a
region and its surroundings largely cancels a uniform transform.

**That expectation was wrong, and the control is what caught it.** ELA holds 24/24
at `jpeg:30`. It also beats `anomaly` outright on `blur:2.0` and `scale:0.25`,
where the compression response of a smoothed region still differs from its
surroundings after the grain that `anomaly` reads is gone.

One implementation detail decided whether this comparison was worth anything.
Ranking ELA by *highest* residual energy — the obvious reading — scores 0/24 on
every rung including clean, because JPEG barely perturbs a smooth region, so raw
ELA is anti-correlated on this fixture for precisely the same reason `texture`
is. Ranked instead by *deviation from the image's median window*, in both
directions, it becomes competitive. The second is also what ELA is in practice:
an analyst looks for the region that responds *unlike* its surroundings, and a
pasted region can respond either more or less than its host. Had the control
shipped in its first form, it would have lost the comparison for a reason with
nothing to do with the cue it exists to represent, and the write-up would have
claimed a win that was an artifact.

So the two cues fail on different rungs rather than together — ELA has its own
blind spot at `jpeg:90`, where re-encoding at the image's own quality leaves
almost no residual to rank on. That is the same complementary-failure argument
this project already makes for fusing two experts, and the reason both stay
registered rather than one replacing the other.

## Cost

Crop selection is the pipeline's bottleneck — more than decoding — so a mode that
multiplies it is not free. Per image at 1024px, 224px crops, 32 candidates:

| mode | relative to `texture` |
|---|---|
| `texture` | 1.0× |
| `anomaly` | **1.8×** |
| `ela` | 0.6× |

Absolute milliseconds are omitted deliberately: the machine was under other load
while these ran, and every mode moved together between runs, so only the ratios
are meaningful.

The first implementation cost 4.5×. Two things fixed it. A window's texture score
is the variance of its luma Laplacian, which is the same band the fingerprint
needs — so both come from one pass and the fallback path, taken on most images,
costs nothing extra. And the fingerprint is computed per window rather than from
a whole-image band map with summed-area tables: a few dozen 224px windows cover
barely more area than the image itself, so the tables cost more to build than the
windows cost to read. The obvious optimisation was the wrong one, and only
profiling said so.

## What this does and does not establish

**Established.** On *this fixture*, `texture` sorts a cleaner-than-its-surroundings
region last, and a within-image outlier ranking finds such a region reliably,
degrades to the existing behaviour instead of to noise when its cue dies, and
costs under 2× the strategy it replaces. All of that held up.

**Not established — and, it turned out, false.** That any of it predicts SID_Set.
The fixture plants one kind of edit, chosen to match the hypothesis, at a known
size, in images whose grain is synthetic and uniform. The run below found the
tampered AUC moving the wrong way.

Three specific transfer risks were written down *before* the run, and the middle
one is now the leading explanation for what happened rather than a hedge:

1. edited regions smaller than a 224px crop, so no window is mostly-edit and the
   contrast is diluted inside every candidate;
2. photographs whose legitimate content already varies enough in grain — sky
   against foliage — to trigger the ranking on an unedited image;
3. `band_map` reading luma only, dropping chroma inconsistency, a genuine
   forensic cue the proxy was blind to and so could not measure the loss of.

**The lesson is about proxy design.** The fixture was not weakly predictive, it
was confidently wrong, and it was confidently wrong because it was built from the
same hypothesis it was used to test: it assumed edited regions are smoother than
their surroundings, then measured whether a smoothness-contrast detector finds
them. 40/40 was never evidence for the hypothesis — it was the hypothesis,
restated. A proxy earns its keep only when it can fail in a way the idea behind
it does not predict, and this one could not.

## The run on SID_Set: the hypothesis is refuted

Three arms, identical in every respect but the crop mode, one 32 GB GPU, **3h45m**
for all three. `texture` was re-run in-job rather than quoted from the first
SID_Set run, so all three share a code version, an extraction path and a card.

| | `texture` | `anomaly` | `ela` |
|---|---|---|---|
| **tampered** | **0.8513** | 0.8317 | 0.8318 |
| full synthetic | **0.9537** | 0.9511 | 0.9505 |
| pooled AUC | **0.9025** | 0.8914 | 0.8912 |
| TPR @ 1% FPR | 0.3362 | **0.3693** | 0.3504 |
| TPR @ 0.1% FPR | 0.1325 | **0.1683** | 0.1357 |
| LOGO mean | **0.6457** | 0.6170 | 0.6211 |

**Tampered AUC is the number this existed to move, and it moved the wrong way**,
by −0.0196. Every rung is slightly down on AUC. The gap between tampered and
fully-synthetic images did not narrow; it widened from 0.1024 to 0.1194.

The strongest part of the result is the agreement between the two arms. `anomaly`
and `ela` read *different physical cues* — grain statistics against compression
response — and land within 0.0001 of each other on tampered and 0.0002 on pooled
AUC. Two independent instruments converging on the same wrong answer is much
better evidence than one failing. It says the problem is not which cue was
chosen; it is the premise that crop placement is what limits tampered detection.

### The one thing that did improve

`anomaly` lifts the operating point while lowering AUC: **+0.0331 at 1% FPR
(+10% relative) and +0.0358 at 0.1% FPR (+27% relative)**, and it holds per rung
— `jpeg:70` 0.3875→0.4550, `scale:0.25` 0.3325→0.3925, `blur:2.0` 0.3675→0.3937.

That is the metric this project argues is the one that counts, since the stated
weakness of the baseline is its operating point rather than its AUC. But it
should be read narrowly, for two reasons. It was not predicted, and there is no
mechanism for it yet. And it is *not* shared by `ela`, whose TPR@0.1%FPR is flat
(0.1357 against 0.1325) despite an almost identical AUC profile — so it is not a
general property of looking at odd regions, and the convenient story that "any
localisation sharpens the confident end" is already contradicted by the control.

A testable guess, written down before anyone goes looking for support: the cue
fires on legitimate grain variation in real photographs — sky against foliage —
which was the second of the three transfer risks listed below, recorded before
the run. That would add ranking noise across the bulk of the population while
leaving genuinely odd regions strongly separated, which is the shape of what we
see. Checking it means looking at where crops actually land on real reals, not
at another aggregate.

### What it says about pooling

Crop embeddings are mean-pooled into a single cached row per image, so a
localized signal is averaged against authentic content before the head ever sees
it. That was flagged before this work started, and deferred: the plan was that if
better placement plateaued, we would have earned evidence that pooling is the
binding constraint.

Placement did not plateau. It *regressed* — which is stronger evidence than a
plateau, and from both cues at once. Pointing crops at a region whose evidence is
then averaged away appears to cost more than it gains.

That is now an empirical argument for max or top-k pooling over crop scores
rather than a speculative one. It is also an architectural change — it alters the
cached row format and invalidates every existing cache — so it needs a spec and a
controlled run, not an afternoon.

### Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_crop_modes.sbatch
```

Read the per-generator table before the pooled AUC: a pooled number hides which
half of the tampered/synthetic split moved.

### A reproducibility result that came free

The `texture` arm reproduces the published baseline **exactly** — 0.9025 pooled,
0.3362 TPR@1%FPR, 0.8513 tampered, 0.9537 full-synthetic, 0.6457 LOGO mean, and
all fifteen rungs to four decimals — despite running on a different GPU
architecture, a different CUDA version, a different torch build, and `--workers 4`
against the baseline's single-threaded path.

That was not the point of the run, and it is worth more than it cost. The claim
that caches are byte-identical whatever `--workers` is set to was previously
pinned only by unit tests on small fixtures; it now holds across a full
24,000-view extraction on different hardware. It also means the `anomaly` numbers
above can be compared directly against the published baseline, not merely against
the arm beside them.
