# Crop modes that know where to look — the fixture measurement

The SID_Set run leaves a specific gap: fully-synthetic images score AUC 0.9537,
tampered ones 0.8513. The hypothesis was the crop strategy. `texture` keeps the
windows richest in high-frequency detail, and in a tampered image — a real
photograph with an AI-edited region — the edited region has no reason to be one
of them.

This documents what two new crop modes do about that **on a planted fixture**.
It is a mechanism result, not a detection result: no AUC appears below, because
none has been measured yet. `scripts/run_crop_modes.sbatch` is the job that
answers the real question and has not been run.

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

**Established.** `texture` actively sorts a cleaner-than-its-surroundings region
last, which is a concrete mechanism for the tampered gap. A within-image outlier
ranking finds such a region reliably, degrades to the existing behaviour instead
of to noise when its cue dies, and costs under 2× the strategy it replaces.

**Not established.** Anything about AUC. The fixture plants one kind of edit,
chosen to match the hypothesis, at a known size, in images whose grain is
synthetic and uniform. Real inpainted regions vary in size, blend at their edges,
and sit in photographs whose grain is neither uniform nor Gaussian. A cue that
scores 40/40 here could be worth nothing on SID_Set.

The three specific ways this could fail to transfer, worth checking against the
run rather than arguing about now: edited regions smaller than a 224px crop, so
no window is mostly-edit and the contrast is diluted within every candidate;
photographs whose legitimate content already varies enough in grain — sky against
foliage — to trigger the ranking on an unedited image; and `band_map` reading
luma only, which drops chroma inconsistency, a genuine forensic cue that the
proxy is blind to and so could not measure the loss of.

## Running the real comparison

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_crop_modes.sbatch
```

Three arms — `texture`, `anomaly`, `ela` — identical in every other respect,
roughly 4h on one GPU. `texture` is re-run rather than quoted from the first
SID_Set run so that all three share a code version and an extraction path.

Read the per-generator table before the pooled AUC. The mode is meant to lift
`tampered` without costing anything on `full_synthetic`, and a pooled number
hides either half of that.
