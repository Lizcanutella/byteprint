# The depth frontier — the last half of the backbone was costing us accuracy

Every BYTEPRINT number published before this run reads one thing: the final
pooled output of the backbone. Not because it was chosen, but because it is all
the shipped adapters return. This run taps eleven depths plus that pooled output
from the *same forward pass* and fits a probe on each.

**The final layer is not the best layer, and it is not close.** Pooled AUC peaks
at **layer 12 of 27** (0.9617) and falls to 0.9429 by layer 27. Reading the
tower's own attention-pooled output — what we have shipped all along — gives
0.9497. The last fifteen blocks are not redundant for this task; they are
actively discarding signal.

Two operating points come out of it, and they are different detectors:

- **Cheaper and better.** Layer 9 scores **AUC 0.9612 / TPR@1%FPR 0.6101** at
  **138M carried parameters (0.34x)** and **2.90x the throughput** of the full
  tower. That beats the published 0.9497 / 0.5854 while running three times
  faster on a third of the weights.
- **Best accuracy.** Concatenating layer 12 with the attention-pooled output
  gives **AUC 0.9717 / TPR@1%FPR 0.6922 / TPR@0.1%FPR 0.4409**, against the
  published 0.9497 / 0.5854 / 0.2554. The operating point improves by **18%
  relative at 1% FPR and 73% relative at 0.1% FPR**. This one needs the whole
  tower, so it buys accuracy, not speed.

Predictions were registered in `docs/depth-frontier-prediction.md` before the
job was submitted. One of the three was refuted, and it is the most interesting
thing here.

## The control passed exactly

The extraction stores the tower's own `pooler_output` as a twelfth block
alongside the eleven mean-pooled taps, because SigLIP2 pools with an attention
head and mean-over-patches at the final layer is a *different function*. A
preflight on the real weights confirmed that block is bit-identical to what
`NaflexVision` returns — max absolute delta **0.000e+00** — before any GPU time
was spent.

A probe fitted on it, with the same splits, seed, crops and holdout, reproduces
the published SigLIP2 result on every reported figure:

| | published | this run |
|---|---|---|
| pooled AUC | 0.9497 | **0.9497** |
| TPR@1%FPR | 0.5854 | **0.5854** |
| TPR@0.1%FPR | 0.2554 | **0.2554** |
| full synthetic | 0.9817 | **0.9817** |
| tampered | 0.9176 | **0.9176** |
| LOGO mean | 0.7208 | **0.7208** |

Every other row on the curve is measured against a row that is exactly the
published detector, on the same 24,000 views.

## The frontier — accuracy

16,000 train / 1,600 test images, 15 §5.2 rungs, 24,000 views. Identical to the
backbone sweep in everything but which columns the probe reads.

| tap | depth | AUC | TPR@1%FPR | TPR@0.1%FPR | full synthetic | tampered | worst rung | LOGO |
|---|---|---|---|---|---|---|---|---|
| layer 1 | 4% | 0.8368 | 0.1705 | 0.0254 | 0.8972 | 0.7765 | blur:2.0 0.7134 | 0.5711 |
| layer 3 | 11% | 0.9261 | 0.4261 | 0.1614 | 0.9661 | 0.8860 | blur:2.0 0.7946 | 0.6090 |
| layer 4 | 15% | 0.9409 | 0.4643 | 0.2278 | 0.9758 | 0.9060 | blur:2.0 0.8196 | 0.6129 |
| layer 5 | 19% | 0.9513 | 0.5635 | 0.3334 | 0.9824 | 0.9203 | blur:2.0 0.8433 | 0.6516 |
| layer 7 | 26% | 0.9546 | 0.5823 | 0.2867 | 0.9837 | 0.9256 | blur:2.0 0.8642 | 0.6946 |
| layer 9 | 33% | 0.9612 | 0.6101 | 0.2308 | 0.9844 | 0.9379 | blur:2.0 0.8927 | 0.6495 |
| **layer 12** | **44%** | **0.9617** | 0.6072 | 0.3815 | 0.9822 | 0.9412 | noise:0.10 0.8971 | 0.6398 |
| layer 15 | 56% | 0.9584 | 0.5767 | 0.3343 | 0.9845 | 0.9323 | noise:0.10 0.8998 | 0.6142 |
| layer 19 | 70% | 0.9495 | 0.5873 | 0.3671 | 0.9793 | 0.9197 | noise:0.10 0.8822 | 0.6504 |
| layer 23 | 85% | 0.9468 | 0.5820 | 0.3493 | 0.9787 | 0.9149 | noise:0.10 0.8793 | 0.6862 |
| layer 27 | 100% | 0.9429 | 0.5490 | 0.1969 | 0.9789 | 0.9068 | noise:0.10 0.8665 | 0.6755 |
| pooler *(published)* | 100% | 0.9497 | 0.5854 | 0.2554 | 0.9817 | 0.9176 | noise:0.10 0.8624 | **0.7208** |
| **layer 12 + pooler** | | **0.9717** | **0.6922** | **0.4409** | **0.9883** | **0.9550** | noise:0.10 **0.9140** | 0.7087 |

The curve rises steeply to layer 9, is flat from 9 to 15, and declines from 15
to 27. Layer 5 — **19% of the parameters** — already matches the full model's
published AUC (0.9513 against 0.9497).

## The frontier — cost

Measured, not counted: the tower is actually truncated and timed on the GPU,
because at 196 tokens attention is memory-bound and an analytic FLOP count would
not know. Parameters are what a detector truncated there would carry — patch and
position embeddings plus the surviving blocks — excluding the attention-pooling
head, which a truncated tower does not instantiate.

| tap | carried params | share of full | crops/s | speed-up |
|---|---|---|---|---|
| layer 1 | 16M | 0.04x | 1860.0 | 19.22x |
| layer 3 | 47M | 0.11x | 765.9 | 7.92x |
| layer 4 | 62M | 0.15x | 593.0 | 6.13x |
| **layer 5** | **77M** | **0.19x** | **481.9** | **4.98x** |
| layer 7 | 108M | 0.26x | 355.7 | 3.68x |
| **layer 9** | **138M** | **0.34x** | **280.5** | **2.90x** |
| layer 12 | 184M | 0.45x | 212.8 | 2.20x |
| layer 15 | 230M | 0.56x | 171.5 | 1.77x |
| layer 19 | 291M | 0.70x | 135.4 | 1.40x |
| layer 23 | 352M | 0.85x | 112.8 | 1.17x |
| layer 27 | 413M | 1.00x | 96.8 | 1.00x |

Read the two tables together. **Layer 5 matches the published AUC at 19% of the
parameters and 5x the throughput. Layer 9 beats it at 34% and 2.9x.** This
continues the arc the backbone sweep started — DINOv2-giant 1.14B, then
SigLIP2-so400m 0.43B, now an effective 0.14B — three steps down the parameter
axis, each one measured, none of them costing accuracy.

## The predictions, scored

**1. A mid-depth tap (40–70%) matches or beats the final layer. CONFIRMED.**
The peak is layer 12, at 44% of depth, +0.0120 AUC over the shipped pooler and
+0.0188 over the mean-pooled final layer.

**2. The mid-tap advantage is largest on clean and light rungs, narrowing or
reversing on `noise:0.10` and `jpeg:30`. REFUTED — it is the exact opposite.**
Layer 12 against the pooler, per rung:

| rung | pooler | layer 12 | gap |
|---|---|---|---|
| `none` | 0.9778 | 0.9816 | +0.0038 |
| `blur:2.0` | 0.9077 | 0.9139 | +0.0062 |
| `scale:0.25` | 0.9065 | 0.9218 | +0.0153 |
| `jpeg:30` | 0.9415 | 0.9617 | **+0.0202** |
| `noise:0.10` | 0.8624 | 0.8971 | **+0.0347** |
| `noise:0.05` | 0.9125 | 0.9521 | **+0.0396** |

The reasoning behind the prediction was that a texture-statistical cue should
die first under the transforms that destroy high-frequency content. The gap is
*smallest* on clean and largest on exactly the two rungs named as the falsifier.
Mid-depth features are not more fragile under laundering — they are markedly
more robust, and robustness is the graded axis.

**3. `layer k + pooler` beats either alone. CONFIRMED**, and the size of the
gain needed a control of its own — see below.

**The unglamorous outcome named in advance did not happen.** The prediction doc
allowed for the best tap landing at layer 23 or 24, leaving no lightweight story.
It landed at 12.

## Where the mid-depth advantage comes from

The per-rung table carries a mechanism that was not predicted and is worth more
than the headline. **The worst rung changes with depth.** Every tap up to layer 9
fails worst on `blur:2.0`; every tap from layer 12 on fails worst on
`noise:0.10`. The crossover sits exactly where the curve peaks.

That is what an optimum made of two competing failure modes looks like. Shallow
features are high-frequency, so blur — which removes high frequencies — destroys
them (layer 1 collapses to 0.7134 on `blur:2.0`). Deep features are semantic and
comparatively blur-tolerant, but noise perturbs them (layer 27 falls to 0.8665
on `noise:0.10`). Layer 12 is where neither failure has taken hold.

This is a mechanism the aggregate AUC hides completely, and it predicts something
checkable: a detector expecting heavy blur should tap deeper than one expecting
heavy sensor noise. Nobody has tested that.

## The control: is the fusion gain just more columns?

`layer 12 + pooler` is 2,304 columns against 1,152, so some of its gain could be
capacity rather than depth diversity. The control holds the depth fixed and
varies only the pooling — `layer 27 + pooler`, also 2,304 columns, both halves
from the same depth:

| | AUC | TPR@1%FPR | TPR@0.1%FPR |
|---|---|---|---|
| pooler alone (1,152 cols) | 0.9497 | 0.5854 | 0.2554 |
| layer 27 + pooler (2,304 cols, same depth) | 0.9553 | 0.6203 | 0.3126 |
| layer 12 + pooler (2,304 cols, two depths) | **0.9717** | **0.6922** | **0.4409** |

Doubling the columns at one depth is worth **+0.0056 AUC**. Taking the second
block from a different depth is worth **+0.0164** on top — three times as much.
On the operating point the split is +0.0349 against +0.0719. So roughly a third
of the fusion gain is capacity and two thirds is genuine depth diversity. Both
are real; only the second is a finding.

## The honest weaknesses

**Transfer did not improve, and the shipped detector is still the best at it.**
The pooler's LOGO mean of 0.7208 is the highest number in that column. Layer 12
scores 0.6398 and the best fusion 0.7087 — slightly *worse* than doing nothing.
Unseen-manipulation-type transfer has been this project's weakest axis
throughout, and the depth frontier does not move it. If anything, the attention
pooler's advantage looks specifically like a transfer advantage: it is mid-table
on in-distribution AUC and top on LOGO. That trade-off is unexplained and should
not be glossed.

**The accuracy winner buys no speed.** `layer 12 + pooler` needs the pooled
output, which exists only at full depth, so it runs the whole tower. The
lightweight result (layer 9, layer 5) and the accurate result are different
points on the frontier, not the same one. Anyone quoting 0.9717 and "0.34x
parameters" in the same sentence is quoting two different detectors.

**LOGO measures manipulation type, not generator.** Unchanged from the first run:
SID_Set gives fully-synthetic and tampered, which are two kinds of editing rather
than two diffusion models. It should not be quoted as generator generalisation.

**One seed, one split, one corpus.** No error bars. The depth effect (+0.012 to
+0.019 AUC, +0.107 TPR@1%FPR) is far outside plausible seed variance, but the
distinction between adjacent taps — layer 9 at 0.9612 against layer 12 at 0.9617
— is not, and the two should be treated as tied.

**Crops are still mean-pooled before the head sees them**, so this run says
nothing about the pooling question the crop-mode work raised. That remains open
and still needs a cache-format change.

## Cost

One 48 GB GPU on a SLURM cluster, **1h13m wall clock**, 0 failures over 72,000 views.

| stage | time |
|---|---|
| train extraction (48,000 views) | 42m45s |
| ladder extraction (24,000 views) | 20m50s |
| twelve probes + LOGO, CPU only | 8m52s |
| truncation benchmark | 31s |

The train extraction is the number to notice: the published single-tap SigLIP2
run took **42m** for the same 48,000 views, and this took **42m45s** for eleven
extra depths. Tapping every layer really is free, because `output_hidden_states`
returns them all from one forward pass. The cache is larger — 2.65 GB against
221 MB — and that is the whole price.

Because the taps are stored, every question asked here after the fact was a
column slice and a few CPU-minutes. The same-depth control above cost nothing but
the CPU time to fit two more probes.

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
sbatch --nodelist="$BYTEPRINT_GPU_NODE" \
    --export=ALL,BYTEPRINT_SRC="$BYTEPRINT_ROOT"/src-depth \
    src/scripts/run_depth_frontier.sbatch
```

The backbone is a plugin (`byteprint_depth.py`, `BYTEPRINT_PLUGINS`), not an
edit to a shared module: it concatenates the taps on the feature axis, so the
`(n, 3, h, w) -> (n, dim)` contract holds and extraction, the cache and the CLI
are untouched.

Reading one depth back out relies on mean-over-crops commuting with a column
slice — `EmbeddingStore` averages an image's crops into a single row before
anything downstream sees it — so the block read back is exactly what a
single-tap extraction would have cached. That identity is pinned by a test,
because the honesty of every row above depends on it.
