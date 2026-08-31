# The depth frontier — what we expect, written before the run

Every BYTEPRINT number published so far reads one thing: the final pooled output
of the backbone. Not because that was chosen, but because it is all the shipped
adapters return — `PooledHFVision` gives `pooler_output`, `NaflexVision` the
same. The question of *which depth* carries the forensic signal has never been
asked.

This document exists so the answer cannot be written after the fact. The
crop-localisation work is the reason: there, a fixture built from the hypothesis
it was meant to test returned 40/40 and the hypothesis was still false. A
prediction recorded before the run is the cheapest defence against reading a
table backwards.

## The hypothesis

SigLIP2's late blocks are trained to align an image with a caption. That is a
semantic objective, and it has no reason to preserve the low-level statistics of
texture, grain and resampling that a forensic head actually uses. If those
statistics peak somewhere in the middle of the tower, the last third is not
merely redundant for this task — it is discarding signal.

That would make a truncated tower both **smaller and better**, which is the only
version of "lightweight" worth reporting. It also continues a finding this
project already has: the backbone sweep showed the pretraining *objective*
matters more than parameter count, SigLIP2-so400m beating DINOv2-giant at 38% of
its parameters. This asks whether you need all of the objective's depth either.

## The predictions

1. **A mid-depth tap (roughly 40–70% of the tower) matches or beats the final
   layer on pooled AUC.**
2. **Its advantage is largest on the clean and light rungs, and narrows or
   reverses on `noise:0.10` and `jpeg:30`** — if the cue really is
   texture-statistical, the transforms that destroy high-frequency content
   should hurt an early tap more than a semantic one. Every backbone tested so
   far has had `noise:0.10` as its worst rung without exception, so that is
   where a depth effect should be most visible.
3. **`layer k + pooler` beats either alone** only if depth and attention pooling
   carry different information. If the concatenation merely matches the better
   half, they do not, and that is worth saying.

## What would falsify it

**Pooled AUC rising monotonically with depth on every rung.** That result says
the last layer really is the right tap, the shipped adapters were right by
accident, and there is no truncation story. It is reportable either way and
costs the same hour.

## The unglamorous outcome, named in advance

If the best tap lands at layer 23 or 24 of 27, the compute saving is a rounding
error and there is **no lightweight architecture here** — only a small note
about where features peak. That is a likely enough outcome to name now rather
than dress up later. The frontier is worth plotting regardless; a flat curve
with a cliff at the end is a real answer to a real question.

## The control that makes the rest readable

The extraction stores the tower's own `pooler_output` as a twelfth block,
alongside the eleven mean-pooled taps. SigLIP2 pools with an attention head, so
mean-over-patches at the final layer is a *different function* from what every
published run used — without this block there would be nothing to check against.

A probe fitted on the pooler block, with the same splits, seed, crops and
holdout, must reproduce the published SigLIP2 result:

| | expected |
|---|---|
| pooled AUC | 0.9497 |
| TPR@1%FPR | 0.5854 |
| LOGO mean | 0.7208 |

If it does not, the plumbing is wrong and no other row on the curve means
anything. This is the same control that made the crop-modes run worth more than
it cost, where the `texture` arm reproduced the baseline to four decimals across
different hardware.

## Why it is cheap

`output_hidden_states=True` returns every layer from the **same forward pass**,
so tapping eleven depths costs what tapping one costs. The taps are concatenated
on the feature axis, which keeps the `(n, 3, h, w) -> (n, dim)` backbone
contract intact — no change to extraction, the cache or the CLI, and the whole
thing is a plugin module rather than an edit to a shared file.

Reading one depth back out is then a column slice. `EmbeddingStore` averages an
image's crops into a single row before anything downstream sees it, and a mean
over rows commutes with a slice over columns, so the block read back is exactly
what a single-tap extraction would have cached. That identity is pinned by a
test, because the honesty of every row depends on it.

Budget: one GPU, one extraction over the same 16,000 train / 1,600 test images
and the same fifteen §5.2 rungs as the backbone sweep. The eleven probes on top
are CPU-only and take minutes.

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_depth_frontier.sbatch
```

The job writes `runs/depth_frontier.md` (accuracy per tap) and
`runs/depth_cost.md` (parameters and measured throughput per truncation depth).
Read them together — separately, one is a ranking and the other is a
speed table; together they are a frontier.

`bench_depth.py` truncates the tower for real and times it rather than counting
FLOPs, because at 196 tokens attention is memory-bound and an analytic count
would not notice.
