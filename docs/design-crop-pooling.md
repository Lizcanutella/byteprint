# Pooling crop evidence: moving the reduction out of the cache

## The problem, stated precisely

`EmbeddingStore.add` ends with:

```python
self._features.append(crop_features.mean(axis=0))
```

An image's crops are embedded, averaged, and the individual crop embeddings are
discarded. That single line is the subject of this document, and the thing worth
naming about it is not that the mean is the wrong reduction. It is that the
reduction happens **at cache-write time**, which makes pooling the one part of
this pipeline that is not a hyperparameter. Every other swappable part —
backbone, head, crop mode — is a flag over a cache that outlives it. Pooling is
baked into the most expensive artifact we own, so asking "is max better than
mean?" currently costs a full re-extraction per answer.

So the fix is not "swap `mean` for `max`". It is **move the reduction from write
time to train/eval time**, after which mean, max and top-k are a sweep over one
cached extraction.

## Why now

`docs/results-crop-localisation.md` closes on this, and it earns the claim from
an unusually strong piece of evidence. Two crop-placement modes reading
*different physical cues* — grain statistics (`anomaly`) and compression response
(`ela`) — were pointed at the edited region of tampered images. Both landed
within 0.0001 of each other on tampered AUC, and both moved it the **wrong way**,
0.8513 → 0.8317 / 0.8318.

Two independent instruments converging on the same wrong answer is much better
evidence than one failing. It says the problem is not which cue was chosen. And
it makes a specific mechanism the leading suspect: pointing crops at a region
whose evidence is then *averaged against crops of authentic content* costs more
than the better placement gains. A regression is stronger evidence than the
plateau the earlier plan waited for, because a plateau is consistent with "the
cue is weak" while a regression is what you predict if better-placed evidence is
being actively diluted.

This is also why the tampered/full-synthetic gap is the right place to look. A
fully-synthetic image is synthetic in every crop, so mean-pooling loses nothing.
A tampered image is a real photograph with one edited region — the signal is
localised by construction, and averaging is exactly the wrong operator for it.
The gap (0.9537 vs 0.8513) is the shape that hypothesis predicts.

## Design

### 1. Cache schema 2

`features.npy` becomes `(total_crops, dim)`. Each `records.jsonl` line gains
`n_crops`, so an image's rows are `features[offset : offset + n_crops]` where the
offsets are a cumulative sum over the records. No second index file: row order
keeps one source of truth, and the store stays append-only and resumable.

Counts are stored per record rather than assumed uniform because they already are
not — `center` and `resize` return one crop whatever `--crops` says, and an image
smaller than the crop size can yield fewer. Reading a ragged bag correctly is
cheaper to build now than to debug later.

`config.json` gains `"schema": 2`. A cache without it is schema 1 and raises
`StaleCacheError` naming the fix:

```
cache at runs/cache/train stores one pooled row per image (schema 1);
pooling needs per-crop rows (schema 2). Re-run `byteprint extract`,
or pass --rebuild to discard it.
```

`--rebuild` discards a schema-1 cache the same way it discards a
settings-mismatched one. There is deliberately no read path for schema 1: the
crop rows it would need were never written, so a compatibility branch could only
support the one pooling that already works, at the cost of a class of bug where
an arm silently cannot run.

**`store.matrix()` keeps its exact present meaning** — one mean-pooled row per
image — computed on read instead of on write. `fusion`, `logo` and `eval` are
untouched. New alongside it: `crop_matrix()`, `crop_counts()`.

### 2. A pooling registry

`byteprint/pooling.py`, following the repo's convention that swappable parts are
registries rather than edits to a shared file:

| name | space | reduction |
|---|---|---|
| `mean` | feature | mean of crop embeddings, then one head call — today's path exactly |
| `mean-score` | score | head per crop, mean of the probabilities |
| `max` | score | head per crop, max of the probabilities |
| `topk:k` | score | head per crop, mean of the k highest probabilities |

`topk:k` uses the `name:arg` form already used by ladder specs (`jpeg:90`), so
the convention is borrowed rather than invented.

`mean` and `mean-score` are not the same operator and the difference is the point
of having both: one averages in feature space and calls the head once, the other
calls the head per crop and averages probabilities. A logistic head is monotone
but not affine in its input, so these genuinely differ, and keeping both is what
lets the run separate *the space the reduction happens in* from *the reduction
itself*.

### 3. Two knobs, because they are a cross product

- **`--train-pooling {mean, crop}`** — how bags become training rows. `mean` is
  today's fit, on one pooled row per image. `crop` fits on one row per crop, with
  the image's label inherited by each.
- **`--pooling {mean, mean-score, max, topk:k}`** — how per-image scores are
  formed at evaluation.

These are separate because the interesting arms are the off-diagonal ones.
`--train-pooling crop --pooling max` is instance-level multiple-instance
learning. `--train-pooling mean --pooling max` changes *only* inference, leaving
the fit byte-identical to today's. Running both is what distinguishes "crop-level
training helped" from "pooling at inference helped", and neither answer is
predictable in advance:

- **Instance-level training carries real label noise.** A tampered image's crops
  are mostly authentic content, and every one of them is labelled synthetic. That
  is a mislabelled positive rate that could be most of the bag — in exactly the
  class this work exists to fix.
- **Bag-level training carries a distribution mismatch.** A head fit on averages
  of several crops is applied to single crops, whose feature distribution has
  visibly larger variance.

One of these costs is probably smaller than the other. Guessing which is not
cheaper than measuring it, because both fits take seconds over a shared cache.

Both knobs travel inside the saved probe, alongside `extract_config`. This
extends the guard that already exists rather than adding a new one: a probe
trained with max pooling scores with max pooling, and `score` cannot silently
disagree with `train`.

### 4. `--crop-limit N`, and the control it buys for nothing

Truncates every bag to its first `N` crops at train and eval time.

This exists because of a property worth stating as a measurement rather than an
assumption: **`texture` crop selection is prefix-stable in `top_k`.** It draws
`max(candidates, top_k)` = 32 candidate windows from a seeded generator, sorts
them all by Laplacian variance, and returns the first `top_k`. The candidate set
and its ordering therefore do not depend on `top_k` at all, so the top 2 of an
8-crop selection *are* the 2 crops a `--crops 2` run would have chosen — the same
pixels, not merely a similar sample. Verified for `texture`, `anomaly`, `ela` and
`center`; **not** true of `random`, which redraws its origins per `top_k`, and a
test pins both halves of that so the property cannot rot silently.

The consequence: extract once at `--crops 8`, and `--crop-limit 2` reproduces the
published two-crop run exactly. The refactor's correctness control costs nothing
and needs no second extraction, and a crop-count ablation nobody budgeted for
falls out of the same cache.

## The run

One extraction per split, shared by every arm. Each arm is a logistic regression
fit over cached features: seconds, not hours.

| arm | `--train-pooling` | `--pooling` | `--crop-limit` | what it establishes |
|---|---|---|---|---|
| 1 | `mean` | `mean` | 2 | **the refactor changed nothing** |
| 2 | `mean` | `mean` | — | the internal control at 8 crops |
| 3 | `crop` | `mean-score` | — | the space, holding the reduction fixed |
| 4 | `crop` | `max` | — | instance-level MIL |
| 5 | `crop` | `topk:2` | — | instance-level, softened |
| 6 | `mean` | `max` | — | inference-only pooling |
| 7 | `mean` | `topk:2` | — | inference-only, softened |

Arms 2/3/4 are the 2×2 that separates the space from the reduction; 4 against 6
and 5 against 7 separate the training scheme from the inference scheme.

**Arm 1 is a gate, not a result.** It must reproduce 0.9025 pooled / 0.3362
TPR@1%FPR / 0.8513 tampered / 0.9537 full-synthetic, to four decimals. If it does
not, the restructuring changed something it should not have and no other arm on
the sheet is trustworthy. The job prints it first for that reason.

Run on `dinov2_large_hf`, because it is the only configuration whose published
baseline has already been reproduced exactly and is where the tampered gap was
measured. Concurrently on a second GPU, the identical arms on
`siglip2_so400m_hf` — the backbone we would actually ship, per
`docs/results-backbone-sweep.md`. The second job costs a GPU slot and no wall
clock.

**Budget: ~2h30m on 2 GPUs of a SLURM cluster.** Cheaper than the 3h45m
crop-mode comparison it follows, because pooling arms share one extraction where
crop modes could not.

### The prediction, written down before the run

Tampered AUC rises under `max` and `topk:2`; full-synthetic stays roughly flat,
having little localised evidence to recover. Pooled AUC rises with the tampered
half.

**What would refute it.** Tampered AUC flat or down across all four score-space
arms. That is a live possibility and it is not a dead end: it would mean neither
crop *placement* nor crop *pooling* is the binding constraint, and the closing
claim of `docs/results-crop-localisation.md` would need amending rather than
confirming. Recording that here is the point — the crop-mode fixture failed
precisely because it was built from the hypothesis it was used to test, and could
not have come out any way but one.

A second, quieter thing to watch: `--crop-limit 8` against `--crop-limit 2` under
`mean`/`mean` is a free crop-count ablation. If most of the movement in the
max-pooled arms is also present there, the story is "more crops help", not
"pooling helps", and the arms must be read against arm 2 rather than arm 1.

## Deliberately out of scope

`fusion.py` joins two caches on `matrix()`. Fusing a max-pooled probe against
mean-pooled features would score it on a distribution it was not trained for, and
would do so silently. Rather than paper over that, `fuse` raises unless the
probe's pooling is `mean`, with a message saying so. Wiring fusion through
bag-level scoring is a follow-up, and not an urgent one: the reconstruction
expert is chance-level on tampered images (`docs/results-recon-fusion.md`), which
is the half of the data this work is about.

Feature-space `max` — a per-dimension maximum over crops — is not registered. It
is a defensible operator, it is not what the evidence argues for, and an
unmotivated arm on a comparison sheet is a cost.

## Test plan

- **Cache.** Schema-2 round-trip preserves per-crop rows; ragged counts survive;
  `matrix()` still returns per-image means; a schema-1 cache raises with the
  message above; `--rebuild` discards it; identical caches under any `--workers`.
- **Pooling.** Registry resolves `topk:2` through the `name:arg` split; segment
  reductions are correct on ragged counts; `mean` equals the previous
  `crop_features.mean(axis=0)` on the same input, pinned against a golden array.
- **Prefix stability.** `select_crops(top_k=8)[:2] == select_crops(top_k=2)` for
  `texture`, and explicitly *not* for `random`.
- **Probe.** `crop` training repeats labels to match crop rows; `score_bags`
  returns one score per image; pooling, train-pooling and crop-limit survive
  save/load.
- **Deliverable.** `ProbeScorer` and `predict` honour the probe's pooling, so a
  max-pooled probe produces max-pooled predictions through the required JSON
  interface.
