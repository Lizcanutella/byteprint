# The JPEG-95 control — is the detector reading compression history?

The first SID_Set run scored AUC 0.9025 on a corpus with a structural problem:
its reals are 100% JPEG-family and its fully-synthetic images 100% PNG.
Materialisation re-encodes every class to PNG so the *container* cannot be the
classifier, but JPEG quantisation survives into the reals' pixels and is absent
from the synthetics. That made 0.9025 an upper bound rather than a result.

This is the control. **The answer is that compression history is not the
signal: 0.9022 against the baseline's 0.9025.**

## What was held fixed

Both splits were re-encoded through JPEG-95 (`subsampling=0`) and the pipeline
rerun. Everything else is the baseline's:

| | |
|---|---|
| Images | the *same* 16,000 train / 1,600 test files, derived from the baseline splits rather than re-materialised from parquet |
| Backbone | `dinov2_large_hf`, frozen |
| Crops | 2 per image, 224px, `texture`, native resolution |
| Head | logistic regression, calibrated at 1% FPR |
| Augmentation | `--augment 3` |
| Ladder | the same fifteen §5.2 rungs |
| Extraction path | single-threaded — the baseline predates `--workers`, so the control uses the same code path rather than the faster one |
| Compute | one 48 GB GPU, **2h 38m** (train extraction 1h48m, ladder 49m), 0 failures |

Deriving the recompressed splits from the baseline splits — rather than
re-materialising — is what guarantees the two runs see an identical image set.
It also costs seven minutes instead of the materialiser's hour and a half.

Recompression audit, confirming the pass did what it claims:

```
sid_train: re-encoded 16000 images as jpeg:95 (0 failed)
  source containers: real/PNG: 8000, full_synthetic/PNG: 4000, tampered/PNG: 4000
sid_test:  re-encoded 1600 images as jpeg:95 (0 failed)
  source containers: real/PNG: 800, full_synthetic/PNG: 400, tampered/PNG: 400
```

17 GB → 4.6 GB (train) and 464 MB (test).

## The result: nothing moved

| | baseline (PNG) | control (JPEG-95) | Δ |
|---|---|---|---|
| **pooled over the ladder** | **0.9025** | **0.9022** | **−0.0003** |
| TPR@1%FPR | 0.3362 | 0.3435 | +0.0073 |
| TPR@0.1%FPR | 0.1325 | 0.1224 | −0.0101 |
| full synthetic | 0.9537 | 0.9535 | −0.0002 |
| tampered | 0.8513 | 0.8510 | −0.0003 |
| LOGO mean (unseen type) | 0.6457 | 0.6486 | +0.0029 |

Per rung, AUC:

| rung | baseline | control | Δ |
|---|---|---|---|
| `none` | 0.9112 | 0.9097 | −0.0015 |
| `jpeg:90` | 0.9175 | 0.9173 | −0.0002 |
| `jpeg:70` | 0.9183 | 0.9173 | −0.0010 |
| `jpeg:50` | 0.9038 | 0.9035 | −0.0003 |
| `jpeg:30` | 0.8966 | 0.8971 | +0.0005 |
| `blur:0.5` | 0.9176 | 0.9170 | −0.0006 |
| `blur:1.0` | 0.9191 | 0.9185 | −0.0006 |
| `blur:2.0` | 0.8889 | 0.8879 | −0.0010 |
| `scale:0.5` | 0.9084 | 0.9082 | −0.0002 |
| `scale:0.25` | 0.8899 | 0.8886 | −0.0013 |
| `noise:0.02` | 0.9023 | 0.9033 | +0.0010 |
| `noise:0.05` | 0.8897 | 0.8913 | +0.0016 |
| `noise:0.10` | 0.8553 | 0.8567 | +0.0014 |
| `jitter:0.2` | 0.9120 | 0.9118 | −0.0002 |
| `crop:0.8` | 0.9126 | 0.9123 | −0.0003 |

**The largest movement on any rung is 0.0016.** The ladder still spans 0.064
AUC (0.8567 to 0.9185), `noise:0.10` is still the worst rung, `blur:1.0` is
still the best, and clean is still not the best. Every conclusion the baseline
supported, the control supports identically.

## What this does and does not establish

**Established.** The presence of JPEG artifacts was not what the probe was
reading. Had it been, forcing both classes through the same encoder should have
moved the number substantially — and it moved it by 0.0003, with the
per-generator split unchanged to three decimal places. The baseline's 0.9025
can be quoted as a result rather than as a ceiling.

**Not established.** The two classes' compression histories are still not
identical: the reals are now JPEG → JPEG (double-compressed) and the synthetics
PNG → JPEG (single). Double-JPEG is itself detectable, so this does not rule out
*every* compression-based shortcut.

That said, the shape of the result argues against one. If the probe were keyed
on compression, changing the compression from "JPEG vs none" to "double vs
single" should have perturbed *something* — a rung, the class split, the
operating point. Fifteen rungs moved by at most 0.0016 and the per-class AUCs
moved by 0.0003. That is what a feature invariant to the encoder looks like.

Closing the remaining gap would mean matching the reals' original quantisation
tables, which SID_Set does not record and which vary image to image. The
cheaper next control is a different corpus whose reals and fakes share a
capture and encoding pipeline.

## What the control does not fix

Every weakness the baseline exposed survives it unchanged, which is itself worth
stating — none of them were compression artifacts either:

- **The operating point is still mediocre.** TPR@1%FPR 0.3435. At a threshold
  strict enough to wrongly flag 1 authentic image in 100, two thirds of
  AI-generated images still get through.
- **Tampered images are still much harder** (0.8510) than fully synthetic ones
  (0.9535). The texture-ranked crop strategy still has no notion of *where* the
  edited region is.
- **Transfer to an unseen manipulation type is still weak**: LOGO mean 0.6486,
  against ~0.90 in-distribution.

## Reproducing

```bash
export BYTEPRINT_ROOT=<your compute directory>
cd "$BYTEPRINT_ROOT"
sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_jpeg95_control.sbatch
```

The job re-encodes both splits if it has not already, then runs the identical
extract → train → ladder → LOGO sequence against them. `BYTEPRINT_WORKERS`
defaults to `0`, which omits `--workers` entirely so the control matches a
baseline that predates the prefetch pool; set it to `4` only when the baseline
being compared against also used the pool.
