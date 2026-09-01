#!/bin/bash
# Do the depth gain and the crop-count gain compose?
#
# Two runs beat the shipped SigLIP2 baseline on different axes, and neither
# tested the other's:
#
#   docs/results-crop-pooling.md   2 -> 8 crops, reading the pooler:  0.9497 -> 0.9688
#   docs/results-depth-frontier.md layer 12 + pooler, at 2 crops:     0.9497 -> 0.9717
#
# Adding those is not allowed. They could be the same gain found twice -- more
# crops and a mid-depth tap might both be recovering the same texture evidence
# the final layer discards -- in which case the combination buys nothing over
# the better of the two.
#
# This run answers it as a 2x2 on one extraction. Depth taps are free
# (`output_hidden_states` returns every layer from one forward pass) and crop
# count is now a *read-time* knob (`--crop-limit`, plus `texture`'s prefix
# stability), so a single 8-crop depth extraction contains all four cells:
#
#                     2 crops              8 crops
#   pooler            baseline ref         crop axis alone
#   layer 12+pooler   depth axis alone     both
#
# Every cell shares one train split, one seed, one calibration holdout and one
# ladder, so the interaction term is measured rather than inferred across runs.
#
#     export BYTEPRINT_ROOT=<your compute directory>
#     cd "$BYTEPRINT_ROOT"
#     BYTEPRINT_STAGE=ladder sbatch --nodelist="$NODE" src/scripts/run_depth_crops.sbatch
#     BYTEPRINT_STAGE=train  sbatch --nodelist="$NODE" src/scripts/run_depth_crops.sbatch
#     BYTEPRINT_STAGE=analyze ...        # after both extractions land
#
# The two extractions write different caches and share no state, so they are
# submitted as separate jobs and run on separate GPUs. That halves the wall
# clock, which is the only reason this fits the budget it was given.
#
# `--nodelist` matters for the usual reason: the GPU nodes do not all carry the
# same driver, and a torch build newer than a node's driver fails at the first
# `.to("cuda")` after the data is already staged.
#SBATCH --job-name=byteprint_depthcrops
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

BYTEPRINT_ROOT="${BYTEPRINT_ROOT:?set this to your compute directory}"
cd "$BYTEPRINT_ROOT"

PY="${BYTEPRINT_PY:-$BYTEPRINT_ROOT/.venv/bin/python}"
SRC="${BYTEPRINT_SRC:-$BYTEPRINT_ROOT/src}"
DATA="$BYTEPRINT_ROOT/data"
RUNS="$BYTEPRINT_ROOT/runs"

export HF_HOME="$DATA/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false
export BYTEPRINT_PLUGINS="${BYTEPRINT_PLUGINS:-byteprint_depth}"

STAGE="${BYTEPRINT_STAGE:?set to train, ladder or analyze}"
BACKBONE="siglip2_depth_hf"
CROPS="${BYTEPRINT_CROPS:-8}"
CROP_SIZE="${BYTEPRINT_CROP_SIZE:-224}"
AUGMENT="${BYTEPRINT_AUGMENT:-3}"
BATCH="${BYTEPRINT_BATCH:-32}"
WORKERS="${BYTEPRINT_WORKERS:-4}"

# The train side runs on a strided symlink subset of the materialised split.
# Fewer *images* at full augmentation, rather than fewer laundered views per
# image: the probe wants distinct photographs more than it wants copies, and
# the laundering ladder is what the robustness axis is actually about. Set
# BYTEPRINT_TRAIN_DIR=data/sid_train for the full-scale version of this run.
TRAIN_DIR="${BYTEPRINT_TRAIN_DIR:-$DATA/sid_train_half}"
TEST_DIR="$DATA/sid_test"
TRAIN_CACHE="$RUNS/cache/sid_train_depthcrops_$BACKBONE"
LADDER_CACHE="$RUNS/cache/sid_ladder_depthcrops_$BACKBONE"

mkdir -p "$RUNS/cache" slurm_logs

echo "=== BYTEPRINT depth x crops: stage=$STAGE ==="
echo "backbone=$BACKBONE crops=$CROPS size=$CROP_SIZE augment=$AUGMENT"
echo "train_dir=$TRAIN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

case "$STAGE" in

  train)
    # Never materialise here. Both splits predate the marker guard and their
    # markers were written by hand; cache keys include mtime, so a rewrite
    # would silently invalidate every cache on disk, including the published
    # baselines. If the subset is missing, build it with scripts/make_subset.py
    # on the login node -- it is symlinks and takes seconds.
    if [ ! -f "$TRAIN_DIR/.materialised" ]; then
      echo "!!! $TRAIN_DIR has no .materialised marker; refusing to guess" >&2
      exit 1
    fi
    echo "--- extracting train ($AUGMENT laundered views per image, 12 blocks, $CROPS crops)"
    "$PY" -m byteprint.cli -v extract \
        --data "$TRAIN_DIR" --cache "$TRAIN_CACHE" \
        --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
        --augment "$AUGMENT" --batch-size "$BATCH" --device cuda --workers "$WORKERS"
    ;;

  ladder)
    if [ ! -f "$TEST_DIR/.materialised" ]; then
      echo "!!! $TEST_DIR has no .materialised marker; refusing to guess" >&2
      exit 1
    fi
    # Full scale, deliberately. The train side can shrink without breaking the
    # 2x2 -- every cell shares it -- but the ladder is what every published
    # number was measured on, so shrinking it would make this run
    # incomparable to all of them.
    echo "--- extracting the official ladder over the full test split"
    "$PY" -m byteprint.cli -v extract \
        --data "$TEST_DIR" --cache "$LADDER_CACHE" \
        --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
        --ladder official --batch-size "$BATCH" --device cuda --workers "$WORKERS"
    ;;

  analyze)
    # CPU only. One pass per crop count, reading the same two caches; the
    # per-tap loop inside gives the depth axis, so the two passes together are
    # the 2x2 plus every other tap for free.
    for limit in ${BYTEPRINT_CROP_LIMITS:-2 4 8}; do
      echo "--- fitting every tap at --crop-limit $limit"
      "$PY" "$SRC/scripts/analyze_depth.py" \
          --train-cache "$TRAIN_CACHE" --ladder-cache "$LADDER_CACHE" \
          --crop-limit "$limit" --logo \
          --out "$RUNS/depth_crops_${limit}.md"
      date
    done
    ;;

  *)
    echo "unknown stage: $STAGE (want train, ladder or analyze)" >&2
    exit 1
    ;;
esac

echo "=== done ==="
date
