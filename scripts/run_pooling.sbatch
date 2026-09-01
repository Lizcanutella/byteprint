#!/bin/bash
# Is pooling what limits tampered detection?
#
# The crop-mode comparison refuted the placement hypothesis from two independent
# cues at once: `anomaly` (grain statistics) and `ela` (compression response)
# landed within 0.0001 of each other on tampered AUC and both moved it the wrong
# way, 0.8513 -> 0.8317/0.8318. Two instruments converging on the same wrong
# answer says the problem is not which cue was chosen. It points instead at what
# happens to the evidence after it is found: crop embeddings were mean-pooled
# into one cached row per image, so a localised signal was averaged against
# authentic content before the head ever saw it.
#
# This job tests that. Every arm reads ONE extraction -- pooling is now decided
# at train time, so the expensive stage is shared rather than repeated. That is
# why this comparison is cheaper than the three-arm crop-mode run it follows.
#
#   arm 1  mean  / mean       --crop-limit 2   the refactor changed nothing
#   arm 2  mean  / mean                        the internal control at 8 crops
#   arm 3  crop  / mean-score                  the space, reduction held fixed
#   arm 4  crop  / max                         instance-level MIL
#   arm 5  crop  / topk:2                      instance-level, softened
#   arm 6  mean  / max                         inference-only pooling
#   arm 7  mean  / topk:2                      inference-only, softened
#
# ARM 1 IS A GATE, NOT A RESULT. `texture` crop selection is prefix-stable in
# top_k -- it ranks a fixed candidate set and returns its head -- so the first
# two of eight crops are the same pixels a --crops 2 run would have chosen. Arm 1
# must therefore reproduce the published baseline exactly: 0.9025 pooled, 0.3362
# TPR@1%FPR, 0.8513 tampered, 0.9537 full-synthetic. If it does not, the cache
# restructuring changed something it should not have and no other arm on the
# sheet is trustworthy. It runs first for that reason.
#
# Read the per-generator table before the pooled AUC: `tampered` against
# `full_synthetic` is the split this exists to move, and a pooled figure hides
# either half of it.
#
#     export BYTEPRINT_ROOT=<your compute directory>
#     cd "$BYTEPRINT_ROOT"
#     sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_pooling.sbatch
#
# Cost: one extraction of the train split plus one of the 15-rung ladder, at 8
# crops per image, then seven probe fits of a few seconds each. Budget ~2h30m on
# one GPU. Set BYTEPRINT_BACKBONE to run the same arms on another backbone --
# dinov2_large_hf has the exactly-reproduced baseline, siglip2_so400m_hf is what
# we would ship, and they are independent jobs that can share a queue.
#SBATCH --job-name=byteprint_pooling
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

BYTEPRINT_ROOT="${BYTEPRINT_ROOT:?set this to your compute directory}"
cd "$BYTEPRINT_ROOT"

# Overridable because not every GPU here runs the same CUDA build; see the note
# in run_crop_modes.sbatch. Every arm runs inside one job, so whichever
# interpreter is used the comparison stays internally controlled.
PY="${BYTEPRINT_PY:-$BYTEPRINT_ROOT/.venv/bin/python}"
SRC="${BYTEPRINT_SRC:-$BYTEPRINT_ROOT/src}"
DATA="$BYTEPRINT_ROOT/data"
RUNS="$BYTEPRINT_ROOT/runs"

# Compute nodes have no internet. Weights come from the staged cache.
export HF_HOME="$DATA/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false

BACKBONE="${BYTEPRINT_BACKBONE:-dinov2_large_hf}"
# Eight, where the baseline used two. Max and top-k need something to choose
# between: at --crops 2, `topk:2` degenerates into `mean-score` and `max` picks
# the better of two. --crop-limit 2 recovers the two-crop run from this cache,
# so raising it costs one extraction rather than forfeiting the control.
CROPS="${BYTEPRINT_CROPS:-8}"
CROP_SIZE="${BYTEPRINT_CROP_SIZE:-224}"
CROP_MODE="${BYTEPRINT_CROP_MODE:-texture}"
AUGMENT="${BYTEPRINT_AUGMENT:-3}"
BATCH="${BYTEPRINT_BATCH:-32}"
WORKERS="${BYTEPRINT_WORKERS:-4}"

PARALLEL=()
if [ "$WORKERS" -gt 0 ]; then
  PARALLEL=(--workers "$WORKERS")
fi

TRAIN_DIR="$DATA/sid_train"
TEST_DIR="$DATA/sid_test"

for dir in "$TRAIN_DIR" "$TEST_DIR"; do
  [ -d "$dir" ] || { echo "missing split $dir -- run run_sid_set.sbatch first" >&2; exit 1; }
done

mkdir -p "$RUNS/cache" slurm_logs

TRAIN_CACHE="$RUNS/cache/sid_train_pool${CROPS}_$BACKBONE"
LADDER_CACHE="$RUNS/cache/sid_ladder_pool${CROPS}_$BACKBONE"

echo "=== BYTEPRINT pooling comparison ==="
echo "backbone=$BACKBONE crops=$CROPS size=$CROP_SIZE mode=$CROP_MODE augment=$AUGMENT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

# -- the one shared extraction ---------------------------------------------
#
# Both caches are read by every arm below. This is the whole reason the
# comparison is affordable: pooling is a flag over a cache that outlives it.

echo
echo "--- extracting train ($AUGMENT laundered views per image, $CROPS crops each)"
"$PY" -m byteprint.cli -v extract \
    --data "$TRAIN_DIR" --cache "$TRAIN_CACHE" \
    --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --crop-mode "$CROP_MODE" \
    --augment "$AUGMENT" --batch-size "$BATCH" --device cuda "${PARALLEL[@]}"
date

echo
echo "--- extracting the official ladder over the test split"
"$PY" -m byteprint.cli -v extract \
    --data "$TEST_DIR" --cache "$LADDER_CACHE" \
    --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --crop-mode "$CROP_MODE" \
    --ladder official --batch-size "$BATCH" --device cuda "${PARALLEL[@]}"
date

# -- the arms --------------------------------------------------------------
#
# name | --train-pooling | --pooling | --crop-limit ("-" for all)
ARMS=(
  "arm1_reproduction|mean|mean|2"
  "arm2_control8|mean|mean|-"
  "arm3_cropfit_meanscore|crop|mean-score|-"
  "arm4_cropfit_max|crop|max|-"
  "arm5_cropfit_top2|crop|topk:2|-"
  "arm6_meanfit_max|mean|max|-"
  "arm7_meanfit_top2|mean|topk:2|-"
)

for SPEC in "${ARMS[@]}"; do
  IFS='|' read -r NAME TRAIN_POOLING POOLING LIMIT <<< "$SPEC"

  LIMIT_FLAG=()
  [ "$LIMIT" != "-" ] && LIMIT_FLAG=(--crop-limit "$LIMIT")

  PROBE="$RUNS/probe_pool_${NAME}_$BACKBONE.joblib"
  REPORT="$RUNS/report_pool_${NAME}_$BACKBONE.txt"

  echo
  echo "############ $NAME  (fit=$TRAIN_POOLING pool=$POOLING crops=$LIMIT)"
  date

  "$PY" -m byteprint.cli train \
      --cache "$TRAIN_CACHE" --out "$PROBE" --target-fpr 0.01 \
      --train-pooling "$TRAIN_POOLING" --pooling "$POOLING" "${LIMIT_FLAG[@]}"

  {
    echo "### BYTEPRINT — POOLING ARM = $NAME"
    echo "### backbone=$BACKBONE, $CROPS@${CROP_SIZE}px $CROP_MODE crops,"
    echo "### train-pooling=$TRAIN_POOLING pooling=$POOLING crop-limit=$LIMIT"
    echo "### augment=$AUGMENT, workers=$WORKERS"
    if [ "$NAME" = "arm1_reproduction" ]; then
      echo "###"
      echo "### GATE: this arm must reproduce the published baseline exactly --"
      echo "### 0.9025 pooled / 0.3362 TPR@1%FPR / 0.8513 tampered /"
      echo "### 0.9537 full-synthetic. If it does not, nothing below counts."
    fi
    echo "### Read the per-generator table first: tampered vs full_synthetic is"
    echo "### what this comparison exists to move."
    echo
    echo "## Per-rung robustness (the §5.2 ladder)"
    "$PY" -m byteprint.cli eval --cache "$LADDER_CACHE" --probe "$PROBE" --by-spec
    echo
    echo "## Leave-one-generator-out"
    "$PY" -m byteprint.cli logo --cache "$TRAIN_CACHE" \
        --train-pooling "$TRAIN_POOLING" --pooling "$POOLING" "${LIMIT_FLAG[@]}"
  } | tee "$REPORT"
done

echo
echo "=== done ==="
date
for SPEC in "${ARMS[@]}"; do
  IFS='|' read -r NAME _ _ _ <<< "$SPEC"
  echo "report: $RUNS/report_pool_${NAME}_$BACKBONE.txt"
done
