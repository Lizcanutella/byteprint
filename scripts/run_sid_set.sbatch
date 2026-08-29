#!/bin/bash
# The first real BYTEPRINT run: SID_Set -> probe -> the official §5.2 ladder.
#
# Every path is an environment variable, so this file names no host, login or
# absolute directory and can live in the public repo. Submit it from the
# project root on the cluster:
#
#     export BYTEPRINT_ROOT=<your compute directory>
#     cd "$BYTEPRINT_ROOT"
#     sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_sid_set.sbatch
#
# `--nodelist` matters: the GPU nodes here do not all carry the same driver, and
# a torch build newer than a node's driver fails at `.to("cuda")` after the data
# is already staged. Pin the node whose driver matches the installed torch --
# which one that is belongs in local notes, not in a public file.
#
# Stages are guarded by their outputs, so resubmitting after a timeout resumes
# rather than repeating -- the embedding cache is keyed by (image, spec, config)
# and skips what it already holds.
#
# Measured budget: extraction runs at roughly 5 images/s (CPU-bound on decode
# and crop selection, not on the backbone), so the sizing below is about five
# GPU-hours end to end.
#SBATCH --job-name=byteprint_sid
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

BYTEPRINT_ROOT="${BYTEPRINT_ROOT:?set this to your compute directory}"
cd "$BYTEPRINT_ROOT"

PY="$BYTEPRINT_ROOT/.venv/bin/python"
SRC="${BYTEPRINT_SRC:-$BYTEPRINT_ROOT/src}"
DATA="$BYTEPRINT_ROOT/data"
RUNS="$BYTEPRINT_ROOT/runs"

# Compute nodes have no internet. Weights are read from the staged cache, and
# any accidental reach for the network fails fast instead of hanging.
export HF_HOME="$DATA/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false

BACKBONE="${BYTEPRINT_BACKBONE:-dinov2_large_hf}"
CROPS="${BYTEPRINT_CROPS:-2}"
CROP_SIZE="${BYTEPRINT_CROP_SIZE:-224}"
AUGMENT="${BYTEPRINT_AUGMENT:-3}"
# Twice as many reals as either fake class, because labels 1 and 2 are both
# AIGC: 8000 real vs 8000 fake, rather than the 1:2 an equal split would give.
TRAIN_PER_CLASS="${BYTEPRINT_TRAIN_PER_CLASS:-0=8000,1=4000,2=4000}"
TEST_PER_CLASS="${BYTEPRINT_TEST_PER_CLASS:-0=800,1=400,2=400}"
BATCH="${BYTEPRINT_BATCH:-32}"

TRAIN_DIR="$DATA/sid_train"
TEST_DIR="$DATA/sid_test"
TRAIN_CACHE="$RUNS/cache/sid_train_$BACKBONE"
LADDER_CACHE="$RUNS/cache/sid_ladder_$BACKBONE"
PROBE="$RUNS/probe_sid_$BACKBONE.joblib"
REPORT="$RUNS/report_sid_$BACKBONE.txt"

mkdir -p "$RUNS/cache" slurm_logs

echo "=== BYTEPRINT SID_Set run ==="
echo "backbone=$BACKBONE crops=$CROPS size=$CROP_SIZE augment=$AUGMENT"
echo "train/class=$TRAIN_PER_CLASS test/class=$TEST_PER_CLASS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

# -- 1. materialise -------------------------------------------------------
# Parquet shards -> real/ + fake/{full_synthetic,tampered}/, every class
# re-encoded to PNG so the container cannot become the classifier.
if [ ! -d "$TRAIN_DIR" ]; then
  echo "--- materialising train split"
  "$PY" "$SRC/scripts/materialize_sid_set.py" "$DATA/SID_Set" "$TRAIN_DIR" \
      --split train --per-class "$TRAIN_PER_CLASS" --seed 0
fi

# The test split comes from SID_Set's own validation shards -- disjoint images,
# and unrelated to the competition's WildFake/COCO demo set, which is never
# touched by training.
if [ ! -d "$TEST_DIR" ]; then
  echo "--- materialising test split"
  "$PY" "$SRC/scripts/materialize_sid_set.py" "$DATA/SID_Set" "$TEST_DIR" \
      --split validation --per-class "$TEST_PER_CLASS" --seed 1
fi

# -- 2. train-side extraction --------------------------------------------
echo "--- extracting train ($AUGMENT laundered views per image)"
date
"$PY" -m byteprint.cli -v extract \
    --data "$TRAIN_DIR" --cache "$TRAIN_CACHE" \
    --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --augment "$AUGMENT" --batch-size "$BATCH" --device cuda

# -- 3. probe -------------------------------------------------------------
echo "--- training probe"
"$PY" -m byteprint.cli train \
    --cache "$TRAIN_CACHE" --out "$PROBE" --target-fpr 0.01

# -- 4. the official ladder ----------------------------------------------
# Fifteen rungs of §5.2, each extracted separately so each can be scored
# separately. This is the expensive stage and the headline result.
echo "--- extracting the official ladder over the test split"
date
"$PY" -m byteprint.cli -v extract \
    --data "$TEST_DIR" --cache "$LADDER_CACHE" \
    --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --ladder official --batch-size "$BATCH" --device cuda

# -- 5. report ------------------------------------------------------------
{
  echo "### BYTEPRINT — SID_Set, backbone=$BACKBONE, crops=$CROPS@$CROP_SIZE, augment=$AUGMENT"
  echo "### train/class=$TRAIN_PER_CLASS  test/class=$TEST_PER_CLASS"
  echo
  echo "## Per-rung robustness (the §5.2 ladder)"
  "$PY" -m byteprint.cli eval --cache "$LADDER_CACHE" --probe "$PROBE" --by-spec
  echo
  echo "## Leave-one-generator-out — does a detector trained on fully synthetic"
  echo "## images transfer to locally tampered ones, and vice versa?"
  "$PY" -m byteprint.cli logo --cache "$TRAIN_CACHE"
} | tee "$REPORT"

echo "=== done ==="
date
echo "report: $REPORT"
