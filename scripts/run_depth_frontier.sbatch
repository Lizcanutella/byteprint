#!/bin/bash
# The depth frontier: does the forensic signal live at the top of SigLIP2?
#
# Every published BYTEPRINT number reads the tower's final pooled output,
# because that is all the shipped adapters return. This run taps eleven depths
# plus the pooler from the *same forward pass* -- `output_hidden_states=True`
# costs almost nothing extra -- so the whole accuracy-versus-depth curve comes
# out of one extraction, and every probe on top of it is a column slice and a
# few seconds of CPU.
#
#     export BYTEPRINT_ROOT=<your compute directory>
#     cd "$BYTEPRINT_ROOT"
#     sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_depth_frontier.sbatch
#
# Deliberately identical to run_sid_set.sbatch in everything but the backbone:
# same splits, same seed, same 2x224 texture crops, same augmentation. That is
# what makes the `pooler` row a control -- it must reproduce the published
# SigLIP2 numbers (AUC 0.9497, TPR@1%FPR 0.5854) to four decimals. If it does
# not, the plumbing is wrong and no other row on the curve means anything.
#
# `--nodelist` matters for the same reason it does in the other jobs: the GPU
# nodes do not all carry the same driver, and a torch build newer than a node's
# driver fails at `.to("cuda")` after the data is already staged.
#SBATCH --job-name=byteprint_depth
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
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

# The backbone lives in a plugin module, never in a shared file, so this branch
# adds no line another branch is also editing. PYTHONPATH already points at the
# source root, which is where byteprint_depth.py sits.
export BYTEPRINT_PLUGINS="${BYTEPRINT_PLUGINS:-byteprint_depth}"

BACKBONE="siglip2_depth_hf"
CROPS="${BYTEPRINT_CROPS:-2}"
CROP_SIZE="${BYTEPRINT_CROP_SIZE:-224}"
AUGMENT="${BYTEPRINT_AUGMENT:-3}"
TRAIN_PER_CLASS="${BYTEPRINT_TRAIN_PER_CLASS:-0=8000,1=4000,2=4000}"
TEST_PER_CLASS="${BYTEPRINT_TEST_PER_CLASS:-0=800,1=400,2=400}"
BATCH="${BYTEPRINT_BATCH:-32}"
WORKERS="${BYTEPRINT_WORKERS:-4}"

TRAIN_DIR="$DATA/sid_train"
TEST_DIR="$DATA/sid_test"
TRAIN_CACHE="$RUNS/cache/sid_train_$BACKBONE"
LADDER_CACHE="$RUNS/cache/sid_ladder_$BACKBONE"
FRONTIER="$RUNS/depth_frontier.md"
COST="$RUNS/depth_cost.md"

mkdir -p "$RUNS/cache" slurm_logs

echo "=== BYTEPRINT depth frontier ==="
echo "backbone=$BACKBONE crops=$CROPS size=$CROP_SIZE augment=$AUGMENT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -m byteprint.cli list | grep -i depth || true
date

# -- 1. materialise -------------------------------------------------------
# Shared with run_sid_set.sbatch and guarded on a marker written only after the
# script returns, so a run killed mid-materialisation is not mistaken for a
# finished one. If the SID_Set run already went through, both are no-ops.
if [ ! -f "$TRAIN_DIR/.materialised" ]; then
  echo "--- materialising train split"
  "$PY" "$SRC/scripts/materialize_sid_set.py" "$DATA/SID_Set" "$TRAIN_DIR" \
      --split train --per-class "$TRAIN_PER_CLASS" --seed 0
  touch "$TRAIN_DIR/.materialised"
fi

if [ ! -f "$TEST_DIR/.materialised" ]; then
  echo "--- materialising test split"
  "$PY" "$SRC/scripts/materialize_sid_set.py" "$DATA/SID_Set" "$TEST_DIR" \
      --split validation --per-class "$TEST_PER_CLASS" --seed 1
  touch "$TEST_DIR/.materialised"
fi

# -- 2. train-side extraction --------------------------------------------
# ~2.7 GB of features: 48,000 views x 12 blocks x 1152. Resumable, keyed by
# (image, spec, config) as always.
echo "--- extracting train ($AUGMENT laundered views per image, 12 blocks each)"
date
"$PY" -m byteprint.cli -v extract \
    --data "$TRAIN_DIR" --cache "$TRAIN_CACHE" \
    --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --augment "$AUGMENT" --batch-size "$BATCH" --device cuda --workers "$WORKERS"

# -- 3. the official ladder ----------------------------------------------
echo "--- extracting the official ladder over the test split"
date
"$PY" -m byteprint.cli -v extract \
    --data "$TEST_DIR" --cache "$LADDER_CACHE" \
    --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
    --ladder official --batch-size "$BATCH" --device cuda --workers "$WORKERS"

# -- 4. the accuracy side, on CPU ----------------------------------------
# One probe per tap. No `byteprint train` on the whole cache here on purpose:
# fitting 13,824 columns in float64 would want ~10 GB of RAM for an arm nobody
# asked for, while this fits one 1,152-column block at a time.
echo "--- fitting a probe per tap"
date
"$PY" "$SRC/scripts/analyze_depth.py" \
    --train-cache "$TRAIN_CACHE" --ladder-cache "$LADDER_CACHE" \
    --logo --out "$FRONTIER"

# -- 5. the cost side -----------------------------------------------------
# Truncate the tower for real and time it, rather than counting FLOPs: at 196
# tokens attention is memory-bound and an analytic count would not know.
echo "--- timing each truncation depth"
date
"$PY" "$SRC/scripts/bench_depth.py" \
    --crop-size "$CROP_SIZE" --batch-size "$BATCH" --out "$COST"

echo "=== done ==="
date
echo "accuracy: $FRONTIER"
echo "cost:     $COST"
