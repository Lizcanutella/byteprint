#!/bin/bash
# Does looking in the right place close the tampered gap?
#
# The SID_Set baseline scores AUC 0.9537 on fully-synthetic images and 0.8513 on
# tampered ones. The stated hypothesis for that gap is the crop strategy:
# `texture` keeps the windows richest in high-frequency detail, and in a tampered
# image the edited region has no reason to be one of them. A region produced by a
# decoder is usually *cleaner* than the captured frame around it, so the
# heuristic does not merely miss it -- on the planted-edit fixture it ranks the
# edited region last, 0 hits out of 40 against `anomaly`'s 40.
#
# This job tests whether that carries over to real tampered images, which the
# fixture cannot answer. It runs the identical pipeline once per crop mode:
#
#   texture   the baseline, re-run here rather than quoted from the earlier run,
#             so all three arms share a code version and an extraction path
#   anomaly   rank windows by distance from the image's own median window
#   ela       the absolute-cue control: rank by response to a fresh JPEG encode
#
# The number to read is not the pooled AUC. It is the per-generator table, where
# `full_synthetic` and `tampered` appear separately -- the mode is meant to lift
# `tampered` without costing anything on `full_synthetic`, and a pooled figure
# would hide either half of that.
#
#     export BYTEPRINT_ROOT=<your compute directory>
#     cd "$BYTEPRINT_ROOT"
#     sbatch --nodelist="$BYTEPRINT_GPU_NODE" src/scripts/run_crop_modes.sbatch
#
# Cost: three arms at roughly 1h20m each with --workers 4, so budget ~4h on one
# GPU. Set BYTEPRINT_CROP_MODES="anomaly" to run a single arm against the
# already-published texture numbers instead -- cheaper, but then the comparison
# crosses a code version and an extraction path, which is exactly the kind of
# uncontrolled difference the JPEG-95 control was written to avoid.
#SBATCH --job-name=byteprint_crop_modes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/%x_%j.out

set -euo pipefail

BYTEPRINT_ROOT="${BYTEPRINT_ROOT:?set this to your compute directory}"
cd "$BYTEPRINT_ROOT"

PY="$BYTEPRINT_ROOT/.venv/bin/python"
SRC="${BYTEPRINT_SRC:-$BYTEPRINT_ROOT/src}"
DATA="$BYTEPRINT_ROOT/data"
RUNS="$BYTEPRINT_ROOT/runs"

# Compute nodes have no internet. Weights come from the staged cache.
export HF_HOME="$DATA/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$SRC"
export TOKENIZERS_PARALLELISM=false

# `anomaly` and `ela` are built in -- byteprint.crops imports byteprint.localize
# for its registrations -- so no --plugin is needed here.
CROP_MODES="${BYTEPRINT_CROP_MODES:-texture anomaly ela}"

# Held identical to the first SID_Set run so that crop mode is the only variable
# that moves between these arms and that run.
BACKBONE="${BYTEPRINT_BACKBONE:-dinov2_large_hf}"
CROPS="${BYTEPRINT_CROPS:-2}"
CROP_SIZE="${BYTEPRINT_CROP_SIZE:-224}"
AUGMENT="${BYTEPRINT_AUGMENT:-3}"
BATCH="${BYTEPRINT_BATCH:-32}"

# Unlike the JPEG-95 control, every arm here is extracted by this same job, so
# they share whatever path this is set to and the comparison stays internally
# controlled. 4 is where the prefetch pool stops paying (see README).
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

echo "=== BYTEPRINT crop-mode comparison ==="
echo "modes=[$CROP_MODES] backbone=$BACKBONE crops=$CROPS size=$CROP_SIZE augment=$AUGMENT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

for MODE in $CROP_MODES; do
  TRAIN_CACHE="$RUNS/cache/sid_train_${MODE}_$BACKBONE"
  LADDER_CACHE="$RUNS/cache/sid_ladder_${MODE}_$BACKBONE"
  PROBE="$RUNS/probe_sid_${MODE}_$BACKBONE.joblib"
  REPORT="$RUNS/report_sid_${MODE}_$BACKBONE.txt"

  echo
  echo "############ crop mode: $MODE"
  date

  echo "--- extracting train ($AUGMENT laundered views per image)"
  "$PY" -m byteprint.cli -v extract \
      --data "$TRAIN_DIR" --cache "$TRAIN_CACHE" \
      --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
      --crop-mode "$MODE" \
      --augment "$AUGMENT" --batch-size "$BATCH" --device cuda "${PARALLEL[@]}"

  echo "--- training probe"
  "$PY" -m byteprint.cli train \
      --cache "$TRAIN_CACHE" --out "$PROBE" --target-fpr 0.01

  echo "--- extracting the official ladder over the test split"
  date
  "$PY" -m byteprint.cli -v extract \
      --data "$TEST_DIR" --cache "$LADDER_CACHE" \
      --backbone "$BACKBONE" --crops "$CROPS" --crop-size "$CROP_SIZE" \
      --crop-mode "$MODE" \
      --ladder official --batch-size "$BATCH" --device cuda "${PARALLEL[@]}"

  {
    echo "### BYTEPRINT — CROP MODE = $MODE"
    echo "### backbone=$BACKBONE, crops=$CROPS@$CROP_SIZE,"
    echo "### augment=$AUGMENT, workers=$WORKERS"
    echo "### Read the per-generator table first: full_synthetic vs tampered is"
    echo "### what this comparison exists to move."
    echo
    echo "## Per-rung robustness (the §5.2 ladder)"
    "$PY" -m byteprint.cli eval --cache "$LADDER_CACHE" --probe "$PROBE" --by-spec
    echo
    echo "## Leave-one-generator-out"
    "$PY" -m byteprint.cli logo --cache "$TRAIN_CACHE"
  } | tee "$REPORT"
done

echo
echo "=== done ==="
date
for MODE in $CROP_MODES; do
  echo "report: $RUNS/report_sid_${MODE}_$BACKBONE.txt"
done
