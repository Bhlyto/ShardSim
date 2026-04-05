#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_SCENARIOS="${TRAIN_SCENARIOS:-24}"
EVAL_SCENARIOS="${EVAL_SCENARIOS:-8}"
GRID_X="${GRID_X:-32}"
GRID_Z="${GRID_Z:-12}"
TRAIN_DIR="${TRAIN_DIR:-runs/training_pairs_3d_train}"
EVAL_DIR="${EVAL_DIR:-runs/training_pairs_3d_eval}"
OUT_DIR="${OUT_DIR:-runs/workflow_3d}"
MODEL="${MODEL:-models/surrogate_3d.pkl}"
REPORT="${REPORT:-reports/surrogate_3d_eval.json}"

mkdir -p "$OUT_DIR"

/home/bhlyto/.venv/bin/python scripts/generate_paired_real_training_data.py \
  --n-scenarios "$TRAIN_SCENARIOS" \
  --grid-size "$GRID_X" \
  --grid-z "$GRID_Z" \
  --complexity complex \
  --distribution id \
  --output-dir "$TRAIN_DIR" \
  --seed 20260412 | tee "$OUT_DIR/generate_train.log"

/home/bhlyto/.venv/bin/python scripts/generate_paired_real_training_data.py \
  --n-scenarios "$EVAL_SCENARIOS" \
  --grid-size "$GRID_X" \
  --grid-z "$GRID_Z" \
  --complexity complex \
  --distribution ood \
  --output-dir "$EVAL_DIR" \
  --seed 20260413 | tee "$OUT_DIR/generate_eval.log"

/home/bhlyto/.venv/bin/python scripts/train_surrogate_3d.py \
  --paired-data-dir "$TRAIN_DIR" \
  --model-output "$MODEL" | tee "$OUT_DIR/train.log"

/home/bhlyto/.venv/bin/python scripts/evaluate_surrogate_3d.py \
  --model "$MODEL" \
  --paired-data-dir "$EVAL_DIR" \
  --output "$REPORT" | tee "$OUT_DIR/eval.log"

ls -lh "$MODEL" "${MODEL%.pkl}.json" "$REPORT"