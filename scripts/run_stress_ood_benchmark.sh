#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

resolve_python_exec() {
  if [[ -n "${SHARDSIM_PYTHON:-}" ]]; then
    printf '%s\n' "$SHARDSIM_PYTHON"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}/bin/python"
    return
  fi
  command -v python3 >/dev/null 2>&1 && { command -v python3; return; }
  command -v python >/dev/null 2>&1 && { command -v python; return; }
  echo "Error: no Python interpreter found (set SHARDSIM_PYTHON or activate a venv)" >&2
  exit 1
}

PYTHON_EXEC="$(resolve_python_exec)"

TRAIN_SCENARIOS="${TRAIN_SCENARIOS:-128}"
EVAL_SCENARIOS="${EVAL_SCENARIOS:-32}"
TRAIN_DIR="${TRAIN_DIR:-runs/stress_id_train}"
EVAL_DIR="${EVAL_DIR:-runs/stress_far_ood_eval}"
OUT_DIR="${OUT_DIR:-runs/stress_ood_fullscale}"
MODEL="${MODEL:-models/surrogate_stress_ood.pkl}"
REPORT="${REPORT:-reports/surrogate_stress_ood.png}"

mkdir -p "$OUT_DIR"

"$PYTHON_EXEC" scripts/generate_paired_real_training_data.py \
  --n-scenarios "$TRAIN_SCENARIOS" \
  --complexity stress \
  --distribution id \
  --output-dir "$TRAIN_DIR" \
  --seed 20260410 | tee "$OUT_DIR/generate_train.log"

"$PYTHON_EXEC" scripts/generate_paired_real_training_data.py \
  --n-scenarios "$EVAL_SCENARIOS" \
  --complexity stress \
  --distribution far_ood \
  --output-dir "$EVAL_DIR" \
  --seed 20260411 | tee "$OUT_DIR/generate_eval.log"

"$PYTHON_EXEC" scripts/train_surrogate.py \
  --grid-size 256 \
  --paired-data-dir "$TRAIN_DIR" \
  --two-stage \
  --active-threshold 1e-8 \
  --clf-rounds 180 \
  --num-rounds 260 \
  --model-output "$MODEL" | tee "$OUT_DIR/train.log"

"$PYTHON_EXEC" scripts/visualize_surrogate.py \
  --model "$MODEL" \
  --grid-size 256 \
  --paired-data-dir "$EVAL_DIR" \
  --n-samples "$EVAL_SCENARIOS" \
  --output "$REPORT" | tee "$OUT_DIR/eval.log"

ls -lh "$MODEL" "${MODEL%.pkl}.json" "$REPORT"