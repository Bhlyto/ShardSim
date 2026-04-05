#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cmake --build build -j >/dev/null

if [[ ! -f models/surrogate_linear_policy.txt ]]; then
  /home/bhlyto/.venv/bin/python scripts/train_linear_policy.py \
    --paired-data-dir runs/hardened_id_train \
    --model-output models/surrogate_linear_policy.txt >/dev/null
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/policy_reference_benchmark/$TIMESTAMP"
mkdir -p "$OUT_DIR/exports"

SOURCE_X="${SOURCE_X:-0.31}"
SOURCE_Y="${SOURCE_Y:-0.74}"
SOURCE_TEMP="${SOURCE_TEMP:-210.0}"
ALPHA="${ALPHA:-1.35}"
DT="${DT:-0.00115}"
SOURCE2_ENABLED="${SOURCE2_ENABLED:-true}"
SOURCE2_X="${SOURCE2_X:-0.79}"
SOURCE2_Y="${SOURCE2_Y:-0.22}"
SOURCE2_TEMP="${SOURCE2_TEMP:-120.0}"

write_cfg() {
  local out="$1"
  local policy="$2"
  local min_fraction="$3"
  local export_subdir="$4"
  local model_path="${5:-}"
  cat > "$out" <<EOF
grid_x: 256
grid_y: 256
grid_z: 1
steps: 200
dt: $DT
alpha: $ALPHA
coarse_tolerance: 0.05
fine_tolerance: 0.03
refine_local_error_tau: 0.05
refine_uncertainty_tau: 0.20
min_critical_fraction: $min_fraction
memory_ceiling_gb: 16
wallclock_limit_minutes: 0
halo_overhead_ratio_max: 0.95
deterministic_mode: true
partitioning_policy: strict_geometric
decision_policy: $policy
export_training_data: true
training_data_export_dir: $OUT_DIR/exports/$export_subdir
source_x_fraction: $SOURCE_X
source_y_fraction: $SOURCE_Y
source_temperature: $SOURCE_TEMP
source2_enabled: $SOURCE2_ENABLED
source2_x_fraction: $SOURCE2_X
source2_y_fraction: $SOURCE2_Y
source2_temperature: $SOURCE2_TEMP
EOF
  if [[ -n "$model_path" ]]; then
    cat >> "$out" <<EOF
surrogate_model_path: $model_path
surrogate_python_executable: /home/bhlyto/.venv/bin/python
surrogate_script_path: scripts/predict_critical_mask.py
surrogate_temp_dir: runs/surrogate_policy_tmp
surrogate_score_threshold: 0.0
surrogate_top_fraction: 0.02
EOF
  fi
}

write_cfg "$OUT_DIR/baseline_a.yaml" heuristic 1.0 baseline_a
write_cfg "$OUT_DIR/baseline_b.yaml" heuristic 0.0 baseline_b
write_cfg "$OUT_DIR/baseline_c_python.yaml" surrogate_python 0.0 baseline_c_python models/surrogate_hardened_fullscale.pkl
write_cfg "$OUT_DIR/baseline_c_linear.yaml" surrogate_linear 0.0 baseline_c_linear models/surrogate_linear_policy.txt

CSV="$OUT_DIR/summary.csv"
echo "baseline,steps,coarse_steps,fine_steps,critical_cells,critical_fraction,decision_ms,runtime_ms,mae,global_error_norm" > "$CSV"

run_one() {
  local name="$1"
  local config="$2"
  local log="$OUT_DIR/${name}.log"

  ./build/shardsim_cli "$config" | tee "$log"

  extract() {
    local key="$1"
    awk -v k="$key" '$1==k":" {print $2}' "$log" | tail -n 1
  }

  echo "$name,$(extract steps),$(extract coarse_steps),$(extract fine_steps),$(extract critical_cells),$(extract critical_fraction),$(extract decision_ms),$(extract runtime_ms),$(extract mae),$(extract global_error_norm)" >> "$CSV"
}

run_one baseline_a "$OUT_DIR/baseline_a.yaml"
run_one baseline_b "$OUT_DIR/baseline_b.yaml"
run_one baseline_c_python "$OUT_DIR/baseline_c_python.yaml"
run_one baseline_c_linear "$OUT_DIR/baseline_c_linear.yaml"

latest_bin() {
  local dir="$1"
  ls -1t "$dir"/*.bin | head -n 1
}

REF_FILE="$(latest_bin "$OUT_DIR/exports/baseline_a")"
/home/bhlyto/.venv/bin/python scripts/compare_policy_exports.py \
  --reference "$REF_FILE" \
  --candidate baseline_b="$(latest_bin "$OUT_DIR/exports/baseline_b")" \
  --candidate baseline_c_python="$(latest_bin "$OUT_DIR/exports/baseline_c_python")" \
  --candidate baseline_c_linear="$(latest_bin "$OUT_DIR/exports/baseline_c_linear")" \
  --output "$OUT_DIR/reference_compare.csv"

echo "summary: $CSV"
echo "reference_compare: $OUT_DIR/reference_compare.csv"