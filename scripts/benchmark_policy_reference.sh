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

cmake --build build -j >/dev/null

if [[ ! -f models/surrogate_linear_policy.txt ]]; then
  "$PYTHON_EXEC" scripts/train_linear_policy.py \
    --paired-data-dir runs/hardened_id_train \
    --model-output models/surrogate_linear_policy.txt >/dev/null
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/policy_reference_benchmark/$TIMESTAMP"
mkdir -p "$OUT_DIR/exports"

N_SCENARIOS="${N_SCENARIOS:-8}"
SCENARIO_SEED="${SCENARIO_SEED:-20260406}"

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
surrogate_python_executable: $PYTHON_EXEC
surrogate_script_path: scripts/predict_critical_mask.py
surrogate_temp_dir: runs/surrogate_policy_tmp
surrogate_score_threshold: 0.0
surrogate_top_fraction: 0.02
EOF
  fi
}

CSV="$OUT_DIR/summary.csv"
REF_ALL="$OUT_DIR/reference_compare.csv"
echo "scenario_id,baseline,steps,coarse_steps,fine_steps,critical_cells,critical_fraction,decision_ms,runtime_ms,mae,global_error_norm" > "$CSV"
echo "scenario_id,baseline,mae,rmse,r2,global_error_norm" > "$REF_ALL"

scenario_params() {
  local sid="$1"
  "$PYTHON_EXEC" - <<PY
import random
seed = int("$SCENARIO_SEED") + int("$sid")
rng = random.Random(seed)
print(
    f"{rng.uniform(0.10, 0.90):.6f},"
    f"{rng.uniform(0.10, 0.90):.6f},"
    f"{rng.uniform(30.0, 260.0):.6f},"
    f"{rng.uniform(0.7, 1.6):.6f},"
    f"{rng.uniform(0.0005, 0.0017):.6f},"
    f"{rng.uniform(0.10, 0.90):.6f},"
    f"{rng.uniform(0.10, 0.90):.6f},"
    f"{rng.uniform(10.0, 220.0):.6f}",
    sep="",
)
PY
}

run_one() {
  local sid="$1"
  local name="$2"
  local config="$3"
  local log="$4"

  ./build/shardsim_cli "$config" | tee "$log"

  extract() {
    local key="$1"
    awk -v k="$key" '$1==k":" {print $2}' "$log" | tail -n 1
  }

  echo "$sid,$name,$(extract steps),$(extract coarse_steps),$(extract fine_steps),$(extract critical_cells),$(extract critical_fraction),$(extract decision_ms),$(extract runtime_ms),$(extract mae),$(extract global_error_norm)" >> "$CSV"
}

latest_bin() {
  local dir="$1"
  ls -1t "$dir"/*.bin | head -n 1
}

for ((sid=0; sid<N_SCENARIOS; sid++)); do
  sid_fmt="$(printf "%04d" "$sid")"
  scenario_dir="$OUT_DIR/scenario_${sid_fmt}"
  mkdir -p "$scenario_dir"

  IFS=',' read -r SOURCE_X SOURCE_Y SOURCE_TEMP ALPHA DT SOURCE2_X SOURCE2_Y SOURCE2_TEMP < <(scenario_params "$sid")

  write_cfg "$scenario_dir/baseline_a.yaml" heuristic 1.0 "baseline_a/s${sid_fmt}"
  write_cfg "$scenario_dir/baseline_b.yaml" heuristic 0.0 "baseline_b/s${sid_fmt}"
  write_cfg "$scenario_dir/baseline_c_python.yaml" surrogate_python 0.0 "baseline_c_python/s${sid_fmt}" models/surrogate_hardened_fullscale.pkl
  write_cfg "$scenario_dir/baseline_c_linear.yaml" surrogate_linear 0.0 "baseline_c_linear/s${sid_fmt}" models/surrogate_linear_policy.txt

  run_one "$sid_fmt" baseline_a "$scenario_dir/baseline_a.yaml" "$scenario_dir/baseline_a.log"
  run_one "$sid_fmt" baseline_b "$scenario_dir/baseline_b.yaml" "$scenario_dir/baseline_b.log"
  run_one "$sid_fmt" baseline_c_python "$scenario_dir/baseline_c_python.yaml" "$scenario_dir/baseline_c_python.log"
  run_one "$sid_fmt" baseline_c_linear "$scenario_dir/baseline_c_linear.yaml" "$scenario_dir/baseline_c_linear.log"

  REF_FILE="$(latest_bin "$OUT_DIR/exports/baseline_a/s${sid_fmt}")"
  scenario_compare="$scenario_dir/reference_compare.csv"
  "$PYTHON_EXEC" scripts/compare_policy_exports.py \
    --reference "$REF_FILE" \
    --candidate baseline_b="$(latest_bin "$OUT_DIR/exports/baseline_b/s${sid_fmt}")" \
    --candidate baseline_c_python="$(latest_bin "$OUT_DIR/exports/baseline_c_python/s${sid_fmt}")" \
    --candidate baseline_c_linear="$(latest_bin "$OUT_DIR/exports/baseline_c_linear/s${sid_fmt}")" \
    --output "$scenario_compare"

  tail -n +2 "$scenario_compare" | sed "s/^/${sid_fmt},/" >> "$REF_ALL"
done

echo "summary: $CSV"
echo "reference_compare: $REF_ALL"