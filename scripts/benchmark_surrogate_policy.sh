#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cmake --build build -j >/dev/null

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/surrogate_policy_benchmark/$TIMESTAMP"
mkdir -p "$OUT_DIR"

CSV="$OUT_DIR/summary.csv"
echo "baseline,steps,coarse_steps,fine_steps,critical_cells,critical_fraction,decision_ms,runtime_ms,mae,global_error_norm" > "$CSV"

if [[ ! -f models/surrogate_linear_policy.txt ]]; then
  /home/bhlyto/.venv/bin/python scripts/train_linear_policy.py \
    --paired-data-dir runs/hardened_id_train \
    --model-output models/surrogate_linear_policy.txt >/dev/null
fi

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

run_one baseline_a config/baseline_a_fullfine_256.yaml
run_one baseline_b config/baseline_b_multifidelity_256.yaml
run_one baseline_c_python config/baseline_c_surrogate_256.yaml
run_one baseline_c_linear config/baseline_c_surrogate_linear_256.yaml

echo "summary: $CSV"