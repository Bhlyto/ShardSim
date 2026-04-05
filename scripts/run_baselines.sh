#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ./build/shardsim_cli ]]; then
  echo "Building shardsim_cli..."
  cmake -S . -B build
  cmake --build build -j
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/baselines/$TIMESTAMP"
mkdir -p "$OUT_DIR"

CSV="$OUT_DIR/summary.csv"
echo "baseline,nranks,steps,coarse_steps,fine_steps,critical_cells,critical_fraction,runtime_ms,halo_ms_avg,halo_overhead_ratio,mae,global_error_norm" > "$CSV"

run_one() {
  local baseline_name="$1"
  local config_path="$2"
  local nranks="$3"
  local log_file="$OUT_DIR/${baseline_name}_n${nranks}.log"

  local cmd
  if [[ "$nranks" -gt 1 ]]; then
    if command -v mpirun >/dev/null 2>&1; then
      cmd=(mpirun -n "$nranks" ./build/shardsim_cli "$config_path")
    else
      echo "mpirun not found; cannot run nranks=$nranks" >&2
      return 1
    fi
  else
    cmd=(./build/shardsim_cli "$config_path")
  fi

  echo "Running $baseline_name (n=$nranks)..."
  "${cmd[@]}" | tee "$log_file"

  extract() {
    local key="$1"
    awk -v k="$key" '$1==k":" {print $2}' "$log_file" | tail -n 1
  }

  local steps coarse_steps fine_steps critical_cells critical_fraction runtime_ms halo_ms_avg halo_overhead_ratio mae global_error_norm
  steps="$(extract steps)"
  coarse_steps="$(extract coarse_steps)"
  fine_steps="$(extract fine_steps)"
  critical_cells="$(extract critical_cells)"
  critical_fraction="$(extract critical_fraction)"
  runtime_ms="$(extract runtime_ms)"
  halo_ms_avg="$(extract halo_ms_avg)"
  halo_overhead_ratio="$(extract halo_overhead_ratio)"
  mae="$(extract mae)"
  global_error_norm="$(extract global_error_norm)"

  echo "$baseline_name,$nranks,$steps,$coarse_steps,$fine_steps,$critical_cells,$critical_fraction,$runtime_ms,$halo_ms_avg,$halo_overhead_ratio,$mae,$global_error_norm" >> "$CSV"
}

RANKS=(1 2)
if [[ "$#" -gt 0 ]]; then
  RANKS=("$@")
fi

for n in "${RANKS[@]}"; do
  run_one baseline_a config/baseline_a_fullfine.yaml "$n"
  run_one baseline_b config/baseline_b_multifidelity.yaml "$n"
done

echo "Baseline run complete."
echo "Artifacts: $OUT_DIR"
echo "Summary:   $CSV"
