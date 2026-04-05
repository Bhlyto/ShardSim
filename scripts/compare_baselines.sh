#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -gt 1 ]]; then
  echo "usage: ./scripts/compare_baselines.sh [summary_csv]"
  exit 2
fi

if [[ "$#" -eq 1 ]]; then
  CSV="$1"
else
  LATEST_DIR="$(ls -1dt runs/baselines/* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$LATEST_DIR" ]]; then
    echo "No baseline runs found under runs/baselines/."
    exit 1
  fi
  CSV="$LATEST_DIR/summary.csv"
fi

if [[ ! -f "$CSV" ]]; then
  echo "summary csv not found: $CSV"
  exit 1
fi

awk -F, '
NR==1 { next }
{
  key=$2;
  baseline=$1;
  runtime=$8+0.0;
  mae=$11+0.0;
  gerr=$12+0.0;
  if (baseline=="baseline_a") {
    a_runtime[key]=runtime;
    a_mae[key]=mae;
    a_gerr[key]=gerr;
  } else if (baseline=="baseline_b") {
    b_runtime[key]=runtime;
    b_mae[key]=mae;
    b_gerr[key]=gerr;
  }
}
END {
  printf("Using summary: %s\n\n", FILENAME);
  printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s\n", "nranks", "A_runtime", "B_runtime", "speedup", "A_gerr", "B_gerr", "gerr_ratio");

  for (k in a_runtime) {
    if (!(k in b_runtime)) {
      continue;
    }

    speedup = (b_runtime[k] > 0.0) ? a_runtime[k] / b_runtime[k] : 0.0;
    gerr_ratio = (a_gerr[k] > 0.0) ? b_gerr[k] / a_gerr[k] : 0.0;

    printf("%-8s %-12.4f %-12.4f %-12.4f %-12.6f %-12.6f %-12.4f\n",
           k, a_runtime[k], b_runtime[k], speedup, a_gerr[k], b_gerr[k], gerr_ratio);
  }

  printf("\nNotes:\n");
  printf("- speedup = A_runtime / B_runtime (>1 means B is faster).\n");
  printf("- gerr_ratio = B_global_error / A_global_error (<1 means B is more accurate).\n");
}
' "$CSV"
