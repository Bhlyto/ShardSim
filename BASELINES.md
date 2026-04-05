# Baseline Automation

## Baseline Definitions

- Baseline A (`config/baseline_a_fullfine.yaml`):
  - full-fine style reference where the decision core enforces near-global refinement (`min_critical_fraction: 1.0`).
- Baseline B (`config/baseline_b_multifidelity.yaml`):
  - adaptive multi-fidelity without ML surrogate (`min_critical_fraction: 0.0`).

## Run Script

```bash
./scripts/run_baselines.sh
```

Optional custom rank list:

```bash
./scripts/run_baselines.sh 1 2 4
```

## Compare Baselines

Use the comparator to compute speedup and error-ratio from a summary CSV.

```bash
./scripts/compare_baselines.sh
```

Optional explicit CSV path:

```bash
./scripts/compare_baselines.sh runs/baselines/<timestamp>/summary.csv
```

## Outputs

Each invocation writes to:

- `runs/baselines/<timestamp>/`
- per-run logs:
  - `baseline_a_n<ranks>.log`
  - `baseline_b_n<ranks>.log`
- summary CSV:
  - `summary.csv`

## CSV Columns

- `baseline`
- `nranks`
- `steps`
- `coarse_steps`
- `fine_steps`
- `critical_cells`
- `critical_fraction`
- `runtime_ms`
- `halo_ms_avg`
- `halo_overhead_ratio`
- `mae`
- `global_error_norm`
