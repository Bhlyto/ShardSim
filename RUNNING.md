# Running ShardSim Scaffold

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/shardsim_cli config/default.yaml
```

## Run Tolerance-Demo Scenario

```bash
./build/shardsim_cli config/tolerance_demo.yaml
```

Expected behavior in this demo profile:
- `coarse_steps` should stop very early due to loose coarse tolerance.
- `fine_steps` should run much longer (often until max `steps`) due to strict fine tolerance.
- Convergence uses transient-rate decay (`||dT/dt||`): tolerance can act as a relative ratio target or an absolute rate threshold.

## Notes

- This is a Phase 1 scaffold aligned with the backlog.
- The transient FDM solver, non-uniform mesh, and mixed boundaries are minimal initial implementations.
- Coarse and fine solves now stop on tolerance convergence or when `steps` max iterations is reached.
- Decision-core adaptive region selection is enabled (`error + uncertainty` thresholds, plus `min_critical_fraction`).
- Runtime guardrails are enabled for wall-clock, memory, and communication-overhead ratio.

## Configuration Highlights

- Decision core:
	- `refine_local_error_tau`
	- `refine_uncertainty_tau`
	- `min_critical_fraction`
- Guardrails:
	- `wallclock_limit_minutes` or `wallclock_limit_ms`
	- `memory_ceiling_gb` or `memory_ceiling_mb`
	- `halo_overhead_ratio_max`

See `IMPLEMENTATION_STATUS.md` for a full status snapshot.

## Baselines

Use `scripts/run_baselines.sh` for Baseline A/B automation and CSV export.
See `BASELINES.md` for details.

## Export Real Training Data

Use the large adaptive profile to export coarse/fine fields for surrogate training:

```bash
./build/shardsim_cli config/baseline_b_large_ml.yaml
```

This writes binary samples under `runs/training_data/` (configurable via `training_data_export_dir`).
Each sample contains:
- coarse field
- fine field
- solver metadata (grid shape, coarse/fine steps, critical fraction)

Relevant config keys:
- `export_training_data` (`true`/`false`)
- `training_data_export_dir` (output directory for `.bin` files)

## Train Surrogate From Real Solver Data

```bash
source /home/bhlyto/.venv/bin/activate
python3 scripts/train_surrogate.py \
	--grid-size 256 \
	--data-dir runs/training_data \
	--model-output models/surrogate_real.pkl
```

If `--data-dir` is omitted, the trainer keeps the previous synthetic-data behavior.

## Generate Paired Real Training Data (Recommended)

The command below generates diverse scenarios (varying source position and temperature), then exports paired binaries:
- `scenario_XXXX_coarse.bin` from adaptive run
- `scenario_XXXX_fullfine.bin` from full-fine reference run

```bash
/home/bhlyto/.venv/bin/python scripts/generate_paired_real_training_data.py \
	--n-scenarios 32 \
	--output-dir runs/training_pairs
```

Then train against full-fine discrepancy:

```bash
python3 scripts/train_surrogate.py \
	--grid-size 256 \
	--paired-data-dir runs/training_pairs \
	--num-rounds 240 \
	--active-threshold 1e-8 \
	--active-weight 10 \
	--model-output models/surrogate_real_paired.pkl
```

For a stricter sparse-target setup, enable two-stage training (active-cell classifier + regressor):

```bash
python3 scripts/train_surrogate.py \
	--grid-size 256 \
	--paired-data-dir runs/training_pairs \
	--two-stage \
	--active-threshold 1e-8 \
	--clf-rounds 150 \
	--model-output models/surrogate_real_paired_twostage.pkl
```

And evaluate on real paired holdout samples:

```bash
python3 scripts/visualize_surrogate.py \
	--model models/surrogate_real_paired.pkl \
	--grid-size 256 \
	--paired-data-dir runs/training_pairs \
	--n-samples 16 \
	--output reports/surrogate_eval_real_paired.png
```

## Hard OOD Split Generation

Generate in-distribution training and OOD evaluation scenarios with the same script:

```bash
python3 scripts/generate_paired_real_training_data.py \
	--n-scenarios 128 \
	--complexity complex \
	--distribution id \
	--output-dir runs/training_pairs_id

python3 scripts/generate_paired_real_training_data.py \
	--n-scenarios 32 \
	--complexity complex \
	--distribution ood \
	--output-dir runs/training_pairs_ood
```

This gives a harder generalization test than seed-only splits.

For a stricter stress test with a farther OOD regime, use:

```bash
bash scripts/run_stress_ood_benchmark.sh
```

This trains on `complexity=stress, distribution=id` and evaluates on `complexity=stress, distribution=far_ood`.

## Surrogate In The Solver Loop

The 2D decision core now supports two policies:
- `decision_policy: heuristic`
- `decision_policy: surrogate_python`

The surrogate-backed policy shells out to `scripts/predict_critical_mask.py`, loads a trained model, and returns a critical-cell mask to the fine stage. Relevant config keys:
- `surrogate_model_path`
- `surrogate_python_executable`
- `surrogate_script_path`
- `surrogate_temp_dir`
- `surrogate_score_threshold`
- `surrogate_top_fraction`

Run the live solver benchmark comparing full-fine, heuristic multifidelity, and surrogate-driven multifidelity:

```bash
bash scripts/benchmark_surrogate_policy.sh
```

This now benchmarks four live policies:
- Baseline A full-fine
- Baseline B heuristic multifidelity
- Baseline C Python/XGBoost selector
- Baseline C native linear selector

It writes CSV/logs under `runs/surrogate_policy_benchmark/<timestamp>/`.

To train the native linear patch selector directly:

```bash
/home/bhlyto/.venv/bin/python scripts/train_linear_policy.py \
	--paired-data-dir runs/hardened_id_train \
	--model-output models/surrogate_linear_policy.txt
```

Use `decision_policy: surrogate_linear` with `surrogate_model_path: models/surrogate_linear_policy.txt` to enable native in-process inference.

## Direct Reference Benchmark

To compare B, Python-C, and native-linear-C directly against a same-scenario full-fine Baseline A field export:

```bash
bash scripts/benchmark_policy_reference.sh
```

This writes:
- runtime/selection metrics to `runs/policy_reference_benchmark/<timestamp>/summary.csv`
- direct field-vs-reference metrics to `runs/policy_reference_benchmark/<timestamp>/reference_compare.csv`

## 3D Meshing Scaffold

3D mesh structures and non-uniform 3D grid generation are scaffolded in:
- `include/shardsim/mesh/mesh3d.hpp`
- `src/mesh/mesh3d.cpp`

There is now also a minimal single-rank 3D transient solve/export path. Generate paired 3D data, train a 3D local-patch surrogate, and evaluate it with:

```bash
bash scripts/run_3d_workflow.sh
```

To scale that 3D workflow up, override the environment variables:

```bash
TRAIN_SCENARIOS=64 EVAL_SCENARIOS=16 GRID_X=40 GRID_Z=16 \
	OUT_DIR=runs/workflow_3d_scaled \
	TRAIN_DIR=runs/training_pairs_3d_scaled_train \
	EVAL_DIR=runs/training_pairs_3d_scaled_eval \
	MODEL=models/surrogate_3d_scaled.pkl \
	REPORT=reports/surrogate_3d_scaled_eval.json \
	bash scripts/run_3d_workflow.sh
```

Defaults:
- `GRID_X=32`
- `GRID_Z=12`
- `TRAIN_SCENARIOS=24`
- `EVAL_SCENARIOS=8`

The 3D path currently supports only `decision_policy: heuristic` and single-rank runs.

Config keys prepared for 3D workflows:
- `grid_z`
- `source_z_fraction`
- `source2_z_fraction`
