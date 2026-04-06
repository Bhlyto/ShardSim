# Implementation Status

Last updated: 2026-04-06

## Phase 1 — Complete ✅

### Solver & Numerics
- Transient 2D heat-diffusion FDM solver on non-uniform mesh.
- Mixed boundary conditions (Dirichlet + Neumann).
- Explicit time-stepping with CFL stability check.
- Tolerance-driven coarse/fine stopping criteria (transient-rate decay detection).

### Distributed Execution (MPI)
- Strict geometric x-partitioning across ranks.
- Bidirectional halo exchange every iteration with per-iteration timing telemetry.
- Global AllReduce for convergence synchronisation.
- Rank-0-only run summary (no chatty distributed logging).
- MPI-optional builds: single-rank fallback when MPI is absent.

### Decision Core — Adaptive Refinement
- Error proxy: Laplacian magnitude (correlates with truncation error).
- Uncertainty proxy: gradient magnitude (rapid-change regions).
- Threshold-based critical-cell selection (OR logic on error + uncertainty thresholds).
- Rank-promotion mechanism to enforce a configurable minimum critical fraction.
- Deterministic candidate ranking for reproducible selection.
- Masked fine solve: non-critical cells frozen at coarse values.

### Runtime Guardrails
- Hard wall-clock limit (fail-fast exception).
- Memory ceiling policed via RSS (Linux `/proc/self/status`).
- Communication-overhead ratio cap (`halo_ms / total_ms`).

### Deterministic Mode
- Compiler flag `SHARDSIM_DETERMINISTIC_MODE` (default ON).
- Fixed update ordering and deterministic reductions.

### Metrics & Observability
- Per-run summary: coarse steps, fine steps, critical cells, critical fraction.
- Communication breakdown: halo time (min/avg/max across ranks), overhead ratio.
- Accuracy metrics: MAE, global error norm.
- CSV export for baseline comparison (`BASELINES.md`).

### Weak-Scaling Validation (n = 1 / 2 / 4)
| n | Baseline A (full-fine) | Baseline B (adaptive) | Speedup | Error ratio |
|---|------------------------|------------------------|---------|-------------|
| 1 | 289.2 ms               | 127.6 ms               | 2.27×   | 1.046×      |
| 2 | 155.7 ms               | 64.7 ms                | 2.41×   | 1.046×      |
| 4 | 89.6 ms                | 35.4 ms                | 2.53×   | 1.046×      |

Adaptive weak-scaling efficiency: **90.3 %** (vs 80.5 % for full-fine).

### Test Coverage — 9/9 Passing
- Partition and halo correctness (`single`, `mpi2`).
- KPI communication-overhead guardrail checks (`single`, `mpi2`).
- Expected-fail tests: wall-clock violation, memory violation (`single`, `mpi2`).
- Decision-core policy unit tests: threshold monotonicity, min-fraction enforcement,
  mask-size consistency.
- 3D mesh smoke test.

---

## Phase 1+ — Complete ✅

### ML Surrogate Integration
Four decision policies are available via the `decision_policy` config key:

| Policy                    | Status | Notes |
|---------------------------|--------|-------|
| `heuristic`               | ✅ Default | Gradient + Laplacian proxies; no external deps |
| `surrogate_python`        | ✅ Working | One-shot subprocess to `predict_critical_mask.py`; XGBoost model |
| `surrogate_python_cached` | ✅ Working | Persistent worker process over stdio JSON protocol; eliminates per-step startup; 2D + 3D |
| `surrogate_linear`        | ✅ Working | Native in-process linear regression; requires pre-trained weight file |

**Cached worker path** (`surrogate_python_cached`):
- Spawns `predict_critical_mask.py --worker-stdio` once at solver start.
- Each step sends a JSON request over stdin and reads a JSON response from stdout.
- Falls back to one-shot mode if the worker process fails.
- Supported in both 2D (`heat_solver.cpp`) and 3D (`heat_solver3d.cpp`).
- Single-rank only for 3D; no restriction on rank count for 2D.

Training pipelines (Python):
- `train_surrogate.py` — XGBoost two-stage trainer (GPU preferred).
- `train_linear_policy.py` — Native linear policy trainer.
- `generate_paired_real_training_data.py` — Diverse scenario dataset generation.
- `train_domain_models_from_tutorials.py` — Per-solver/domain trainer from OpenFOAM tutorials;
  selects compatible tutorial cases, builds case-paired data, trains and emits a router manifest.
- `benchmark_surrogate_policy.sh` / `benchmark_policy_reference.sh` — Live benchmarking.

### Domain Model Router
- Per-solver models stored under `runs/models_by_domain/<Domain>/<solver>/`.
- Router manifest: `reports/surrogate_model_router.json` — consumed directly by
  `surrogate_model_path` config key at inference time.
- Domains: `Thermal` (solidFoam, laplacianFoam), `CFD` (icoFoam).

### 3D Support
- Non-uniform 3D mesh structures scaffolded (`mesh3d.hpp` / `mesh3d.cpp`).
- 3D heat solver implemented (`heat_solver3d.cpp`); **single-rank only**.
- 3D surrogate mask script: `predict_critical_mask_3d.py`; supports one-shot and
  `--worker-stdio` cached-server modes.
- `run_3d_workflow.sh` pipeline exists; validated via smoke test.

### Training Data Export
- Binary format export of paired coarse + fine fields.
- Config keys: `export_training_data`, `training_data_export_dir`.
- Supports scenario metadata (grid shape, convergence steps).

### Iterative Active Learning Loop
- `scripts/iterative_openfoam_active_learning.py` automates continual retraining:
  - benchmark OpenFOAM case,
  - build case-paired training data,
  - accumulate global dataset across rounds/cases,
  - retrain surrogate each round,
  - select best model by validation metric,
  - emit decision-core-ready config snippet.
- `scripts/build_case_paired_training_data.py` — reusable ingestion bridge from
  OpenFOAM benchmark runs to paired surrogate training samples.

### Multi-Loop Retrain + Inference Automation
- `scripts/run_retrain_loops.py` — orchestrates N retrain loops; after each loop runs
  per-solver inference and writes `loop_reports/loop_<N>.json` +
  `loop_reports/inference_loop_<N>.json`; produces a consolidated `summary.json`.
- `scripts/infer_domain_models_selected_cases.py` — standalone inference/eval runner;
  benchmarks selected cases, injects surrogate config via current domain router,
  applies retry profiles (`quality` → `safe_mid` → `safe_high`) to handle zero-interior-cell
  failures, parses metrics, and writes a JSON report.

**5-loop benchmark results (2026-04-06, single tutorial case per solver):**

| Solver | Baseline MAE | Surrogate MAE | Δ MAE | Δ % | Sur. runtime |
|--------|-------------|---------------|-------|-----|--------------|
| solidFoam | 0.004174 | 1.38 × 10⁻⁷ | −0.004174 | **−100 %** | ~990 ms |
| laplacianFoam | 0.002356 | 9.88 × 10⁻⁵ | −0.002257 | **−95.8 %** | ~1 080 ms |
| icoFoam | 0.001600 | 0.001251 | −0.000349 | **−21.8 %** | ~1 070 ms |

All three solvers converged to a stable floor after loop 1; additional loops on the same
single-case training set yield no further MAE reduction. Runtime (~1 s per step) is dominated
by Python process overhead; switching to `surrogate_python_cached` eliminates this in
continuous solver runs.

### External Case Ingestion (OpenFOAM)
- `scripts/import_openfoam_case.py` scaffold added.
- Converts OpenFOAM case metadata into canonical ShardSim YAML + JSON conversion report.
- Supports `strict` and `permissive` validation modes.
- Initial extraction includes mesh boundary patches, mesh counts, controlDict values, and transport properties.

---

## Phase 3 — Pre-Simulation Module ✅

### Overview
A fast analytical pre-simulation runs a coarsened heat solve before the main adaptive workflow.
Its uncertainty scores augment the decision-core critical-cell mask, catching regions that the
purely reactive heuristic misses.

### Implementation
- `include/shardsim/presim/presim.hpp` — public API (`UncertaintyMap`, `run_presim()`, `blend_presim_scores()`).
- `src/presim/presim.cpp` — implementation (~220 lines):
  - `make_coarse_grid()`: downsample full grid by `presim_coarsening_factor` (min 2×2).
  - `run_coarse_heat()`: explicit FDM on coarse grid, CFL-stable dt, Dirichlet x=0, Neumann other edges.
  - `compute_uncertainty_scores()`: normalised `0.5·|∇T| + 0.5·|∇²T|` per cell.
  - `upsample()`: bilinear interpolation back to full grid dimensions.
  - `run_presim()`: orchestrates the above; returns empty map when `presim_steps == 0` (disabled).
  - `blend_presim_scores()`: weighted blend `(1-w)·heuristic + w·presim`.
- Policy dispatcher augmented: after the base policy returns a mask, any cell where
  `presim_map.scores[k] > refine_local_error_tau` is additionally marked critical (OR logic).

### Config Keys
| Key | Default | Description |
|-----|---------|-------------|
| `presim_steps` | `0` (disabled) | Steps to run in coarsened presim pass |
| `presim_coarsening_factor` | `4` | Spatial downscaling factor per dimension |

### Test Coverage — 14/14 Passing
Added `tests/presim_test.cpp` with 6 unit tests:
- Disabled presim returns empty `UncertaintyMap`.
- Output dimensions match full grid.
- All scores are in [0, 1].
- `blend_presim_scores` preserves size and produces correct weighted average.
- Coarsening factor of 1 uses same-size intermediate grid.
- Blend is a no-op when the uncertainty map is empty.

---

## Phase 4 — ML Correction Loop ✅

### Overview
A trained ML model predicts the coarse-to-fine discrepancy (`fine − coarse`), enabling
direct correction of the coarse field without running an expensive fine solve.

### Implementation
- `include/shardsim/correction/correction.hpp` — public API (`apply_correction()`).
- `src/correction/correction.cpp` — implementation (~280 lines):
  - `LinearPatchModel` loading from `.txt` format (same as surrogate training).
  - `apply_linear()`: native in-process patch-based linear regression applied per cell.
  - `apply_python()`: shells out to `apply_correction.py` for XGBoost models.
  - Supports binary field I/O for Python interop.
- `scripts/apply_correction.py` — Python inference script using XGBoost or sklearn models.
- Solver integration:
  - `SolveResult` + `SolveResult3D` gain `correction_applied` bool flag.
  - In `run_transient_heat()`, when `correction_policy != "none"`, the coarse field
    is corrected and returned as `fine`; no actual fine solve runs (`fine_steps = 0`).
  - Config keys: `correction_policy` (`"none"` | `"linear"` | `"python"`),
    `correction_model_path`, `correction_script_path`, `correction_python_executable`.

### Training Data & Models
- Training data: exported via `export_training_data` in binary paired format.
- Model training: `train_surrogate.py` (XGBoost), `train_linear_policy.py` (Ridge).
- Pre-trained model: `models/surrogate_linear_policy.txt` (native linear patch format).
- Alternative: `models/surrogate_hardened_fullscale.pkl` (XGBoost, via Python script).

### Test Coverage — 15/15 Passing
Added `tests/correction_test.cpp` with 7 unit tests:
- `correction_policy = "none"` passes field through unchanged.
- Empty `correction_policy` treated as no-op.
- `correction_policy = "linear"` with missing/bad model path raises error.
- Real model produces correct output dimensions, all finite values.
- Real model modifies field (non-zero corrections).
- Unsupported policy raises error.

### Config Examples
```yaml
# No correction (baseline)
correction_policy: none

# Native linear correction
correction_policy: linear
correction_model_path: models/surrogate_linear_policy.txt

# XGBoost Python correction
correction_policy: python
correction_model_path: models/surrogate_hardened_fullscale.pkl
correction_script_path: scripts/apply_correction.py
correction_python_executable: <venv>/bin/python
```

---

## Not Yet Implemented ❌ (Phase 5+)

- MPI-distributed 3D solver.
- 2D/3D domain decomposition (currently 1D x-strips only).
- Load-aware cell migration and dynamic task redistribution.
- Checkpoint / restart after coarse stage.
- Extended PDE support (CFD, structural mechanics).
- GitHub Actions CI/CD integration.
- Real-time solver visualisation.

---

## Key Config Parameters

Core numerics:
- `grid_x`, `grid_y`, `steps`
- `dt`, `alpha`
- `coarse_tolerance`, `fine_tolerance`

Decision core:
- `decision_policy` (`heuristic` | `surrogate_python` | `surrogate_python_cached` | `surrogate_linear`)
- `refine_local_error_tau`
- `refine_uncertainty_tau`
- `min_critical_fraction`
- `surrogate_model_path` (for surrogate policies)

Guardrails:
- `memory_ceiling_gb` or `memory_ceiling_mb`
- `wallclock_limit_minutes` or `wallclock_limit_ms`
- `halo_overhead_ratio_max`

Runtime:
- `deterministic_mode`
- `partitioning_policy` (`strict_geometric`)
- `export_training_data`, `training_data_export_dir`
