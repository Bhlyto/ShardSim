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

## Phase 1+ — Partially Implemented 🔄

### ML Surrogate Integration
Three decision policies are available via the `decision_policy` config key:

| Policy             | Status | Notes |
|--------------------|--------|-------|
| `heuristic`        | ✅ Default | Gradient + Laplacian proxies; no external deps |
| `surrogate_python` | 🔄 Working | Shells out to `scripts/predict_critical_mask.py`; requires XGBoost model |
| `surrogate_linear` | 🔄 Working | Native in-process linear regression; requires pre-trained weight file |
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
- `benchmark_surrogate_policy.sh` / `benchmark_policy_reference.sh` — Live benchmarking.

### 3D Support
- Non-uniform 3D mesh structures scaffolded (`mesh3d.hpp` / `mesh3d.cpp`).
- 3D heat solver implemented (`heat_solver3d.cpp`) but **single-rank only**.
- `run_3d_workflow.sh` pipeline exists; not production-grade.

### Training Data Export
- Binary format export of paired coarse + fine fields.
- Config keys: `export_training_data`, `training_data_export_dir`.
- Supports scenario metadata (grid shape, convergence steps).

---

## Not Yet Implemented ❌ (Phase 2+)

- MPI-distributed 3D solver.
- 2D/3D domain decomposition (currently 1D x-strips only).
- Load-aware cell migration and dynamic task redistribution.
- Checkpoint / restart after coarse stage.
- Analytical pre-simulation module (fast approximation + uncertainty map).
- Learned uncertainty estimation (replacing heuristic proxies).
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
- `decision_policy` (`heuristic` | `surrogate_python` | `surrogate_linear`)
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
