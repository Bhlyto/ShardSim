# Implementation Status (Phase 1)

Date: 2026-04-04

## Implemented

- Transient thermal FDM solver on non-uniform mesh.
- Mixed boundary conditions.
- Tolerance-driven coarse/fine stopping criteria.
- Strict geometric MPI x-partitioning.
- Halo exchange each iteration with communication telemetry.
- Rank-0-only run summary output.
- Runtime guardrails:
  - wall-clock limit,
  - memory ceiling,
  - max communication-overhead ratio.
- Decision-core adaptive region selection:
  - error proxy + uncertainty proxy thresholds,
  - configurable minimum critical fraction.
- Fine stage restricted to selected critical regions.

## Key Config Parameters

Core numerics:
- `grid_x`, `grid_y`, `steps`
- `dt`, `alpha`
- `coarse_tolerance`, `fine_tolerance`

Decision core:
- `refine_local_error_tau`
- `refine_uncertainty_tau`
- `min_critical_fraction`

Guardrails:
- `memory_ceiling_gb` or `memory_ceiling_mb`
- `wallclock_limit_minutes` or `wallclock_limit_ms`
- `halo_overhead_ratio_max`

Runtime:
- `deterministic_mode`
- `partitioning_policy` (`strict_geometric`)

## Available Test Coverage

- Partition and halo correctness (`single`, `mpi2`).
- KPI overhead guardrail checks (`single`, `mpi2`).
- Policy-failure tests for wall-clock and memory (expected-fail tests).
- Decision-core policy checks:
  - threshold monotonicity,
  - minimum critical-fraction enforcement,
  - mask size consistency.

## Current Limits

- No ML surrogate integrated yet (uncertainty proxy is gradient-based).
- Adaptive selection currently uses heuristic proxies, not trained uncertainty.
- Baseline automation/reporting is script-based (not yet integrated with CI artifacts).
