# Phase 0 Requirements Draft

Date: 2026-04-04
Project: ShardSim / ChatMPI
Scope: v1 thermal domain only

## 1. Confirmed Design Decisions

### 1.1 Objectives
- Primary objective: target runtime reduction versus full-fine baseline.
- Accuracy remains constrained by full-fine reference comparison.

### 1.2 Physics and Numerics
- Problem class: transient heat equation.
- Numerical method for v1: finite difference method (FDM).
- Boundary conditions: mixed boundary conditions must be supported.
- Mesh strategy: non-uniform mesh is required.

### 1.3 Fidelity and Refinement
- Fidelity controls: solver tolerance-based control with coarse/fine ratio target around 5-10%.
- Refinement trigger: hybrid criterion using:
  - threshold on estimated local error,
  - uncertainty estimate.

### 1.4 Distributed Strategy
- Partitioning policy: strict geometric partitioning.
- Runtime scheduling: dynamic task reassignment by the decision core.

### 1.5 Data and Ground Truth
- Reference truth for validation/training: full-fine simulation only.
- Data contract scope: include all key elements (field format, metadata, units, coordinates, checkpoints).

### 1.6 Observability and Reproducibility
- Required metrics/logging scope: include all major operational signals.
- Reproducibility priority: deterministic solver mode required.

### 1.7 Learning Module Behavior
- Surrogate objectives:
  - predict correction field,
  - predict fine field directly.
- Safety policy:
  - enforce uncertainty bounds,
  - automatic fallback to physics-only mode when uncertainty/policy conditions are violated.

### 1.8 Non-Functional Constraints
- Must define and enforce:
  - memory ceiling per node,
  - max wall-clock time per scenario,
  - portability targets.

## 2. Architecture Implications (Phase 0 Outcome)

- Decision core must support both adaptive region scoring and dynamic reassignment.
- Solver interface must expose deterministic execution controls and tolerance knobs.
- Mesh subsystem must support non-uniform discretization and geometric ownership mapping.
- Error-estimation pipeline must unify numerical local error and surrogate uncertainty into a single refinement score.
- Runtime guardrails must include fallback path switching from surrogate-assisted to physics-only computation.
- Data model must be standardized early to avoid integration debt between solver, orchestrator, and ML pipeline.

## 3. KPI Definition Template (To Finalize)

Populate numeric targets before Phase 1 starts.

- Runtime reduction target vs full-fine baseline: 30% (minimum).
- Max global error threshold E_max: TBD.
- Local error threshold tau for refinement: TBD.
- Parallel efficiency floor at N nodes: TBD.
- Max memory per node: 16 GB/node.
- Max wall-clock per benchmark scenario: configurable per case via config file.

## 4. Module-Level Acceptance Criteria (Draft)

### 4.1 Solver
- Produces deterministic outputs under fixed seed/config.
- Supports transient FDM with non-uniform mesh and mixed BCs.
- Exposes coarse/fine tolerance profiles.

### 4.2 Decision Core
- Computes refinement score from local error + uncertainty.
- Executes dynamic task reassignment while preserving geometric partition constraints.
- Logs reassignment and refinement decisions per iteration.

### 4.3 Learning
- Supports two outputs (correction field and direct fine prediction).
- Emits uncertainty estimate for every prediction region.
- Triggers fallback automatically based on uncertainty policy.

### 4.4 Observability
- Captures runtime breakdown (compute, communication, orchestration).
- Captures refinement-map statistics and estimator reliability signals.
- Captures final accuracy/cost metrics per run.

## 5. Open Items To Lock This Week

1. Numerical values for E_max and tau.
2. Node counts and hardware profile used for efficiency target.
3. Portability matrix (Linux distro, compiler versions, MPI implementation).
4. Fallback trigger formula (uncertainty threshold and hysteresis rule).

## 6. Working Defaults (Proposed Until Final Values Exist)

These defaults are temporary to unblock Phase 1 planning.

- E_max default: 2.0% normalized global error.
- tau default: 5.0% local relative error threshold.
- Parallel efficiency default target: >= 70% at 8 nodes.
- Portability baseline: Ubuntu 22.04 LTS, GCC 12+, OpenMPI 5.x, CMake 3.24+.
- Fallback default: switch to physics-only mode when predictive uncertainty > 0.20 for 3 consecutive iterations.

## 7. Immediate Next Step (Phase 0 -> Phase 1)

Once section 5 is numerically finalized, generate:
- formal requirements spec (v1.0),
- architecture decision records (ADRs),
- benchmark protocol sheet,
- module interfaces for solver, decision core, and learning pipeline.
