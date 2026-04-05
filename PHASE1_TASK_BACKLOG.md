# Phase 1 Task Backlog

Date: 2026-04-04
Project: ShardSim / ChatMPI
Phase objective: deliver distributed baseline solver with coarse/fine execution foundations

## 1. Sprint Structure (Suggested)
- Sprint 1 (Week 2): core solver skeleton + deterministic controls
- Sprint 2 (Week 3): MPI decomposition + halo exchange
- Sprint 3 (Week 4): coarse/fine modes + baseline automation + validation

## 2. Epic A - Repository and Build Foundation
A1. Initialize source layout
- Create modules: solver, mesh, mpi_runtime, orchestrator, metrics, io, config
- Define public headers and ownership boundaries

A2. Build system setup
- CMake targets for core library, CLI runner, unit tests
- Compiler flags for reproducible deterministic builds

A3. Configuration system
- YAML/TOML/INI-based runtime config (choose one)
- Parameters for tolerances, wall-clock cap, fallback policy, logging levels

Definition of done:
- clean configure/build on baseline environment
- reproducible binary generation path documented

## 3. Epic B - Transient FDM Thermal Solver
B1. PDE core implementation
- Implement transient heat equation update for non-uniform mesh
- Support mixed boundary conditions

B2. Deterministic mode
- Eliminate non-deterministic iteration ordering where possible
- Add deterministic reductions and fixed update ordering

B3. Solver verification harness
- Include at least two known cases with expected behavior
- Add error computation against synthetic or reference patterns

Definition of done:
- deterministic reruns produce matching outputs under identical config
- mixed BC and non-uniform mesh tests pass

## 4. Epic C - MPI Distribution and Geometric Partitioning
C1. Geometric partition module
- Implement strict geometric partition assignment
- Define subdomain ownership and interface boundaries

C2. Halo exchange implementation
- Exchange boundary values with neighboring ranks each iteration
- Track communication cost and message sizes

C3. Distributed assembly
- Reconstruct global field snapshots for validation and metrics

Definition of done:
- stable multi-rank runs with correct boundary continuity
- communication telemetry emitted per iteration window

## 5. Epic D - Coarse/Fine Baseline Modes
D1. Fidelity profile implementation
- Coarse mode (looser tolerance)
- Fine mode (strict tolerance)

D2. Runtime mode selection
- CLI/config switch for full-fine and coarse/fine execution profiles

D3. Baseline A and B automation
- A: full-fine baseline runs
- B: multi-fidelity without learning

Definition of done:
- both baselines runnable from scripted commands
- outputs captured in standardized run directories

## 6. Epic E - Metrics, Logging, and Run Artifacts
E1. Metrics schema
- Runtime breakdown: compute, communication, orchestration
- Accuracy metrics: MAE and normalized global error

E2. Artifact format
- Run manifest (config hash, seed, environment, git ref)
- Structured output files for post-analysis

E3. Quality gate checks
- Fail run when memory ceiling or wall-clock policy is violated

Definition of done:
- every run produces machine-readable metrics and manifest
- policy violations are explicit in logs and exit codes

## 7. Epic F - Testing and CI (Phase 1 Minimum)
F1. Unit tests
- mesh indexing and boundary handling
- deterministic mode checks

F2. Integration tests
- 1-rank and multi-rank smoke tests
- baseline scenario regression checks

F3. CI pipeline
- build + unit tests + integration smoke test

Definition of done:
- CI green on baseline branch
- regression artifacts retained for failed runs

## 8. Priority Order
1. Epic A
2. Epic B
3. Epic C
4. Epic D
5. Epic E
6. Epic F

## 9. Risk Controls During Phase 1
- If communication overhead exceeds expected bounds, profile before adding new features
- If deterministic mode cost is too high, preserve deterministic validation path and keep optional perf mode
- If non-uniform mesh complexity blocks progress, ship constrained non-uniform variant first (documented)

## 10. Exit Criteria for Phase 1
Phase 1 is complete when:
- distributed transient solver runs on multiple ranks
- strict geometric partitioning and halo exchange are validated
- coarse and fine profiles are operational
- baseline A and B can be executed reproducibly
- metrics and policy guardrails are generated per run
