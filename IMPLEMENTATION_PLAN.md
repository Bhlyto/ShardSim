# ShardSim / ChatMPI Implementation Plan

Date: 2026-04-04
Scope: v1 focused on 2D thermal diffusion (single physical domain)

## 1) Project Goal
Build a distributed multi-fidelity simulation platform that combines:
- a decision core,
- a pre-simulation module,
- coarse and fine simulation layers,
- a thermal-domain surrogate learning loop,

with the objective of minimizing compute cost while respecting an accuracy bound.

## 2) Workstreams
- W1: Core solver and MPI distribution
- W2: Decision core and adaptive refinement
- W3: Pre-simulation and uncertainty map
- W4: ML thermal surrogate (domain-specific)
- W5: Experiment protocol and benchmarking
- W6: Engineering hardening (CI, reproducibility, observability)

## 3) Phased Timeline (Approximate)

### Phase 0 - Requirements and architecture (Week 1)
Deliverables:
- Functional and non-functional requirements
- Success KPIs (error threshold, speedup target, parallel efficiency)
- Architecture Decision Record (ADR)

Tasks:
- Freeze v1 scope to thermal 2D only
- Define fidelity levels and refinement criteria
- Define benchmark scenarios and acceptance criteria

### Phase 1 - Distributed baseline solver (Weeks 2-4)
Deliverables:
- MPI thermal diffusion solver with domain decomposition and halo exchange
- Coarse and fine solver modes
- Baseline run scripts

Tasks:
- Implement PDE discretization and time stepping
- Add decomposition and boundary synchronization
- Validate numerical consistency on known test cases

### Phase 2 - Decision core and adaptive refinement (Weeks 5-7)
Deliverables:
- Orchestrator that selects coarse/fine regions
- Refinement policy based on local error estimator

Tasks:
- Implement region scoring (critical vs non-critical)
- Add allocation strategy and simulation scheduling
- Add aggregation of regional outputs into global field

### Phase 3 - Pre-simulation module (Weeks 8-9)
Deliverables:
- Fast low-resolution pre-simulation
- Uncertainty/error proxy map

Tasks:
- Build pre-sim pipeline
- Feed uncertainty map into decision core
- Validate correlation between proxy error and true error

### Phase 4 - ML correction loop (Weeks 10-12)
Deliverables:
- Thermal surrogate training/inference pipeline
- Correction model integrated in adaptive loop

Tasks:
- Define training dataset format (coarse state, BCs, geometry -> correction/fine estimate)
- Train and evaluate model under in-domain scenarios
- Add model versioning and fallback policy

### Phase 5 - Validation protocol (Weeks 13-14)
Deliverables:
- Results comparing 3 strategies:
  - A: Full fine baseline
  - B: Multi-fidelity without learning
  - C: Full ChatMPI loop
- Metrics report and conclusions

Tasks:
- Compute MAE, global error norm, runtime, fine-cell count, cost-accuracy efficiency
- Execute multiple seeds/scenarios
- Produce reproducible reports

### Phase 6 - Hardening and operations (Weeks 15-16)
Deliverables:
- CI pipeline and regression suite
- Profiling and communication-cost dashboards
- Runbooks and reproducibility guide

Tasks:
- Add automated testing and quality gates
- Add runtime instrumentation and performance traces
- Package launch configs and documented procedures

## 4) Resource Requirements

### 4.1 Material Resources
Compute:
- Dev workstation: >= 8 CPU cores, >= 32 GB RAM
- Cluster/cloud test capacity: 4-16 nodes preferred for scaling experiments

Storage:
- >= 200 GB for datasets, checkpoints, and logs

Software:
- C++17/20 toolchain (GCC/Clang)
- CMake
- MPI stack (OpenMPI or MPICH)
- Python environment for orchestration/training/evaluation
- Plot/report stack (matplotlib/pandas or equivalent)

Optional:
- GPU for faster surrogate training when dataset scales up

### 4.2 Intellectual Resources (People)
- HPC engineer (MPI, decomposition, communication optimization)
- Numerical methods engineer (PDE correctness, stability, error estimators)
- ML engineer (surrogate design, training, evaluation)
- Tech lead/research lead (architecture and scientific protocol)
- DevOps/reproducibility support (CI/CD and experiment automation)

Suggested minimal team:
- 3-4 people for prototype
- 5-7 people for robust research-grade v1

### 4.3 Documentary Resources
- Requirements specification
- Architecture diagrams and ADRs
- Data and model schema documentation
- Experiment protocol with exact scenario definitions
- Testing strategy and acceptance checklist
- Operational runbook and troubleshooting guide

## 5) KPI Targets (Initial)
- Accuracy:
  - Global error <= E_max (to be set in Phase 0)
- Performance:
  - Runtime reduction vs full-fine baseline >= 2x (target)
- Efficiency:
  - Cost/accuracy metric better than baseline B and A
- Scalability:
  - Acceptable parallel efficiency at target node counts

## 6) Risks and Mitigations
- Risk: Error estimator is weak
  - Mitigation: Conservative thresholding and frequent calibration against full-fine runs

- Risk: MPI communication overhead dominates
  - Mitigation: Partition optimization, overlap compute/comm, profiling from Phase 1 onward

- Risk: Surrogate overfitting
  - Mitigation: Strict train/val/test separation and in-domain validity boundaries

- Risk: Scope creep to multi-domain too early
  - Mitigation: Keep v1 thermal-only, create separate roadmap for CFD/other domains

- Risk: Non-reproducible experiments
  - Mitigation: Pinned dependencies, seeded runs, versioned datasets/configs

## 7) Immediate Next Actions (Week 1)
1. Finalize v1 KPI thresholds and benchmark cases.
2. Create architecture skeleton (modules: decision_core, presim, solver, learning, orchestrator, metrics).
3. Implement first MPI heat solver with coarse/fine modes.
4. Add baseline automation scripts for A and B strategies.
5. Define dataset schema for future ML correction training.

## 8) Estimated Total Duration
- Fast prototype: 6-8 weeks (reduced validation/hardening)
- Robust v1: ~16 weeks (full plan above)
- Extended publication-grade program: 4-6 months

## 9) Post-v1 Extensibility Track (Geometry/External Case Import)
Objective:
- Allow potential users to import external simulation cases (starting with OpenFOAM) and run them through a unified ShardSim workflow without rewriting orchestration.

Planned approach:
- Introduce a canonical ShardSim case schema (YAML/JSON) as the single runtime input format.
- Build an importer/translator CLI for OpenFOAM that converts case data into canonical schema.
- Keep solver-specific parsing isolated in adapters; keep workflow/orchestration physics-agnostic.

Planned deliverables:
- `import-openfoam` script/app:
  - Input: OpenFOAM case directory (`constant/polyMesh`, `0/`, `system/`).
  - Output: canonical ShardSim case YAML + conversion report.
- Schema and mapping docs:
  - Geometry/topology, BCs, material/transport properties, initial fields, solver controls.
  - Provenance fields (`source_format`, `source_path`, `schema_version`, conversion warnings).
- Validation modes:
  - Strict mode (fail on unsupported constructs).
  - Permissive mode (warn + default fallbacks).

Integration notes:
- Runtime should consume only canonical schema regardless of source tool.
- OpenFOAM support is an ingestion layer, not a hard dependency for execution.
- Future CFD backend can be added via solver plugin/adapter interface rather than platform rewrite.
