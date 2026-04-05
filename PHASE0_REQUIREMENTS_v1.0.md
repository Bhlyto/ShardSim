# Phase 0 Requirements Specification v1.0

Date: 2026-04-04
Project: ShardSim / ChatMPI
Status: Baseline approved for Phase 1 kickoff

## 1. Scope and Intent
This specification defines the frozen Phase 0 requirements for the first operational version of ShardSim/ChatMPI.

In scope:
- Thermal domain only
- Transient heat simulation
- Distributed multi-fidelity execution
- Decision-core-driven adaptive refinement
- Domain-specific surrogate assistance with safety fallback

Out of scope (v1):
- Cross-domain surrogate reuse (e.g., thermal -> CFD)
- Universal surrogate across physics domains

## 2. Primary Objective
Achieve at least 30% runtime reduction versus full-fine baseline while preserving acceptable error relative to full-fine reference.

## 3. Technical Baseline Decisions

### 3.1 Physics and Numerics
- Governing class: transient heat equation
- Numerical method: finite difference method (FDM)
- Boundary conditions: mixed boundary conditions required
- Mesh: non-uniform mesh required

### 3.2 Fidelity and Adaptivity
- Coarse/fine operating ratio target: 5-10% (tolerance-driven)
- Refinement score inputs:
  - local estimated numerical error
  - predictive uncertainty

### 3.3 Distribution and Scheduling
- Partitioning policy: strict geometric partitioning
- Runtime policy: dynamic task reassignment by decision core

### 3.4 Ground Truth and Learning Constraints
- Reference truth: full-fine simulation only
- Surrogate outputs:
  - correction field prediction
  - direct fine-field prediction
- Safety contract:
  - uncertainty-bounded usage
  - automatic fallback to physics-only mode when policy is violated

### 3.5 Reproducibility and Observability
- Deterministic solver mode: required
- Observability: full operational telemetry required (runtime, communication, orchestration, refinement behavior, estimator reliability)

### 3.6 Data Contract Requirements
Data interfaces must include:
- field tensor format
- metadata
- units
- coordinate convention
- checkpoint schema

## 4. KPI Targets (Baseline)
- Runtime reduction target: >= 30%
- Max memory per node: 16 GB
- Max wall-clock per scenario: configurable per case via configuration

Pending KPI values (using temporary defaults):
- Global error bound E_max: default 2.0% normalized global error
- Local refinement threshold tau: default 5.0% local relative error
- Parallel efficiency target: default >= 70% at 8 nodes

## 5. Platform Portability Baseline (Temporary)
Until final matrix is approved, baseline target is:
- Ubuntu 22.04 LTS
- GCC 12+
- OpenMPI 5.x
- CMake 3.24+

## 6. Safety Fallback Policy (Temporary)
Default fallback trigger:
- switch to physics-only mode when predictive uncertainty > 0.20 for 3 consecutive iterations

This policy must be externalized to configuration.

## 7. Acceptance Criteria for Phase 1 Entry
Phase 1 may start when:
- module interface contracts are documented
- deterministic mode is testable
- telemetry schema is defined
- configuration supports wall-clock policy and fallback policy
- benchmark protocol defines full-fine comparator scenarios

## 8. Open Items to Finalize (Non-Blocking)
- Final E_max and tau values
- Final node count and hardware profile for efficiency target
- Final portability matrix details
- Final fallback threshold/hysteresis parameters

## 9. Governance Rule
No requirement in this document may be changed during Phase 1 without explicit change note in an ADR.
