# ShardSim
DomainSim: A Distributed Multi-Fidelity Simulation Framework with Domain-Specific Learning

# ChatMPI: a distributed multi-fidelity framework for adaptive scientific simulation with domain-specialized learning

## Abstract

We propose a conceptual framework for distributed scientific computing, named **ChatMPI**, intended to orchestrate multi-fidelity numerical simulations coupled with domain-specialized machine learning models. The central idea is to explicitly separate three layers: (i) a **decision core** responsible for domain decomposition, resource allocation, and triggering refinement; (ii) a **pre-simulation layer** producing approximate fields and an uncertainty map; (iii) a **simulation layer** with variable fidelity, composed of coarse and fine computations. A learning module trains from the discrepancies between approximate results and reference results to improve surrogate models. We also advocate a strong methodological principle: **one ML model per physical domain**. A model trained on thermal problems should not be expected to be valid for CFD, structural mechanics, or other domains whose constraints, invariants, and data structures differ deeply. We present a concrete experiment on 2D heat diffusion to evaluate potential gains in computational cost, accuracy, and overall efficiency.

**Keywords:** numerical simulation, HPC, MPI, multi-fidelity, surrogate model, error estimation, pre-simulation, scientific learning.

---

## 1. Introduction

High-fidelity numerical simulations are a pillar of engineering sciences. They enable the study of complex physical phenomena with high accuracy, but at the price of significant computational cost. In many cases, computational resources are poorly allocated: some regions of the domain require fine resolution, while others can be treated coarsely without significant degradation of overall result quality.

Multi-fidelity and adaptive mesh refinement approaches have been widely studied to reduce this cost. At the same time, machine learning methods have shown their usefulness as surrogates or substitution models, able to quickly predict physical fields or error corrections. However, attempts at excessive generalization across multiple physical domains often lead to models that are not robust, hard to interpret, and sensitive to violations of physical constraints.

The ChatMPI project formalizes a hybrid architecture that combines:
- a central decision engine;
- distributed simulation;
- a fast pre-simulation;
- specialized supervised learning;
- a feedback loop enabling adaptive refinement.

The goal is not to replace the physical solver with a language model or a generic model, but to build a scientific infrastructure oriented around decision-making, where learning guides the simulation rather than fully absorbing it.

---

## 2. Guiding principle: one ML model per physical domain

A structuring hypothesis of the proposed framework is the following:

> **A machine learning model must be specialized for a given physical domain.**

In other words, a thermal surrogate is not expected to be a CFD surrogate, and a model trained on structural mechanics should not be used as an approximation for turbulence or compressible flow.

This position is based on several observations:
1. The mathematical structures of physical phenomena differ significantly.
2. State variables, constraints, and symmetries are not the same.
3. The training data distribution depends on the solver, boundary conditions, and geometry.
4. Uncontrolled inter-domain generalization can produce non-physical outputs.

Thus, ChatMPI does not seek a universal model, but a **constellation of specialized models**, each operating within its domain of expertise.

---

## 3. Overall architecture of the framework

The system is composed of four main subsystems.

### 3.1 Decision core

The decision core orchestrates everything. Its responsibilities are:
- domain decomposition;
- identification of coarse regions and critical regions;
- allocation of computing resources;
- deciding whether to launch a pre-simulation, a coarse simulation, or a fine simulation;
- estimating a local error or uncertainty;
- deciding on refinement.

The decision core does not necessarily perform the physical computations itself; it acts as a **numerical strategy manager**.

### 3.2 Pre-simulation module

The pre-simulation provides a quick initial approximation of the physical field. It can be obtained by:
- a low-resolution simulation;
- a surrogate model;
- a partial analytical approximation;
- a reduced version of the main solver.

The pre-simulation produces two useful objects:
- an approximate field;
- an uncertainty or potential error map.

### 3.3 Multi-fidelity simulation layer

The simulation layer includes at least two levels:
- **coarse simulation**: low cost, reduced resolution;
- **fine simulation**: high cost, high precision.

The fine simulation is triggered locally on regions deemed critical by the decision core, while the coarse simulation covers less sensitive regions.

### 3.4 Learning module

The ML module receives:
- the approximate results;
- the reference results;
- the observed discrepancies between the two.

It updates a surrogate model dedicated to the studied domain. This model is then reused in future pre-simulation cycles.

---

## 4. Problem formalization

Let a physical domain be $\Omega$ and a state variable $u(x,t)$ defined over $\Omega$. The physical dynamics can be represented in the general form:

$$
\frac{\partial u}{\partial t} = \mathcal{F}(u, x, t)
$$

where $\mathcal{F}$ denotes the physical operator of the problem.

We distinguish:
- $u_f$: high-fidelity solution;
- $u_c$: coarse solution;
- $u_s$: surrogate (substitution) solution.

The local error is defined by:

$$
\varepsilon(x,t) = \left|u_f(x,t) - u_c(x,t)\right|
$$

or, in a global formulation:

$$
E = \|u_f - u_c\|
$$

The decision core aims to identify regions where:

$$
\varepsilon(x,t) > \tau
$$

with $\tau$ a refinement threshold.

The general objective is to minimize computational cost $C$ under a precision constraint:

$$
\min C \quad \text{subject to} \quad E \leq E_{\max}
$$

---

## 5. Operational workflow of the system

The framework operates following an adaptive loop:

1. **Initialization**  
   The decision core receives the physical model and problem parameters.

2. **Pre-simulation**  
   A rapid estimate of the field is produced, accompanied by an uncertainty estimate.

3. **Critical region analysis**  
   The system identifies regions where the estimated error is high.

4. **Resource allocation**  
   computing resources are assigned to coarse or fine simulation according to criticality.

5. **Distributed simulation**  
   Subdomains are solved in parallel.

6. **Aggregation**  
   Partial results are merged.

7. **Comparison and learning**  
   Discrepancies between approximation and reference are used to train or correct the surrogate.

8. **Iteration**  
   The loop restarts with an improved ML model.

---

## 6. Distributed implementation

The framework can be implemented in a distributed computing environment using a message-passing paradigm. Each node handles a subdomain and exchanges boundary information with its neighbors.

In this logic:
- exchanged data are mesh boundaries, local states, or corrections;
- communication cost must remain lower than the parallelization gain;
- strongly interacting regions should be carefully partitioned.

The decision core acts as an orchestrator that gathers local measures, decides on refinements, and redistributes tasks.

---

## 7. Concrete experiment: 2D heat diffusion

### 7.1 Test problem

We consider a square two-dimensional plate subject to thermal diffusion. The heat equation is:

$$
\frac{\partial T}{\partial t} = \alpha \nabla^2 T
$$

where:
- $T$ is temperature;
- $\alpha$ is thermal diffusivity.

### 7.2 Geometry and initial conditions

We consider:
- a 2D square domain;
- fixed boundary conditions on the edges;
- a heat source localized at the center or in a specific region.

This problem is simple enough to be tested quickly, yet rich enough to illustrate:
- the usefulness of a coarse simulation;
- the benefit of local refinement;
- the added value of a thermal surrogate.

### 7.3 Fidelity levels

We define three levels.

**Level 1 — pre-simulation**  
A coarse grid produces an approximate temperature map. This step is fast and can serve as initialization.

**Level 2 — coarse simulation**  
An intermediate resolution provides a more accurate field while keeping moderate cost.

**Level 3 — fine simulation**  
A high-resolution run serves as the local numerical reference.

### 7.4 Role of the ML model

The ML model is trained on pairs:
- input: coarse state, local geometry, boundary conditions;
- output: correction towards the fine field or an estimate of the fine field.

The thermal surrogate is valid only for thermal problems. Its extension to CFD is not expected to be correct without specific retraining.

---

## 8. Experimental protocol

### 8.1 Objectives

The experiment aims to measure:
1. reduction of computational cost;
2. quality of the pre-simulation;
3. gain obtained by local refinement;
4. progressive improvement of the surrogate.

### 8.2 Baselines

Three strategies are compared.

**A. Pure fine baseline**  
The entire grid is computed at high resolution.

**B. Multi-fidelity without learning**  
The coarse grid is complemented locally by fine zones.

**C. Full ChatMPI**  
Pre-simulation + adaptive selection + local fine simulation + surrogate learning.

### 8.3 Metrics

The chosen indicators are:
- **mean absolute error**;
- **global error norm**;
- **total runtime**;
- **number of fine cells computed**;
- **cost/accuracy efficiency**.

We can define a simplified efficiency:

$$
\eta = \frac{\text{accuracy}}{\text{cost}}
$$

### 8.4 Expected hypotheses

We expect that:
- the fine baseline gives the lowest error but the highest cost;
- multi-fidelity reduces cost;
- ChatMPI achieves a superior cost-accuracy tradeoff;
- the surrogate improves over iterations.

---

## 9. Cross-domain generalization experiment

To test the principle “one model per domain”, one can run an intentionally incorrect but instructive experiment: use a thermal surrogate on a CFD problem.

### 9.1 Hypothesis

The thermal model should not produce reliable results on a fluid flow problem because:
- the physical variables differ;
- the structure of the state space is different;
- conservation constraints are not the same.

### 9.2 Expected result

We expect:
- a strong degradation in accuracy;
- non-physical predictions;
- low robustness outside the training distribution.

This experiment serves to methodologically justify model specialization.

---

## 10. Discussion

The ChatMPI framework rests on a clear distinction between:
- **physical computation**;
- **allocation decision**;
- **correction learning**.

This separation brings several advantages:
- better scientific interpretability;
- local adaptation to the problem;
- reduced computational cost;
- better interpretability than a monolithic model.

However, several difficulties remain:
- error estimation must be reliable;
- distributed communications can become costly;
- learning may overfit on too narrow a domain;
- transfer between physical domains should not be assumed.

---

## 11. Limitations

This framework has important limitations:
- it does not replace rigorous physical modeling;
- it depends strongly on the quality of the reference solver;
- it requires a good definition of critical regions;
- it assumes training data are consistent with the treated domain.

In particular, the illusion of a universal model must be avoided. The ML model is not the system's core; it is an auxiliary component serving the simulation.

---

## 12. Conclusion

We proposed a conceptual framework for a distributed adaptive scientific simulation system called ChatMPI. Its architecture relies on a decision core, a pre-simulation, multi-fidelity simulations, and a domain-specialized learning module. The guiding principle is explicit: **an ML model should not be expected to be universal, but as an expert for a given physical domain**.

The proposed experiment on 2D heat diffusion offers a concrete basis to measure gains in accuracy and cost. This framework provides a solid foundation for future research in distributed scientific computing, numerical optimization, and physics-informed learning.

---

## Conceptual references

This document is intentionally written as a pseudo-scientific paper. For a full academic version, it would be appropriate to add a bibliography on:
- high performance computing;
- multi-fidelity methods;
- surrogate models;
- PINNs;
- uncertainty estimation;
- domain partitioning and adaptive numerical schemes.
