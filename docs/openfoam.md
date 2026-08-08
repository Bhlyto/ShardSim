# OpenFOAM nominal solver

ShardSim treats OpenFOAM as a versioned solver backend, not as its canonical data model.

## Reproducibility contract

- Distribution: OpenCFD OpenFOAM
- Release: `2606`
- Runtime image: `opencfd/openfoam-run:2606`
- Pinned image digest: `sha256:4229997e74defb81548222d511b8e3b95b98305e5df41b8e88b031813fe47eeb`
- Heat application: `laplacianFoam`
- Field location: finite-volume cell centres

The adapter records the complete image reference, solver name, spatial steps, temporal step,
runtime, and the tail of the solver log in `SimulationResult.metadata`.

## Canonical boundary

`ProblemSpec` and `SimulationCase` remain solver-independent. The adapter performs these steps:

1. sample the structured initial field at finite-volume cell centres;
2. emit the OpenFOAM `0`, `constant`, and `system` dictionaries;
3. run `blockMesh` and `laplacianFoam` in an ephemeral Docker container;
4. parse the final ASCII `T` field into a top-first NumPy array;
5. return a normal `SimulationResult`.

Canonical scientific fields use seven SI base-dimension exponents ordered as mass, length,
time, temperature, amount of substance, electric current, and luminous intensity. ML scaling
is a separate transformation and must be fitted only on training data.

## Verification benchmark

The first verification problem is the exact sine mode on a rectangular domain with homogeneous
Dirichlet boundaries:

```text
T(x,y,t) = sin(pi*x/Lx) sin(pi*y/Ly)
             exp(-alpha*pi^2*(1/Lx^2 + 1/Ly^2)*t)
```

`examples/heat_openfoam_verification.py` runs grid refinement for both the internal solver and
OpenFOAM. It reports relative L2 error, maximum error, energy error, observed convergence order,
cross-solver discrepancy, runtime, and time-step count.

Run it with:

```powershell
$env:PYTHONPATH = "src"
python examples/heat_openfoam_verification.py
```

Run the complete internal-coarse/OpenFOAM-nominal learning loop with:

```powershell
$env:PYTHONPATH = "src"
python examples/heat_openfoam_learning_workflow.py
```

The example uses separate Latin Hypercube designs for bootstrap, active-learning candidates, and
holdout evaluation, then trains the equation-specific `HeatLocalResidualSurrogate`.

Docker integration tests are opt-in:

```powershell
$env:SHARDSIM_RUN_OPENFOAM_TESTS = "1"
python -m pytest tests/test_openfoam_foundation.py
```
