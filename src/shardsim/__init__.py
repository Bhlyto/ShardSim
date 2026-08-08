"""Stable ShardSim V1 public API.

Advanced campaign, OpenFOAM, refinement, and surrogate modules remain available
through their explicit ``shardsim.<module>`` paths, but are not part of the V1
compatibility contract.
"""

from shardsim.contracts import BoundaryConditions, Fidelity, ProblemSpec, SimulationCase, SimulationResult
from shardsim.execution import inspect_result, run_scenario, run_scenario_file
from shardsim.scenario import (
    MODEL_VERSION,
    SCENARIO_SCHEMA_VERSION,
    Scenario,
    ScenarioValidationError,
    SolverConfig,
    load_scenario,
    parse_scenario,
)
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.version import __version__

__all__ = [
    "BoundaryConditions",
    "Fidelity",
    "HeatEquationSolver",
    "MODEL_VERSION",
    "ProblemSpec",
    "SCENARIO_SCHEMA_VERSION",
    "Scenario",
    "ScenarioValidationError",
    "SimulationCase",
    "SimulationResult",
    "SolverConfig",
    "__version__",
    "inspect_result",
    "load_scenario",
    "parse_scenario",
    "run_scenario",
    "run_scenario_file",
]
