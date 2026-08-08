from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from shardsim.contracts import Fidelity, ProblemSpec, SimulationCase, SimulationResult
from shardsim.interpolation import bilinear_resample
from shardsim.metrics import compare_fields
from shardsim.solvers.base import Solver


@dataclass(frozen=True, slots=True)
class FidelityPlan:
    coarse_shape: tuple[int, int] = (25, 25)
    nominal_shape: tuple[int, int] = (100, 100)

    def __post_init__(self) -> None:
        if len(self.coarse_shape) != 2 or len(self.nominal_shape) != 2:
            raise ValueError("Fidelity shapes must be two-dimensional.")
        if min(self.coarse_shape) < 3 or min(self.nominal_shape) < 3:
            raise ValueError("Fidelity grids must be at least 3x3.")
        if any(coarse >= nominal for coarse, nominal in zip(self.coarse_shape, self.nominal_shape)):
            raise ValueError("The nominal grid must be finer than the coarse grid on every axis.")


@dataclass(frozen=True, slots=True)
class ReferenceSample:
    case_id: str
    problem: ProblemSpec
    coarse: SimulationResult
    nominal: SimulationResult
    coarse_on_nominal: np.ndarray
    delta: np.ndarray
    error_map: np.ndarray
    metrics: Mapping[str, float]
    case_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coarse_on_nominal = np.asarray(self.coarse_on_nominal, dtype=np.float64)
        delta = np.asarray(self.delta, dtype=np.float64)
        error_map = np.asarray(self.error_map, dtype=np.float64)
        expected_shape = self.nominal.grid_shape
        if any(array.shape != expected_shape for array in (coarse_on_nominal, delta, error_map)):
            raise ValueError("Reference arrays must use the nominal grid.")
        if not np.allclose(self.nominal.field, coarse_on_nominal + delta):
            raise ValueError("Reference sample violates nominal = coarse + delta.")


class BootstrapPipeline:
    def __init__(
        self,
        solver: Solver,
        plan: FidelityPlan | None = None,
        nominal_solver: Solver | None = None,
    ) -> None:
        self.solver = solver
        self.coarse_solver = solver
        self.nominal_solver = nominal_solver or solver
        self.plan = plan or FidelityPlan()

    def run_case(self, case: SimulationCase) -> ReferenceSample:
        coarse = self.coarse_solver.solve(case, Fidelity.COARSE, self.plan.coarse_shape)
        return self.complete_reference(case, coarse)

    def complete_reference(
        self,
        case: SimulationCase,
        coarse: SimulationResult,
    ) -> ReferenceSample:
        if coarse.case_id != case.case_id or coarse.fidelity is not Fidelity.COARSE:
            raise ValueError("A reference requires the matching coarse simulation result.")
        if coarse.grid_shape != self.plan.coarse_shape:
            raise ValueError("The coarse result does not match the fidelity plan.")
        nominal = self.nominal_solver.solve(case, Fidelity.NOMINAL, self.plan.nominal_shape)
        return self.reference_from_results(case, coarse, nominal)

    def reference_from_results(
        self,
        case: SimulationCase,
        coarse: SimulationResult,
        nominal: SimulationResult,
    ) -> ReferenceSample:
        if coarse.case_id != case.case_id or nominal.case_id != case.case_id:
            raise ValueError("Reference results must belong to the requested case.")
        if coarse.fidelity is not Fidelity.COARSE or nominal.fidelity is not Fidelity.NOMINAL:
            raise ValueError("Reference results have invalid fidelity labels.")
        if coarse.grid_shape != self.plan.coarse_shape:
            raise ValueError("The coarse result does not match the fidelity plan.")
        if nominal.grid_shape != self.plan.nominal_shape:
            raise ValueError("The nominal result does not match the fidelity plan.")
        if not np.isclose(coarse.t_end, nominal.t_end) or not np.isclose(
            nominal.t_end,
            case.problem.t_end,
        ):
            raise ValueError("Coarse and nominal results must reach the requested physical horizon.")
        coarse_on_nominal = bilinear_resample(coarse.field, nominal.grid_shape)
        delta = nominal.field - coarse_on_nominal
        error_map = np.abs(delta)
        return ReferenceSample(
            case_id=case.case_id,
            problem=case.problem,
            coarse=coarse,
            nominal=nominal,
            coarse_on_nominal=coarse_on_nominal,
            delta=delta,
            error_map=error_map,
            metrics=compare_fields(coarse_on_nominal, nominal.field),
            case_metadata=case.metadata,
        )
