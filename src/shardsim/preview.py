from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from shardsim.canonical import FieldLocation
from shardsim.contracts import Fidelity, Prediction, SimulationCase, SimulationResult
from shardsim.interpolation import bilinear_resample
from shardsim.metrics import boundary_residual, compare_fields
from shardsim.pipeline import FidelityPlan
from shardsim.solvers.base import Solver
from shardsim.surrogates.base import DeltaSurrogate


@dataclass(frozen=True, slots=True)
class PreviewPolicy:
    max_ood_score: float = 3.0
    max_mean_uncertainty: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(float(self.max_ood_score)) or self.max_ood_score < 0:
            raise ValueError("max_ood_score must be finite and non-negative.")
        if self.max_mean_uncertainty is not None and (
            not isfinite(float(self.max_mean_uncertainty)) or self.max_mean_uncertainty < 0
        ):
            raise ValueError("max_mean_uncertainty must be finite and non-negative.")

    def accepts(self, prediction: Prediction) -> bool:
        if prediction.ood_score > self.max_ood_score:
            return False
        if self.max_mean_uncertainty is None:
            return True
        return float(np.mean(prediction.uncertainty)) <= self.max_mean_uncertainty


@dataclass(frozen=True, slots=True)
class PreviewResult:
    case_id: str
    coarse: SimulationResult
    coarse_on_nominal: np.ndarray
    prediction: Prediction


@dataclass(frozen=True, slots=True)
class PreviewValidation:
    preview: PreviewResult
    nominal: SimulationResult
    metrics: Mapping[str, float]


class PreviewPipeline:
    def __init__(
        self,
        solver: Solver,
        surrogate: DeltaSurrogate,
        plan: FidelityPlan | None = None,
        nominal_solver: Solver | None = None,
    ) -> None:
        self.solver = solver
        self.coarse_solver = solver
        self.nominal_solver = nominal_solver or solver
        self.surrogate = surrogate
        self.plan = plan or FidelityPlan()

    def preview(self, case: SimulationCase) -> PreviewResult:
        coarse = self.coarse_solver.solve(case, Fidelity.COARSE, self.plan.coarse_shape)
        return self.preview_from_coarse(case, coarse)

    def preview_from_coarse(
        self,
        case: SimulationCase,
        coarse: SimulationResult,
    ) -> PreviewResult:
        descriptor = self.surrogate.descriptor
        if case.problem.domain != descriptor.domain or case.problem.equation != descriptor.equation:
            raise ValueError("The surrogate is not compatible with this problem.")
        if case.problem.schema_version != descriptor.schema_version:
            raise ValueError("The surrogate schema is not compatible with this problem.")
        if self.plan.nominal_shape != descriptor.input_shape:
            raise ValueError("The fidelity plan does not match the surrogate grid shape.")
        if coarse.case_id != case.case_id:
            raise ValueError("The coarse result belongs to a different case.")
        if coarse.fidelity is not Fidelity.COARSE:
            raise ValueError("Preview construction requires a coarse result.")
        if coarse.grid_shape != self.plan.coarse_shape:
            raise ValueError("The coarse result does not match the fidelity plan.")

        coarse_on_nominal = bilinear_resample(coarse.field, self.plan.nominal_shape)
        prediction = self.surrogate.predict(case, coarse_on_nominal)
        return PreviewResult(
            case_id=case.case_id,
            coarse=coarse,
            coarse_on_nominal=coarse_on_nominal,
            prediction=prediction,
        )

    def validate(
        self,
        case: SimulationCase,
        preview: PreviewResult | None = None,
    ) -> PreviewValidation:
        preview_result = preview or self.preview(case)
        if preview_result.case_id != case.case_id:
            raise ValueError("Preview and validation case identifiers differ.")

        nominal = self.nominal_solver.solve(case, Fidelity.NOMINAL, self.plan.nominal_shape)
        return self.validate_against_nominal(case, preview_result, nominal)

    def validate_against_nominal(
        self,
        case: SimulationCase,
        preview: PreviewResult,
        nominal: SimulationResult,
    ) -> PreviewValidation:
        if preview.case_id != case.case_id or nominal.case_id != case.case_id:
            raise ValueError("Preview and nominal result must belong to the requested case.")
        if nominal.fidelity is not Fidelity.NOMINAL:
            raise ValueError("Preview validation requires a nominal result.")
        if nominal.grid_shape != self.plan.nominal_shape:
            raise ValueError("The nominal result does not match the fidelity plan.")
        preview_metrics = compare_fields(preview.prediction.mean, nominal.field)
        coarse_metrics = compare_fields(preview.coarse_on_nominal, nominal.field)
        absolute_error = np.abs(preview.prediction.mean - nominal.field)
        uncertainty = preview.prediction.uncertainty
        metrics = {
            **{f"preview_{name}": value for name, value in preview_metrics.items()},
            **{f"coarse_{name}": value for name, value in coarse_metrics.items()},
            "coverage_1sigma": float(np.mean(absolute_error <= uncertainty)),
            "coverage_2sigma": float(np.mean(absolute_error <= 2.0 * uncertainty)),
            "mean_uncertainty": float(np.mean(uncertainty)),
        }
        if nominal.field_location is FieldLocation.POINT:
            metrics["preview_boundary_residual"] = boundary_residual(
                preview.prediction.mean,
                case.boundaries,
            )
        return PreviewValidation(preview=preview, nominal=nominal, metrics=metrics)
