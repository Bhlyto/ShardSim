from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from shardsim.contracts import Fidelity, SimulationCase, SimulationResult
from shardsim.metrics import compare_fields
from shardsim.pipeline import FidelityPlan
from shardsim.preview import PreviewPipeline, PreviewResult
from shardsim.refinement import HeatLocalRefiner, LocalRefinementResult
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.solvers.base import Solver
from shardsim.surrogates.base import DeltaSurrogate


@dataclass(frozen=True, slots=True)
class AdaptivePreviewResult:
    preview: PreviewResult
    refinement: LocalRefinementResult

    @property
    def field(self) -> np.ndarray:
        return self.refinement.merged_field


@dataclass(frozen=True, slots=True)
class AdaptivePreviewValidation:
    adaptive_preview: AdaptivePreviewResult
    nominal: SimulationResult
    metrics: Mapping[str, float]


class AdaptivePreviewPipeline:
    def __init__(
        self,
        solver: HeatEquationSolver,
        surrogate: DeltaSurrogate,
        plan: FidelityPlan | None = None,
        tile_shape: tuple[int, int] = (16, 16),
        max_regions: int = 4,
        halo: int = 4,
        min_score: float = 0.0,
        nominal_solver: Solver | None = None,
    ) -> None:
        self.solver = solver
        self.nominal_solver = nominal_solver or solver
        self.surrogate = surrogate
        self.plan = plan or FidelityPlan()
        self.tile_shape = tile_shape
        self.max_regions = max_regions
        self.halo = halo
        self.min_score = min_score
        self.preview_pipeline = PreviewPipeline(
            solver,
            surrogate,
            self.plan,
            nominal_solver=self.nominal_solver,
        )
        self.local_refiner = HeatLocalRefiner(solver)

    def run(
        self,
        case: SimulationCase,
        score_map: np.ndarray | None = None,
    ) -> AdaptivePreviewResult:
        coarse_trace = self.solver.solve_trace(case, Fidelity.COARSE, self.plan.coarse_shape)
        preview = self.preview_pipeline.preview_from_coarse(case, coarse_trace.final_result())
        refinement = self.local_refiner.refine(
            case=case,
            base_field=preview.prediction.mean,
            score_map=score_map if score_map is not None else preview.prediction.uncertainty,
            coarse_trace=coarse_trace,
            nominal_shape=self.plan.nominal_shape,
            tile_shape=self.tile_shape,
            max_regions=self.max_regions,
            halo=self.halo,
            min_score=self.min_score,
        )
        return AdaptivePreviewResult(preview=preview, refinement=refinement)

    def validate(
        self,
        case: SimulationCase,
        adaptive_preview: AdaptivePreviewResult | None = None,
    ) -> AdaptivePreviewValidation:
        result = adaptive_preview or self.run(case)
        if result.preview.case_id != case.case_id:
            raise ValueError("Adaptive preview and validation case identifiers differ.")

        nominal = self.nominal_solver.solve(case, Fidelity.NOMINAL, self.plan.nominal_shape)
        adaptive_metrics = compare_fields(result.field, nominal.field)
        preview_metrics = compare_fields(result.preview.prediction.mean, nominal.field)
        coarse_metrics = compare_fields(result.preview.coarse_on_nominal, nominal.field)
        metrics = {
            **{f"adaptive_{name}": value for name, value in adaptive_metrics.items()},
            **{f"preview_{name}": value for name, value in preview_metrics.items()},
            **{f"coarse_{name}": value for name, value in coarse_metrics.items()},
            "refined_domain_fraction": result.refinement.refined_domain_fraction,
            "local_compute_fraction": result.refinement.local_compute_fraction,
            "estimated_total_compute_fraction": (
                result.refinement.estimated_total_compute_fraction
            ),
        }
        return AdaptivePreviewValidation(
            adaptive_preview=result,
            nominal=nominal,
            metrics=metrics,
        )
