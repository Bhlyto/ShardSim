from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.adaptive import AdaptivePreviewPipeline
from shardsim.contracts import Fidelity
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.refinement import HeatLocalRefiner, select_refinement_regions
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.surrogates.mean_delta import MeanDeltaSurrogate


class LocalRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = HeatEquationSolver()
        self.plan = FidelityPlan(coarse_shape=(17, 17), nominal_shape=(65, 65))
        self.case = self.make_case("local-evaluation", (0.5, 0.5), 0.02)
        self.reference = BootstrapPipeline(self.solver, self.plan).run_case(self.case)

    @staticmethod
    def make_case(case_id: str, center: tuple[float, float], alpha: float):
        return make_heat_case(
            case_id=case_id,
            alpha=alpha,
            t_end=0.04,
            initial_field=gaussian_initial_field(
                (65, 65),
                center=center,
                sigma=(0.08, 0.08),
            ),
        )

    def test_trace_final_state_matches_regular_solver(self) -> None:
        trace = self.solver.solve_trace(self.case, Fidelity.COARSE, self.plan.coarse_shape)
        result = self.solver.solve(self.case, Fidelity.COARSE, self.plan.coarse_shape)
        np.testing.assert_allclose(trace.final_result().field, result.field, atol=1e-12)
        self.assertEqual(trace.field_at(0.5 * trace.t_end, self.plan.nominal_shape).shape, (65, 65))

    def test_region_selector_returns_non_overlapping_cores(self) -> None:
        score_map = np.zeros(self.plan.nominal_shape)
        score_map[10:20, 10:20] = 2.0
        score_map[42:52, 42:52] = 1.0
        regions = select_refinement_regions(
            score_map,
            tile_shape=(16, 16),
            max_regions=2,
            halo=4,
        )
        self.assertEqual(len(regions), 2)
        first, second = regions
        row_overlap = first.row_start < second.row_end and second.row_start < first.row_end
        column_overlap = (
            first.column_start < second.column_end
            and second.column_start < first.column_end
        )
        self.assertFalse(row_overlap and column_overlap)

    def test_local_refinement_changes_only_selected_core_and_reduces_error(self) -> None:
        trace = self.solver.solve_trace(self.case, Fidelity.COARSE, self.plan.coarse_shape)
        result = HeatLocalRefiner(self.solver).refine(
            case=self.case,
            base_field=self.reference.coarse_on_nominal,
            score_map=self.reference.error_map,
            coarse_trace=trace,
            nominal_shape=self.plan.nominal_shape,
            tile_shape=(16, 16),
            max_regions=1,
            halo=6,
        )
        self.assertEqual(len(result.regions), 1)
        region = result.regions[0]
        selected = np.zeros(self.plan.nominal_shape, dtype=bool)
        selected[region.core_slice] = True
        np.testing.assert_allclose(result.merged_field[~selected], result.base_field[~selected])

        base_error = np.mean(
            np.abs(result.base_field[region.core_slice] - self.reference.nominal.field[region.core_slice])
        )
        refined_error = np.mean(
            np.abs(
                result.merged_field[region.core_slice]
                - self.reference.nominal.field[region.core_slice]
            )
        )
        self.assertLess(refined_error, base_error)
        self.assertLess(result.local_compute_fraction, 1.0)
        self.assertLess(result.estimated_total_compute_fraction, 1.0)

    def test_adaptive_preview_reports_accuracy_and_cost(self) -> None:
        training_cases = [
            self.make_case("train-0", (0.45, 0.45), 0.018),
            self.make_case("train-1", (0.55, 0.45), 0.020),
            self.make_case("train-2", (0.45, 0.55), 0.022),
            self.make_case("train-3", (0.55, 0.55), 0.024),
        ]
        bootstrap = BootstrapPipeline(self.solver, self.plan)
        surrogate = MeanDeltaSurrogate()
        surrogate.fit([bootstrap.run_case(case) for case in training_cases])
        pipeline = AdaptivePreviewPipeline(
            solver=self.solver,
            surrogate=surrogate,
            plan=self.plan,
            tile_shape=(16, 16),
            max_regions=1,
            halo=6,
        )
        adaptive_preview = pipeline.run(self.case)
        validation = pipeline.validate(self.case, adaptive_preview)
        self.assertEqual(adaptive_preview.field.shape, self.plan.nominal_shape)
        self.assertIn("adaptive_mae", validation.metrics)
        self.assertIn("estimated_total_compute_fraction", validation.metrics)
        self.assertLess(validation.metrics["estimated_total_compute_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
