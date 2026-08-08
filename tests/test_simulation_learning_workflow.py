from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.canonical import FieldLocation
from shardsim.contracts import Fidelity, SimulationCase, SimulationResult
from shardsim.dataset import ReferenceDatasetStore
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.pipeline import FidelityPlan
from shardsim.preview import PreviewPolicy
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.workflow import (
    CampaignStopReason,
    LearningCampaignPolicy,
    ModelQualityPolicy,
    SimulationLearningWorkflow,
    VariableRole,
)


class RecordingSolver:
    def __init__(self, solver_id: str) -> None:
        self.solver_id = solver_id
        self.delegate = HeatEquationSolver()
        self.calls: list[tuple[str, Fidelity, tuple[int, int]]] = []

    @property
    def adapter_id(self) -> str:
        return self.solver_id

    @property
    def output_location(self) -> FieldLocation:
        return FieldLocation.POINT

    def supports(self, case: SimulationCase) -> bool:
        return case.problem.domain == "heat-2d"

    def solve(
        self,
        case: SimulationCase,
        fidelity: Fidelity,
        grid_shape: tuple[int, int],
    ) -> SimulationResult:
        self.calls.append((case.case_id, fidelity, grid_shape))
        return self.delegate.solve(case, fidelity, grid_shape)


def make_case(case_id: str, center: tuple[float, float], alpha: float) -> SimulationCase:
    return make_heat_case(
        case_id=case_id,
        alpha=alpha,
        t_end=0.02,
        initial_field=gaussian_initial_field(
            (33, 33),
            center=center,
            sigma=(0.09, 0.09),
        ),
    )


class SimulationLearningWorkflowTests(unittest.TestCase):
    def test_full_cycle_separates_solvers_and_reuses_coarse_results(self) -> None:
        coarse_solver = RecordingSolver("fast-heat")
        nominal_solver = RecordingSolver("nominal-heat")
        plan = FidelityPlan(coarse_shape=(9, 9), nominal_shape=(17, 17))
        bootstrap_cases = (
            make_case("bootstrap-0", (0.40, 0.45), 0.018),
            make_case("bootstrap-1", (0.50, 0.50), 0.020),
            make_case("bootstrap-2", (0.60, 0.55), 0.022),
        )
        candidates = (
            make_case("candidate-0", (0.25, 0.70), 0.026),
            make_case("candidate-1", (0.75, 0.30), 0.032),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ReferenceDatasetStore(root / "dataset", "workflow-test")
            artifact = root / "models" / "heat.npz"
            workflow = SimulationLearningWorkflow(
                coarse_solver=coarse_solver,
                nominal_solver=nominal_solver,
                plan=plan,
                store=store,
                model_artifact=artifact,
                preview_policy=PreviewPolicy(max_ood_score=100.0),
                model_quality_policy=ModelQualityPolicy(
                    max_relative_l2=10.0,
                    max_error_ratio_vs_coarse=10.0,
                    min_coverage_2sigma=0.0,
                ),
            )

            analysis = workflow.analyze(bootstrap_cases[0])
            self.assertEqual(analysis.coarse_solver_id, "fast-heat")
            self.assertEqual(analysis.nominal_solver_id, "nominal-heat")
            self.assertEqual(analysis.outputs[0].location, FieldLocation.POINT)
            self.assertTrue(any(item.name == "alpha" for item in analysis.inputs))
            self.assertTrue(all(item.role is VariableRole.INPUT for item in analysis.inputs))

            bootstrap = workflow.bootstrap(bootstrap_cases)
            self.assertEqual(bootstrap.total_reference_count, 3)
            self.assertEqual(len(bootstrap.new_references), 3)
            self.assertTrue(artifact.is_file())
            self.assertEqual(len(store.case_ids()), 3)
            self.assertEqual(
                [fidelity for _, fidelity, _ in coarse_solver.calls],
                [Fidelity.COARSE] * 3,
            )
            self.assertEqual(
                [fidelity for _, fidelity, _ in nominal_solver.calls],
                [Fidelity.NOMINAL] * 3,
            )

            evaluation_case = make_case("evaluation", (0.52, 0.48), 0.021)
            evaluation = workflow.evaluate((evaluation_case,))
            self.assertTrue(evaluation.passed)
            self.assertTrue(evaluation.report_path.is_file())
            promoted_preview = workflow.preview(evaluation_case, validate_nominal=False)
            self.assertTrue(promoted_preview.model_promoted)
            self.assertTrue(promoted_preview.accepted_by_policy)
            with self.assertRaises(ValueError):
                workflow.evaluate((bootstrap_cases[0],))

            coarse_before = len(coarse_solver.calls)
            nominal_before = len(nominal_solver.calls)
            iteration = workflow.run_iteration(candidates, selection_count=1)
            self.assertEqual(len(iteration.active_learning.selected), 1)
            self.assertEqual(len(iteration.active_learning.new_references), 1)
            self.assertEqual(len(iteration.active_learning.validations), 1)
            self.assertEqual(iteration.active_learning.total_reference_count, 4)
            self.assertEqual(len(coarse_solver.calls) - coarse_before, 2)
            self.assertEqual(len(nominal_solver.calls) - nominal_before, 1)
            self.assertIn(
                "preview_relative_l2",
                iteration.active_learning.validations[0].metrics,
            )

            runtime_case = make_case("runtime", (0.52, 0.48), 0.021)
            nominal_before = len(nominal_solver.calls)
            preview = workflow.preview(runtime_case, validate_nominal=False)
            self.assertIsNone(preview.validation)
            self.assertFalse(preview.model_promoted)
            self.assertFalse(preview.accepted_by_policy)
            self.assertEqual(len(nominal_solver.calls), nominal_before)

            reloaded = SimulationLearningWorkflow(
                coarse_solver=RecordingSolver("fast-heat"),
                nominal_solver=RecordingSolver("nominal-heat"),
                plan=plan,
                store=store,
            )
            self.assertEqual(len(reloaded.references), 4)
            self.assertEqual(reloaded.preview(runtime_case).preview.case_id, "runtime")

    def test_campaign_reuses_holdout_nominal_until_quality_passes(self) -> None:
        coarse_solver = RecordingSolver("fast-heat")
        nominal_solver = RecordingSolver("nominal-heat")
        plan = FidelityPlan(coarse_shape=(9, 9), nominal_shape=(17, 17))
        workflow = SimulationLearningWorkflow(
            coarse_solver=coarse_solver,
            nominal_solver=nominal_solver,
            plan=plan,
            preview_policy=PreviewPolicy(max_ood_score=100.0),
            model_quality_policy=ModelQualityPolicy(
                max_relative_l2=10.0,
                max_error_ratio_vs_coarse=10.0,
                min_coverage_2sigma=0.0,
            ),
        )
        workflow.bootstrap(
            (
                make_case("bootstrap-a", (0.40, 0.45), 0.018),
                make_case("bootstrap-b", (0.50, 0.50), 0.020),
                make_case("bootstrap-c", (0.60, 0.55), 0.022),
            )
        )
        campaign = workflow.run_campaign(
            candidates=(
                make_case("candidate-a", (0.25, 0.70), 0.028),
                make_case("candidate-b", (0.75, 0.30), 0.032),
            ),
            evaluation_cases=(make_case("holdout", (0.52, 0.48), 0.021),),
            policy=LearningCampaignPolicy(
                selection_count=1,
                max_iterations=2,
                minimum_iterations=1,
            ),
        )
        self.assertEqual(campaign.stop_reason, CampaignStopReason.QUALITY_PASSED)
        self.assertEqual(len(campaign.rounds), 1)
        self.assertEqual(campaign.total_reference_count, 4)
        self.assertEqual(len(nominal_solver.calls), 5)
        self.assertEqual(
            [case_id for case_id, _, _ in nominal_solver.calls].count("holdout"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
