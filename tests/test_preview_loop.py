from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.contracts import Fidelity
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.preview import PreviewPipeline, PreviewPolicy
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.surrogates.mean_delta import MeanDeltaSurrogate


class CountingSolver:
    def __init__(self) -> None:
        self.delegate = HeatEquationSolver()
        self.calls: list[Fidelity] = []

    def solve(self, case, fidelity, grid_shape):
        self.calls.append(fidelity)
        return self.delegate.solve(case, fidelity, grid_shape)


class PreviewLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = FidelityPlan(coarse_shape=(17, 17), nominal_shape=(65, 65))
        self.training_solver = HeatEquationSolver()
        self.bootstrap = BootstrapPipeline(self.training_solver, self.plan)
        self.training_cases = [
            self.make_case("train-0", (0.45, 0.45), 0.018),
            self.make_case("train-1", (0.55, 0.45), 0.020),
            self.make_case("train-2", (0.45, 0.55), 0.022),
            self.make_case("train-3", (0.55, 0.55), 0.024),
        ]
        references = [self.bootstrap.run_case(case) for case in self.training_cases]
        self.surrogate = MeanDeltaSurrogate()
        self.descriptor = self.surrogate.fit(references)

    @staticmethod
    def make_case(case_id: str, center: tuple[float, float], alpha: float):
        return make_heat_case(
            case_id=case_id,
            alpha=alpha,
            t_end=0.04,
            initial_field=gaussian_initial_field((65, 65), center=center, sigma=(0.08, 0.08)),
        )

    def test_model_descriptor_captures_domain_and_training_scope(self) -> None:
        self.assertEqual(self.descriptor.domain, "heat-2d")
        self.assertEqual(self.descriptor.input_shape, self.plan.nominal_shape)
        self.assertEqual(len(self.descriptor.training_case_ids), len(self.training_cases))

    def test_preview_runs_coarse_before_nominal_validation(self) -> None:
        counting_solver = CountingSolver()
        pipeline = PreviewPipeline(counting_solver, self.surrogate, self.plan)
        case = self.make_case("evaluation", (0.51, 0.49), 0.021)

        preview = pipeline.preview(case)
        self.assertEqual(counting_solver.calls, [Fidelity.COARSE])
        self.assertEqual(preview.prediction.mean.shape, self.plan.nominal_shape)
        self.assertTrue(np.all(preview.prediction.uncertainty >= 0))

        validation = pipeline.validate(case, preview)
        self.assertEqual(counting_solver.calls, [Fidelity.COARSE, Fidelity.NOMINAL])
        self.assertIn("preview_mae", validation.metrics)
        self.assertIn("coarse_mae", validation.metrics)
        self.assertIn("coverage_2sigma", validation.metrics)

    def test_policy_rejects_prediction_above_ood_threshold(self) -> None:
        pipeline = PreviewPipeline(self.training_solver, self.surrogate, self.plan)
        case = self.make_case("far-case", (0.1, 0.1), 0.08)
        preview = pipeline.preview(case)
        self.assertGreater(preview.prediction.ood_score, 0.0)
        strict_policy = PreviewPolicy(max_ood_score=0.5 * preview.prediction.ood_score)
        self.assertFalse(strict_policy.accepts(preview.prediction))

    def test_unfitted_model_cannot_predict(self) -> None:
        model = MeanDeltaSurrogate()
        case = self.make_case("unfitted", (0.5, 0.5), 0.02)
        with self.assertRaises(RuntimeError):
            model.predict(case, np.zeros(self.plan.nominal_shape))

    def test_model_artifact_round_trip_preserves_prediction(self) -> None:
        case = self.make_case("round-trip", (0.51, 0.49), 0.021)
        pipeline = PreviewPipeline(self.training_solver, self.surrogate, self.plan)
        before = pipeline.preview(case).prediction
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surrogate.npz"
            self.surrogate.save(path)
            loaded = MeanDeltaSurrogate.load(path)
            after = PreviewPipeline(self.training_solver, loaded, self.plan).preview(case).prediction
        np.testing.assert_allclose(after.mean, before.mean)
        np.testing.assert_allclose(after.uncertainty, before.uncertainty)
        self.assertAlmostEqual(after.ood_score, before.ood_score)


if __name__ == "__main__":
    unittest.main()
