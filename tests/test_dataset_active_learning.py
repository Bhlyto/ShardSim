from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.active_learning import ActiveLearningLoop, ActiveLearningPolicy
from shardsim.contracts import Fidelity
from shardsim.dataset import ReferenceDatasetStore
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.preview import PreviewPipeline
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.surrogates.mean_delta import MeanDeltaSurrogate


class CountingSolver:
    def __init__(self) -> None:
        self.delegate = HeatEquationSolver()
        self.calls: list[Fidelity] = []

    def solve(self, case, fidelity, grid_shape):
        self.calls.append(fidelity)
        return self.delegate.solve(case, fidelity, grid_shape)


class DatasetAndActiveLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = HeatEquationSolver()
        self.plan = FidelityPlan(coarse_shape=(17, 17), nominal_shape=(65, 65))
        self.bootstrap = BootstrapPipeline(self.solver, self.plan)
        self.training_cases = [
            self.make_case("train-0", (0.45, 0.45), 0.018),
            self.make_case("train-1", (0.55, 0.45), 0.020),
            self.make_case("train-2", (0.45, 0.55), 0.022),
            self.make_case("train-3", (0.55, 0.55), 0.024),
        ]
        self.references = [self.bootstrap.run_case(case) for case in self.training_cases]

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

    def test_reference_dataset_round_trip_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReferenceDatasetStore(Path(directory) / "dataset", dataset_id="heat-tests")
            for sample in self.references[:2]:
                store.add(sample)
            self.assertEqual(store.case_ids(), ("train-0", "train-1"))
            loaded = store.load_all()

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].problem.domain, "heat-2d")
        np.testing.assert_allclose(loaded[0].coarse.field, self.references[0].coarse.field)
        np.testing.assert_allclose(loaded[0].nominal.field, self.references[0].nominal.field)
        np.testing.assert_allclose(loaded[0].delta, self.references[0].delta)

    def test_reference_dataset_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReferenceDatasetStore(Path(directory) / "dataset", dataset_id="heat-tests")
            path = store.add(self.references[0])
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(ValueError):
                store.load_all()

    def test_active_learning_prioritizes_ood_and_retrains(self) -> None:
        counting_solver = CountingSolver()
        surrogate = MeanDeltaSurrogate()
        preview_pipeline = PreviewPipeline(counting_solver, surrogate, self.plan)
        candidates = [
            self.make_case("candidate-near", (0.50, 0.50), 0.021),
            self.make_case("candidate-shifted", (0.30, 0.70), 0.026),
            self.make_case("candidate-ood", (0.50, 0.50), 0.080),
        ]
        with tempfile.TemporaryDirectory() as directory:
            store = ReferenceDatasetStore(Path(directory) / "dataset", dataset_id="active-heat")
            loop = ActiveLearningLoop(
                bootstrap_pipeline=self.bootstrap,
                preview_pipeline=preview_pipeline,
                surrogate=surrogate,
                references=self.references,
                policy=ActiveLearningPolicy(
                    ood_weight=1.0,
                    uncertainty_weight=0.0,
                    diversity_weight=0.0,
                ),
                store=store,
            )
            ranked = loop.select(candidates, count=len(candidates))
            self.assertEqual(ranked[0].ood_score, max(item.ood_score for item in ranked))
            self.assertNotEqual(ranked[0].case.case_id, "candidate-near")
            self.assertEqual(counting_solver.calls, [Fidelity.COARSE] * len(candidates))

            iteration = loop.enrich(ranked[:1])
            self.assertEqual(iteration.total_reference_count, 5)
            self.assertIn(ranked[0].case.case_id, iteration.model.training_case_ids)
            self.assertEqual(len(store.case_ids()), 5)


if __name__ == "__main__":
    unittest.main()
