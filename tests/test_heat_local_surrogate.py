from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.dataset import ReferenceDatasetStore
from shardsim.design import HeatDesignSpace
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.preview import PreviewPipeline
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.surrogates.heat_local import HeatLocalResidualSurrogate
from shardsim.surrogates.mean_delta import MeanDeltaSurrogate


class HeatDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = HeatDesignSpace(
            alpha=(0.015, 0.030),
            t_end=(0.015, 0.040),
            center_x=(0.25, 0.75),
            center_y=(0.25, 0.75),
            sigma_x=(0.07, 0.13),
            sigma_y=(0.07, 0.13),
            amplitude=(0.8, 1.2),
            initial_shape=(33, 33),
        )

    def test_latin_hypercube_is_reproducible_and_stratified(self) -> None:
        first = self.space.sample(8, seed=11, prefix="design")
        second = self.space.sample(8, seed=11, prefix="design")
        self.assertEqual([case.case_id for case in first], [case.case_id for case in second])
        np.testing.assert_allclose(first[0].initial_field, second[0].initial_field)
        alphas = np.array([case.problem.parameter("alpha") for case in first])
        normalized = (alphas - self.space.alpha[0]) / (
            self.space.alpha[1] - self.space.alpha[0]
        )
        strata = np.floor(normalized * len(first)).astype(int)
        self.assertEqual(set(strata), set(range(len(first))))

    def test_design_metadata_survives_reference_storage(self) -> None:
        case = self.space.sample(1, seed=7, prefix="stored")[0]
        plan = FidelityPlan(coarse_shape=(9, 9), nominal_shape=(17, 17))
        sample = BootstrapPipeline(HeatEquationSolver(), plan).run_case(case)
        with tempfile.TemporaryDirectory() as directory:
            store = ReferenceDatasetStore(Path(directory) / "dataset", "doe-metadata")
            store.add(sample)
            loaded = store.load_all()[0]
        self.assertEqual(loaded.case_metadata["design"], "latin-hypercube")
        self.assertEqual(loaded.case_metadata["design_seed"], 7)


class HeatLocalResidualSurrogateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = HeatDesignSpace(
            alpha=(0.015, 0.030),
            t_end=(0.015, 0.040),
            center_x=(0.25, 0.75),
            center_y=(0.25, 0.75),
            sigma_x=(0.07, 0.13),
            sigma_y=(0.07, 0.13),
            amplitude=(0.8, 1.2),
            initial_shape=(33, 33),
        )
        self.solver = HeatEquationSolver()
        self.plan = FidelityPlan(coarse_shape=(9, 9), nominal_shape=(33, 33))
        bootstrap = BootstrapPipeline(self.solver, self.plan)
        self.training = self.space.sample(8, seed=11, prefix="train")
        self.holdout = self.space.sample(4, seed=29, prefix="holdout")
        self.references = tuple(bootstrap.run_case(case) for case in self.training)

    def test_local_model_improves_holdout_and_mean_delta_baseline(self) -> None:
        local = HeatLocalResidualSurrogate()
        mean_delta = MeanDeltaSurrogate()
        local.fit(self.references)
        mean_delta.fit(self.references)
        local_pipeline = PreviewPipeline(self.solver, local, self.plan)
        mean_pipeline = PreviewPipeline(self.solver, mean_delta, self.plan)
        coarse_errors = []
        local_errors = []
        mean_errors = []
        for case in self.holdout:
            local_validation = local_pipeline.validate(case)
            mean_validation = mean_pipeline.validate(case)
            coarse_errors.append(local_validation.metrics["coarse_relative_l2"])
            local_errors.append(local_validation.metrics["preview_relative_l2"])
            mean_errors.append(mean_validation.metrics["preview_relative_l2"])
        self.assertLess(np.mean(local_errors), 0.85 * np.mean(coarse_errors))
        self.assertLess(np.mean(local_errors), np.mean(mean_errors))

    def test_artifact_round_trip_preserves_prediction(self) -> None:
        model = HeatLocalResidualSurrogate()
        model.fit(self.references)
        pipeline = PreviewPipeline(self.solver, model, self.plan)
        before = pipeline.preview(self.holdout[0]).prediction
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "heat-local.npz")
            loaded = HeatLocalResidualSurrogate.load(path)
            after = PreviewPipeline(self.solver, loaded, self.plan).preview(
                self.holdout[0]
            ).prediction
        np.testing.assert_allclose(before.mean, after.mean)
        np.testing.assert_allclose(before.uncertainty, after.uncertainty)
        self.assertEqual(before.ood_score, after.ood_score)


if __name__ == "__main__":
    unittest.main()
