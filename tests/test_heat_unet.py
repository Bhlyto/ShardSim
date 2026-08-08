from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.campaign import ReproducibleCampaign
from shardsim.design import HeatDesignSpace
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.solvers.heat import HeatEquationSolver


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Install the optional cnn dependency to run CNN tests.")
class HeatResidualUNetTests(unittest.TestCase):
    def test_artifact_round_trip_preserves_prediction(self) -> None:
        from shardsim.surrogates.heat_unet import HeatResidualUNetSurrogate

        space = HeatDesignSpace(initial_shape=(17, 17))
        cases = space.sample(5, seed=13, prefix="cnn")
        pipeline = BootstrapPipeline(
            HeatEquationSolver(), FidelityPlan((9, 9), (17, 17))
        )
        references = tuple(pipeline.run_case(case) for case in cases[:4])
        holdout_reference = pipeline.run_case(cases[4])
        model = HeatResidualUNetSurrogate(
            width=4,
            epochs=2,
            batch_size=2,
            seed=17,
            device="cpu",
        )
        descriptor = model.fit(references)
        before = model.predict(cases[4], holdout_reference.coarse_on_nominal)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "heat-unet.pt")
            loaded = HeatResidualUNetSurrogate.load(path)
            after = loaded.predict(cases[4], holdout_reference.coarse_on_nominal)
        np.testing.assert_allclose(before.mean, after.mean, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            before.uncertainty, after.uncertainty, rtol=0.0, atol=0.0
        )
        self.assertEqual(descriptor.metadata["algorithm"], "residual-unet-2d")
        self.assertEqual(descriptor.metadata["epochs_trained"], 2)

    def test_cold_training_is_content_reproducible(self) -> None:
        from shardsim.campaign.core import _model_artifact_content_sha256
        from shardsim.surrogates.heat_unet import HeatResidualUNetSurrogate

        space = HeatDesignSpace(initial_shape=(17, 17))
        cases = space.sample(4, seed=31, prefix="repro-cnn")
        pipeline = BootstrapPipeline(
            HeatEquationSolver(), FidelityPlan((9, 9), (17, 17))
        )
        references = tuple(pipeline.run_case(case) for case in cases)
        first = HeatResidualUNetSurrogate(
            width=4, epochs=2, batch_size=2, seed=19, device="cpu"
        )
        second = HeatResidualUNetSurrogate(
            width=4, epochs=2, batch_size=2, seed=19, device="cpu"
        )
        first.fit(references)
        second.fit(references)
        with tempfile.TemporaryDirectory() as directory:
            first_path = first.save(Path(directory) / "first.pt")
            second_path = second.save(Path(directory) / "second.pt")
            self.assertEqual(
                _model_artifact_content_sha256(first_path),
                _model_artifact_content_sha256(second_path),
            )

    def test_campaign_warm_starts_same_cnn_lineage_with_new_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ReproducibleCampaign.initialize(
                Path(directory) / "campaign",
                name="cnn-lineage",
                seed=23,
            )
            spec = json.loads(campaign.spec_path.read_text(encoding="utf-8"))
            spec["solvers"]["nominal"] = "internal"
            spec["fidelity"] = {"coarse_shape": [9, 9], "nominal_shape": [17, 17]}
            spec["design"]["initial_shape"] = [17, 17]
            spec["design"]["families"]["train-centered"]["count"] = 5
            spec["design"]["families"]["validation-centered"]["count"] = 2
            spec["design"]["families"]["test-corner-nw"]["count"] = 1
            spec["design"]["families"]["test-fast"]["count"] = 1
            campaign.spec_path.write_text(
                json.dumps(spec, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            campaign.lock()
            campaign.run(split="train", limit=4)
            options = {
                "width": 4,
                "epochs": 2,
                "batch_size": 2,
                "device": "cpu",
            }
            first = campaign.train(
                allow_partial=True,
                algorithm="heat-residual-unet",
                model_overrides=options,
            )
            self.assertEqual(first["checkpoint_index"], 1)
            self.assertEqual(first["training_mode"], "cumulative-cold-start")
            self.assertTrue(first["artifact_path"].endswith(".pt"))

            campaign.run(split="train")
            second = campaign.train(
                algorithm="heat-residual-unet",
                model_overrides=options,
            )
            self.assertEqual(second["lineage_id"], first["lineage_id"])
            self.assertEqual(second["checkpoint_index"], 2)
            self.assertEqual(second["training_mode"], "cumulative-warm-start")
            self.assertEqual(
                second["parent_reproducibility_key"], first["reproducibility_key"]
            )
            self.assertEqual(len(second["new_training_case_ids"]), 1)
            repeated = campaign.train(
                algorithm="heat-residual-unet",
                model_overrides=options,
            )
            self.assertEqual(
                repeated["reproducibility_key"], second["reproducibility_key"]
            )

            campaign.run(split="validation")
            evaluation = campaign.evaluate(split="validation")
            self.assertEqual(
                evaluation["model_reproducibility_key"],
                second["reproducibility_key"],
            )
            self.assertIn("mean_gradient_relative_l2", evaluation["metrics"])
            completed_before = campaign.status().completed_cases
            local = campaign.train(algorithm="heat-local-residual")
            self.assertEqual(local["algorithm"], "heat-local-residual")
            self.assertNotEqual(local["lineage_id"], second["lineage_id"])
            self.assertEqual(campaign.status().completed_cases, completed_before)
            self.assertEqual(len(campaign.list_models()), 3)


if __name__ == "__main__":
    unittest.main()
