from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.campaign import (
    CampaignLockError,
    ReproducibleCampaign,
    wide_heat_campaign_spec,
)


def configure_small_internal_campaign(campaign: ReproducibleCampaign) -> None:
    spec = json.loads(campaign.spec_path.read_text(encoding="utf-8"))
    spec["solvers"]["nominal"] = "internal"
    spec["fidelity"] = {"coarse_shape": [9, 9], "nominal_shape": [17, 17]}
    spec["design"]["initial_shape"] = [33, 33]
    families = spec["design"]["families"]
    families["train-centered"]["count"] = 3
    families["validation-centered"]["count"] = 2
    families["test-corner-nw"]["count"] = 1
    families["test-fast"]["count"] = 1
    campaign.spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ReproducibleCampaignTests(unittest.TestCase):
    def test_wide_profile_has_disjoint_massive_splits(self) -> None:
        spec = wide_heat_campaign_spec("wide-heat", 20260715)
        families = spec["design"]["families"]
        counts = {
            split: sum(
                family["count"]
                for family in families.values()
                if family["split"] == split
            )
            for split in ("train", "validation", "test")
        }

        self.assertEqual(counts, {"train": 600, "validation": 80, "test": 80})
        self.assertEqual(spec["model"]["algorithm"], "heat-residual-unet")
        self.assertIn("train-corner-nw", families)
        self.assertIn("train-fast", families)
        self.assertIn("train-long", families)

    def test_seeded_case_lock_is_byte_reproducible_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = ReproducibleCampaign.initialize(
                Path(directory) / "first",
                name="deterministic-heat",
                seed=20260714,
            )
            second = ReproducibleCampaign.initialize(
                Path(directory) / "second",
                name="deterministic-heat",
                seed=20260714,
            )
            configure_small_internal_campaign(first)
            configure_small_internal_campaign(second)
            first_lock = first.lock()
            second_lock = second.lock()

            self.assertEqual(first.cases_path.read_bytes(), second.cases_path.read_bytes())
            self.assertEqual(first_lock["spec_sha256"], second_lock["spec_sha256"])
            self.assertEqual(first_lock["cases_sha256"], second_lock["cases_sha256"])
            self.assertEqual(
                [definition.case_id for definition in first.load_definitions()],
                [definition.case_id for definition in second.load_definitions()],
            )
            training = first.load_definitions(split="train")
            alphas = [definition.alpha for definition in training]
            lower, upper = 0.015, 0.030
            strata = {
                int((alpha - lower) / (upper - lower) * len(training))
                for alpha in alphas
            }
            self.assertEqual(strata, set(range(len(training))))

            spec = json.loads(first.spec_path.read_text(encoding="utf-8"))
            spec["seed"] += 1
            first.spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
            with self.assertRaises(CampaignLockError):
                first.verify_lock()

    def test_manual_batches_resume_train_and_evaluate_from_locked_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ReproducibleCampaign.initialize(
                Path(directory) / "campaign",
                name="manual-heat",
                seed=42,
            )
            configure_small_internal_campaign(campaign)
            campaign.lock()

            dry_run = campaign.run(split="train", limit=2, dry_run=True)
            self.assertEqual(dry_run["pending"], 2)
            first_batch = campaign.run(split="train", limit=2)
            second_batch = campaign.run(split="train", limit=2)
            third_batch = campaign.run(split="train", limit=2)
            self.assertEqual(len(first_batch["completed_case_ids"]), 2)
            self.assertEqual(len(second_batch["completed_case_ids"]), 1)
            self.assertEqual(len(third_batch["completed_case_ids"]), 0)

            status = campaign.status()
            self.assertEqual(status.split_counts["train"]["completed"], 3)
            self.assertEqual(status.split_counts["validation"]["completed"], 0)

            first_model = campaign.train()
            second_model = campaign.train()
            self.assertEqual(
                first_model["reproducibility_key"],
                second_model["reproducibility_key"],
            )
            self.assertEqual(
                first_model["artifact_content_sha256"],
                second_model["artifact_content_sha256"],
            )
            self.assertEqual(first_model["algorithm"], "heat-local-residual")
            self.assertEqual(first_model["training_mode"], "cumulative-refit")
            self.assertEqual(first_model["checkpoint_index"], 1)
            self.assertIsNone(first_model["parent_reproducibility_key"])
            self.assertEqual(first_model["lineage_id"], second_model["lineage_id"])
            self.assertEqual(second_model["checkpoint_index"], 1)

            campaign.run(split="validation")
            evaluation = campaign.evaluate(split="validation")
            self.assertEqual(evaluation["split"], "validation")
            self.assertEqual(len(evaluation["cases"]), 2)
            self.assertTrue(
                set(first_model["training_case_ids"]).isdisjoint(
                    row["case_id"] for row in evaluation["cases"]
                )
            )
            self.assertIn("mean_gradient_relative_l2", evaluation["metrics"])
            self.assertIn("median_preview_speedup", evaluation["metrics"])
            self.assertIn("worst_boundary_residual", evaluation["metrics"])
            self.assertIn("validation-centered", evaluation["metrics_by_family"])
            self.assertIsNone(evaluation["checkpoint_comparison"])
            final_status = campaign.status()
            self.assertTrue(final_status.model_exists)
            self.assertEqual(final_status.model_count, 1)
            self.assertEqual(
                final_status.active_model_key,
                first_model["reproducibility_key"],
            )
            self.assertIn("validation", final_status.evaluations)

            export = campaign.export_results()
            csv_path = campaign.root / export["csv_path"]
            npz_path = campaign.root / export["npz_path"]
            self.assertTrue(csv_path.is_file())
            self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 6)
            with np.load(npz_path, allow_pickle=False) as archive:
                self.assertEqual(archive["nominal_fields"].shape, (5, 17, 17))
                self.assertEqual(archive["coarse_fields"].shape, (5, 9, 9))
                self.assertEqual(archive["case_ids"].shape, (5,))
                self.assertEqual(archive["alphas"].shape, (5,))
                self.assertEqual(archive["centers"].shape, (5, 2))
                self.assertEqual(archive["boundaries"].shape, (5, 4))
            dashboard_path = campaign.dashboard()
            dashboard = dashboard_path.read_text(encoding="utf-8")
            self.assertIn("Résultats concaténés", dashboard)
            self.assertIn("Courbe d’apprentissage cumulative", dashboard)
            self.assertIn(first_model["reproducibility_key"], dashboard)
            self.assertIn('"checkpoint_index":1', dashboard)
            self.assertIn("manual-heat", dashboard)

    def test_training_requires_complete_split_unless_explicitly_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ReproducibleCampaign.initialize(
                Path(directory) / "campaign",
                name="partial-heat",
                seed=7,
            )
            configure_small_internal_campaign(campaign)
            campaign.lock()
            campaign.run(split="train", limit=2)
            with self.assertRaises(RuntimeError):
                campaign.train()
            partial_manifest = campaign.train(allow_partial=True)
            self.assertEqual(len(partial_manifest["training_case_ids"]), 2)
            campaign.run(split="validation")
            partial_evaluation = campaign.evaluate(split="validation")
            self.assertIsNone(partial_evaluation["checkpoint_comparison"])
            campaign.run(split="train")
            complete_manifest = campaign.train()
            self.assertNotEqual(
                partial_manifest["reproducibility_key"],
                complete_manifest["reproducibility_key"],
            )
            self.assertEqual(complete_manifest["checkpoint_index"], 2)
            self.assertEqual(
                complete_manifest["parent_reproducibility_key"],
                partial_manifest["reproducibility_key"],
            )
            self.assertEqual(len(complete_manifest["new_training_case_ids"]), 1)
            complete_evaluation = campaign.evaluate(split="validation")
            comparison = complete_evaluation["checkpoint_comparison"]
            self.assertIsNotNone(comparison)
            self.assertEqual(comparison["paired_case_count"], 2)
            models = campaign.list_models()
            self.assertEqual(len(models), 2)
            self.assertEqual(sum(model["active"] for model in models), 1)
            restored = campaign.activate_model(
                partial_manifest["reproducibility_key"][:12]
            )
            self.assertEqual(
                restored["reproducibility_key"],
                partial_manifest["reproducibility_key"],
            )
            self.assertEqual(
                campaign.status().active_model_key,
                partial_manifest["reproducibility_key"],
            )

    def test_full_campaign_batches_checkpoint_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = ReproducibleCampaign.initialize(
                Path(directory) / "campaign",
                name="full-heat",
                seed=91,
            )
            configure_small_internal_campaign(campaign)
            campaign.lock()

            first = campaign.run_full_training_campaign(
                cases_per_batch=2,
                algorithm="heat-local-residual",
            )

            self.assertEqual(first["status"], "completed")
            self.assertEqual(len(first["batches"]), 2)
            self.assertEqual(
                [batch["training_case_count"] for batch in first["batches"]],
                [2, 3],
            )
            status = campaign.status()
            self.assertEqual(status.split_counts["train"]["pending"], 0)
            self.assertEqual(status.split_counts["validation"]["pending"], 0)
            self.assertEqual(status.split_counts["test"]["completed"], 0)
            self.assertEqual(status.model_count, 2)

            resumed = campaign.run_full_training_campaign(
                cases_per_batch=2,
                algorithm="heat-local-residual",
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(len(resumed["batches"]), 2)
            self.assertEqual(campaign.status().model_count, 2)


if __name__ == "__main__":
    unittest.main()
