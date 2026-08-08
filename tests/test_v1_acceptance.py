from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.cli import main
from shardsim.execution import inspect_result, run_scenario
from shardsim.scenario import ScenarioValidationError, parse_scenario


def valid_payload(scenario_id: str = "acceptance-001") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "scenario_id": scenario_id,
        "model": "heat-2d",
        "parameters": {"alpha": 0.02, "t_end": 0.002, "extent": [1.0, 1.0]},
        "initial_conditions": {
            "type": "gaussian",
            "shape": [9, 9],
            "center": [0.5, 0.5],
            "sigma": [0.1, 0.1],
            "amplitude": 1.0,
            "baseline": 0.0,
        },
        "boundary_conditions": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
        "solver": {"backend": "internal", "grid_shape": [9, 9], "safety_factor": 0.9},
        "seed": None,
        "metadata": {"suite": "v1-acceptance"},
    }


class V1AcceptanceTests(unittest.TestCase):
    def test_valid_scenario_loads_and_invalid_scenario_is_explicit(self) -> None:
        scenario = parse_scenario(valid_payload())
        self.assertEqual(scenario.scenario_id, "acceptance-001")
        invalid = valid_payload()
        invalid["parameters"] = {"alpha": -1.0, "t_end": 0.01}
        with self.assertRaisesRegex(ScenarioValidationError, r"\$\.parameters\.alpha"):
            parse_scenario(invalid)

    def test_run_saves_structured_reproducible_result_and_logs(self) -> None:
        scenario = parse_scenario(valid_payload())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_scenario(scenario, root / "first")
            second = run_scenario(scenario, root / "second")

            self.assertEqual(first["status"], "success")
            self.assertEqual(first["exit_code"], 0)
            self.assertEqual(first["reproducibility_key"], second["reproducibility_key"])
            self.assertEqual(first["scenario_sha256"], second["scenario_sha256"])
            np.testing.assert_array_equal(
                np.load(root / "first" / "field.npy", allow_pickle=False),
                np.load(root / "second" / "field.npy", allow_pickle=False),
            )
            self.assertIn("completed scenario=acceptance-001", (root / "first" / "run.log").read_text())
            self.assertTrue((root / "first" / "result.json").is_file())
            self.assertTrue(inspect_result(root / "first" / "result.json")["valid"])

            with self.assertRaisesRegex(FileExistsError, "already contains"):
                run_scenario(scenario, root / "first")

            field_path = root / "first" / "field.npy"
            with field_path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                inspect_result(root / "first" / "result.json")

    def test_cli_is_usable_as_a_plain_external_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario_path = root / "scenario.json"
            scenario_path.write_text(json.dumps(valid_payload()), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "shardsim", "run", str(scenario_path), "--output", str(root / "result")],
                cwd=Path(__file__).resolve().parents[1],
                env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "success")

            inspected = subprocess.run(
                [sys.executable, "-m", "shardsim", "inspect", str(root / "result" / "result.json")],
                cwd=Path(__file__).resolve().parents[1],
                env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            self.assertTrue(json.loads(inspected.stdout)["valid"])

    def test_invalid_cli_input_returns_nonzero_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"schema_version":"9"}', encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(["validate", str(path)])
            self.assertEqual(exit_code, 2)
            self.assertIn("missing required field", stderr.getvalue())

    def test_one_hundred_scenarios_run_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys: list[str] = []
            for index in range(100):
                scenario = parse_scenario(valid_payload(f"batch-{index:04d}"))
                record = run_scenario(scenario, root / scenario.scenario_id)
                self.assertEqual(record["status"], "success")
                self.assertTrue((root / scenario.scenario_id / "result.json").is_file())
                keys.append(record["reproducibility_key"])
            self.assertEqual(len(keys), 100)
            self.assertEqual(len(set(keys)), 100)
            self.assertFalse(list(root.rglob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
