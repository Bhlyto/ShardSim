from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.canonical import (
    DIFFUSIVITY,
    TEMPERATURE,
    CanonicalField,
    FieldLocation,
)
from shardsim.contracts import Fidelity
from shardsim.domains.heat2d import make_heat_case
from shardsim.solvers.openfoam import (
    OpenFOAMAdapter,
    OpenFOAMHeatCaseBuilder,
    parse_openfoam_scalar_field,
    sample_cell_centers,
)
from shardsim.verification.heat import (
    exact_sine_mode_cells,
    run_heat_verification,
    sine_mode_initial_field,
)


class CanonicalContractsTests(unittest.TestCase):
    def test_dimensions_match_openfoam_order(self) -> None:
        self.assertEqual(DIFFUSIVITY.to_openfoam(), "[0 2 -1 0 0 0 0]")
        self.assertEqual(TEMPERATURE.to_openfoam(), "[0 0 0 1 0 0 0]")

    def test_field_values_are_immutable(self) -> None:
        source = np.ones((3, 4))
        field = CanonicalField("T", source, TEMPERATURE, FieldLocation.CELL, unit="K")
        source[:] = 2.0
        np.testing.assert_allclose(field.values, 1.0)
        with self.assertRaises(ValueError):
            field.values[0, 0] = 3.0


class OpenFOAMCaseTests(unittest.TestCase):
    def make_case(self, size: int = 9):
        return make_heat_case(
            "openfoam-test",
            alpha=0.02,
            t_end=0.01,
            initial_field=sine_mode_initial_field((size, size)),
        )

    def test_cell_center_sampling_preserves_linear_fields(self) -> None:
        field = np.add.outer(np.arange(3.0), np.arange(4.0))
        sampled = sample_cell_centers(field, (2, 3))
        expected_rows = np.array([0.5, 1.5])[:, None]
        expected_columns = np.array([0.5, 1.5, 2.5])[None, :]
        np.testing.assert_allclose(sampled, expected_rows + expected_columns)

    def test_builder_writes_versioned_heat_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discretization = OpenFOAMHeatCaseBuilder().write(self.make_case(), (8, 8), root)
            self.assertTrue((root / "system" / "blockMeshDict").is_file())
            self.assertIn("laplacian(DT,T)", (root / "system" / "fvSchemes").read_text())
            self.assertIn("[0 2 -1 0 0 0 0]", (root / "constant" / "transportProperties").read_text())
            self.assertAlmostEqual(discretization.dt * discretization.n_steps, 0.01)

    def test_parser_restores_top_first_array_order(self) -> None:
        content = """internalField nonuniform List<scalar>
4
(
1
2
3
4
);
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "T"
            path.write_text(content)
            field = parse_openfoam_scalar_field(path, (2, 2))
        np.testing.assert_allclose(field, [[3, 4], [1, 2]])

    def test_analytical_solution_decays(self) -> None:
        initial = exact_sine_mode_cells((8, 8), alpha=0.02, time=0.0)
        later = exact_sine_mode_cells((8, 8), alpha=0.02, time=0.1)
        self.assertTrue(np.all(later < initial))

    def test_internal_verification_converges(self) -> None:
        report = run_heat_verification(resolutions=(8, 16, 32), t_end=0.02)
        errors = [record.internal.relative_l2 for record in report.records]
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertGreater(report.finest.internal_observed_order, 1.5)

    @unittest.skipUnless(
        os.environ.get("SHARDSIM_RUN_OPENFOAM_TESTS") == "1",
        "Set SHARDSIM_RUN_OPENFOAM_TESTS=1 to run Docker integration.",
    )
    def test_openfoam_matches_manufactured_solution(self) -> None:
        report = run_heat_verification(
            resolutions=(8, 16),
            t_end=0.02,
            openfoam=OpenFOAMAdapter(),
        )
        self.assertLess(report.finest.openfoam.relative_l2, 0.02)
        self.assertLess(report.finest.cross_solver_relative_l2, 0.02)


if __name__ == "__main__":
    unittest.main()
