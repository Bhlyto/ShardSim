from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.contracts import BoundaryConditions, Fidelity
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case, make_heat_problem
from shardsim.interpolation import bilinear_resample
from shardsim.metrics import boundary_residual
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.solvers.heat import HeatEquationSolver


class HeatFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_field = gaussian_initial_field((65, 65), sigma=(0.09, 0.09))
        self.solver = HeatEquationSolver()

    def make_case(self, alpha: float = 0.02, t_end: float = 0.03):
        return make_heat_case(
            case_id=f"heat-alpha-{alpha}",
            alpha=alpha,
            t_end=t_end,
            initial_field=self.initial_field,
        )

    def test_alpha_changes_solution_at_fixed_physical_time(self) -> None:
        slow = self.solver.solve(self.make_case(alpha=0.01), Fidelity.NOMINAL, (65, 65))
        fast = self.solver.solve(self.make_case(alpha=0.05), Fidelity.NOMINAL, (65, 65))
        self.assertGreater(float(np.max(np.abs(slow.field - fast.field))), 1e-4)
        self.assertAlmostEqual(slow.t_end, fast.t_end)

    def test_solver_reaches_exact_requested_horizon_stably(self) -> None:
        result = self.solver.solve(self.make_case(), Fidelity.NOMINAL, (65, 65))
        self.assertAlmostEqual(result.dt * result.n_steps, result.t_end, places=14)
        self.assertLessEqual(result.metadata["stability_number"], 0.5)

    def test_constant_state_is_stationary(self) -> None:
        initial_field = np.full((33, 33), 2.5)
        boundaries = BoundaryConditions(top=2.5, bottom=2.5, left=2.5, right=2.5)
        case = make_heat_case(
            case_id="constant-state",
            alpha=0.03,
            t_end=0.1,
            initial_field=initial_field,
            boundaries=boundaries,
        )
        result = self.solver.solve(case, Fidelity.NOMINAL, (33, 33))
        np.testing.assert_allclose(result.field, 2.5, atol=1e-12)
        self.assertEqual(boundary_residual(result.field, boundaries), 0.0)

    def test_reference_sample_preserves_delta_identity_and_time(self) -> None:
        pipeline = BootstrapPipeline(
            self.solver,
            FidelityPlan(coarse_shape=(17, 17), nominal_shape=(65, 65)),
        )
        sample = pipeline.run_case(self.make_case())
        np.testing.assert_allclose(
            sample.coarse_on_nominal + sample.delta,
            sample.nominal.field,
            atol=1e-12,
        )
        self.assertAlmostEqual(sample.coarse.t_end, sample.nominal.t_end)
        self.assertEqual(sample.error_map.shape, sample.nominal.grid_shape)

    def test_refinement_reduces_discretization_error(self) -> None:
        case = self.make_case(t_end=0.02)
        nominal = self.solver.solve(case, Fidelity.NOMINAL, (65, 65))
        coarse = self.solver.solve(case, Fidelity.COARSE, (17, 17))
        medium = self.solver.solve(case, Fidelity.COARSE, (33, 33))
        coarse_error = np.mean(
            np.abs(bilinear_resample(coarse.field, nominal.grid_shape) - nominal.field)
        )
        medium_error = np.mean(
            np.abs(bilinear_resample(medium.field, nominal.grid_shape) - nominal.field)
        )
        self.assertLess(medium_error, coarse_error)

    def test_invalid_diffusivity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_heat_problem(alpha=0.0, t_end=0.1)


if __name__ == "__main__":
    unittest.main()
