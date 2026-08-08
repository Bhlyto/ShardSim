from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from shardsim.contracts import BoundaryConditions, SimulationCase
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case


@dataclass(frozen=True, slots=True)
class HeatDesignSpace:
    alpha: tuple[float, float] = (0.01, 0.05)
    t_end: tuple[float, float] = (0.01, 0.08)
    center_x: tuple[float, float] = (0.2, 0.8)
    center_y: tuple[float, float] = (0.2, 0.8)
    sigma_x: tuple[float, float] = (0.05, 0.15)
    sigma_y: tuple[float, float] = (0.05, 0.15)
    amplitude: tuple[float, float] = (0.5, 1.5)
    baseline: tuple[float, float] = (0.0, 0.0)
    extent: tuple[float, float] = (1.0, 1.0)
    initial_shape: tuple[int, int] = (65, 65)
    boundaries: BoundaryConditions = BoundaryConditions()

    def __post_init__(self) -> None:
        ranges = (
            self.alpha,
            self.t_end,
            self.center_x,
            self.center_y,
            self.sigma_x,
            self.sigma_y,
            self.amplitude,
            self.baseline,
        )
        if any(
            len(value_range) != 2
            or not all(isfinite(float(value)) for value in value_range)
            or value_range[0] > value_range[1]
            for value_range in ranges
        ):
            raise ValueError("Each DOE range requires finite ordered bounds.")
        if self.alpha[0] <= 0 or self.t_end[0] <= 0:
            raise ValueError("Diffusivity and final time ranges must be positive.")
        if self.sigma_x[0] <= 0 or self.sigma_y[0] <= 0:
            raise ValueError("Gaussian widths must be positive.")
        if any(value < 0 or value > 1 for value in (*self.center_x, *self.center_y)):
            raise ValueError("Normalized Gaussian centers must stay in [0, 1].")
        if len(self.initial_shape) != 2 or min(self.initial_shape) < 3:
            raise ValueError("initial_shape must describe at least a 3x3 grid.")

    def sample(
        self,
        count: int,
        seed: int,
        prefix: str = "heat-doe",
    ) -> tuple[SimulationCase, ...]:
        if count < 1:
            raise ValueError("DOE sample count must be positive.")
        if not prefix.strip():
            raise ValueError("DOE case prefix cannot be empty.")
        ranges = np.asarray(
            [
                self.alpha,
                self.t_end,
                self.center_x,
                self.center_y,
                self.sigma_x,
                self.sigma_y,
                self.amplitude,
                self.baseline,
            ],
            dtype=np.float64,
        )
        unit_samples = _latin_hypercube(count, ranges.shape[0], seed)
        samples = ranges[:, 0] + unit_samples * (ranges[:, 1] - ranges[:, 0])
        cases = []
        for index, values in enumerate(samples):
            alpha, t_end, center_x, center_y, sigma_x, sigma_y, amplitude, baseline = values
            cases.append(
                make_heat_case(
                    case_id=f"{prefix}-{index:04d}",
                    alpha=float(alpha),
                    t_end=float(t_end),
                    extent=self.extent,
                    boundaries=self.boundaries,
                    initial_field=gaussian_initial_field(
                        self.initial_shape,
                        center=(float(center_x), float(center_y)),
                        sigma=(float(sigma_x), float(sigma_y)),
                        amplitude=float(amplitude),
                        baseline=float(baseline),
                    ),
                    metadata={
                        "design": "latin-hypercube",
                        "design_seed": seed,
                        "design_index": index,
                        "center": (float(center_x), float(center_y)),
                        "sigma": (float(sigma_x), float(sigma_y)),
                        "amplitude": float(amplitude),
                        "baseline": float(baseline),
                    },
                )
            )
        return tuple(cases)


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    samples = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        strata = (np.arange(count, dtype=np.float64) + generator.random(count)) / count
        samples[:, dimension] = strata[generator.permutation(count)]
    return samples
