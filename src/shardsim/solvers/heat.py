from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter

import numpy as np

from shardsim.canonical import FieldLocation
from shardsim.contracts import BoundaryConditions, Fidelity, SimulationCase, SimulationResult
from shardsim.interpolation import bilinear_resample


def apply_dirichlet(field: np.ndarray, boundaries: BoundaryConditions) -> None:
    field[0, 1:-1] = boundaries.top
    field[-1, 1:-1] = boundaries.bottom
    field[1:-1, 0] = boundaries.left
    field[1:-1, -1] = boundaries.right
    field[0, 0] = 0.5 * (boundaries.top + boundaries.left)
    field[0, -1] = 0.5 * (boundaries.top + boundaries.right)
    field[-1, 0] = 0.5 * (boundaries.bottom + boundaries.left)
    field[-1, -1] = 0.5 * (boundaries.bottom + boundaries.right)


@dataclass(frozen=True, slots=True)
class HeatDiscretization:
    alpha: float
    dx: float
    dy: float
    dt: float
    n_steps: int
    stability_number: float


@dataclass(frozen=True, slots=True)
class HeatSimulationTrace:
    case_id: str
    fidelity: Fidelity
    fields: np.ndarray
    t_end: float
    dt: float
    runtime_seconds: float
    metadata: dict[str, float | str]

    def __post_init__(self) -> None:
        fields = np.asarray(self.fields, dtype=np.float64)
        if fields.ndim != 3 or fields.shape[0] < 2 or min(fields.shape[1:]) < 3:
            raise ValueError("A heat trace must contain at least two two-dimensional states.")
        if not np.isfinite(fields).all():
            raise ValueError("A heat trace must contain only finite values.")
        if not np.isclose(self.dt * (fields.shape[0] - 1), self.t_end):
            raise ValueError("Trace steps do not reach the requested physical horizon.")
        fields = fields.copy()
        fields.setflags(write=False)
        object.__setattr__(self, "fields", fields)

    @property
    def n_steps(self) -> int:
        return self.fields.shape[0] - 1

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.fields.shape[1:]

    def field_at(
        self,
        time: float,
        target_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if time < 0 or time > self.t_end:
            raise ValueError("Requested trace time lies outside the simulation horizon.")
        position = min(time / self.dt, float(self.n_steps))
        lower_index = int(np.floor(position))
        upper_index = min(lower_index + 1, self.n_steps)
        weight = position - lower_index
        field = (1.0 - weight) * self.fields[lower_index] + weight * self.fields[upper_index]
        if target_shape is not None:
            return bilinear_resample(field, target_shape)
        return field.copy()

    def final_result(self) -> SimulationResult:
        return SimulationResult(
            case_id=self.case_id,
            fidelity=self.fidelity,
            field=self.fields[-1],
            t_end=self.t_end,
            dt=self.dt,
            n_steps=self.n_steps,
            runtime_seconds=self.runtime_seconds,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class HeatEquationSolver:
    safety_factor: float = 0.9

    @property
    def output_location(self) -> FieldLocation:
        return FieldLocation.POINT

    def __post_init__(self) -> None:
        if not 0 < self.safety_factor <= 1:
            raise ValueError("safety_factor must be in (0, 1].")

    def solve(
        self,
        case: SimulationCase,
        fidelity: Fidelity,
        grid_shape: tuple[int, int],
    ) -> SimulationResult:
        result, _ = self._simulate(case, fidelity, grid_shape, capture_trace=False)
        return result

    def solve_trace(
        self,
        case: SimulationCase,
        fidelity: Fidelity,
        grid_shape: tuple[int, int],
    ) -> HeatSimulationTrace:
        result, frames = self._simulate(case, fidelity, grid_shape, capture_trace=True)
        if frames is None:
            raise RuntimeError("Trace capture unexpectedly produced no frames.")
        return HeatSimulationTrace(
            case_id=result.case_id,
            fidelity=result.fidelity,
            fields=frames,
            t_end=result.t_end,
            dt=result.dt,
            runtime_seconds=result.runtime_seconds,
            metadata=dict(result.metadata),
        )

    def discretization(
        self,
        case: SimulationCase,
        grid_shape: tuple[int, int],
    ) -> HeatDiscretization:
        if case.problem.domain != "heat-2d":
            raise ValueError(f"HeatEquationSolver cannot solve domain {case.problem.domain!r}.")
        if len(grid_shape) != 2 or min(grid_shape) < 3:
            raise ValueError("grid_shape must describe a grid of at least 3x3.")

        alpha = case.problem.parameter("alpha")
        if alpha <= 0:
            raise ValueError("Thermal diffusivity alpha must be positive.")

        length_x, length_y = case.problem.extent
        rows, columns = grid_shape
        dx = length_x / (columns - 1)
        dy = length_y / (rows - 1)
        inverse_spacing_sum = (1.0 / (dx * dx)) + (1.0 / (dy * dy))
        stable_dt = self.safety_factor / (2.0 * alpha * inverse_spacing_sum)
        n_steps = max(1, ceil(case.problem.t_end / stable_dt))
        dt = case.problem.t_end / n_steps
        return HeatDiscretization(
            alpha=alpha,
            dx=dx,
            dy=dy,
            dt=dt,
            n_steps=n_steps,
            stability_number=alpha * dt * inverse_spacing_sum,
        )

    def _simulate(
        self,
        case: SimulationCase,
        fidelity: Fidelity,
        grid_shape: tuple[int, int],
        capture_trace: bool,
    ) -> tuple[SimulationResult, np.ndarray | None]:
        discretization = self.discretization(case, grid_shape)

        temperature = bilinear_resample(case.initial_field, grid_shape)
        apply_dirichlet(temperature, case.boundaries)
        frames = [temperature.copy()] if capture_trace else None

        started_at = perf_counter()
        for _ in range(discretization.n_steps):
            previous = temperature.copy()
            center = previous[1:-1, 1:-1]
            laplacian_x = (
                previous[1:-1, 2:] - 2.0 * center + previous[1:-1, :-2]
            ) / (discretization.dx * discretization.dx)
            laplacian_y = (
                previous[2:, 1:-1] - 2.0 * center + previous[:-2, 1:-1]
            ) / (discretization.dy * discretization.dy)
            temperature[1:-1, 1:-1] = center + discretization.alpha * discretization.dt * (
                laplacian_x + laplacian_y
            )
            apply_dirichlet(temperature, case.boundaries)
            if frames is not None:
                frames.append(temperature.copy())
        runtime_seconds = perf_counter() - started_at

        result = SimulationResult(
            case_id=case.case_id,
            fidelity=fidelity,
            field=temperature,
            t_end=case.problem.t_end,
            dt=discretization.dt,
            n_steps=discretization.n_steps,
            runtime_seconds=runtime_seconds,
            metadata={
                "alpha": discretization.alpha,
                "dx": discretization.dx,
                "dy": discretization.dy,
                "stability_number": discretization.stability_number,
                "solver": "explicit-euler-5-point",
            },
        )
        stacked_frames = np.stack(frames) if frames is not None else None
        return result, stacked_frames
