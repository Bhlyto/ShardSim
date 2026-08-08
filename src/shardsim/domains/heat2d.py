from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from shardsim.contracts import BoundaryConditions, ProblemSpec, SimulationCase


def make_heat_problem(
    alpha: float,
    t_end: float,
    extent: tuple[float, float] = (1.0, 1.0),
) -> ProblemSpec:
    if alpha <= 0:
        raise ValueError("Thermal diffusivity alpha must be positive.")
    return ProblemSpec(
        domain="heat-2d",
        equation="du/dt=alpha*laplacian(u)",
        parameters={"alpha": float(alpha)},
        t_end=float(t_end),
        extent=extent,
        input_units={"x": "m", "y": "m", "time": "s", "alpha": "m^2/s"},
        output_units={"temperature": "K"},
    )


def gaussian_initial_field(
    shape: tuple[int, int],
    center: tuple[float, float] = (0.5, 0.5),
    sigma: tuple[float, float] = (0.1, 0.1),
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    if len(shape) != 2 or min(shape) < 3:
        raise ValueError("shape must describe a grid of at least 3x3.")
    if len(center) != 2 or any(not 0 <= coordinate <= 1 for coordinate in center):
        raise ValueError("center coordinates must be in [0, 1].")
    if len(sigma) != 2 or any(width <= 0 for width in sigma):
        raise ValueError("sigma values must be positive.")

    y = np.linspace(0.0, 1.0, shape[0])
    x = np.linspace(0.0, 1.0, shape[1])
    x_grid, y_grid = np.meshgrid(x, y)
    exponent = -0.5 * (
        np.square((x_grid - center[0]) / sigma[0])
        + np.square((y_grid - center[1]) / sigma[1])
    )
    return baseline + amplitude * np.exp(exponent)


def make_heat_case(
    case_id: str,
    alpha: float,
    t_end: float,
    initial_field: np.ndarray,
    boundaries: BoundaryConditions | None = None,
    extent: tuple[float, float] = (1.0, 1.0),
    metadata: Mapping[str, Any] | None = None,
) -> SimulationCase:
    return SimulationCase(
        case_id=case_id,
        problem=make_heat_problem(alpha=alpha, t_end=t_end, extent=extent),
        initial_field=initial_field,
        boundaries=boundaries or BoundaryConditions(),
        metadata=metadata or {},
    )
