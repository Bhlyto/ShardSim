from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Any

import numpy as np

from shardsim.contracts import Fidelity
from shardsim.domains.heat2d import make_heat_case
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.solvers.openfoam import OpenFOAMAdapter, sample_cell_centers


@dataclass(frozen=True, slots=True)
class SolverVerificationMetrics:
    relative_l2: float
    max_abs_error: float
    relative_energy_error: float
    runtime_seconds: float
    n_steps: int


@dataclass(frozen=True, slots=True)
class HeatGridVerification:
    cells_per_axis: int
    internal: SolverVerificationMetrics
    openfoam: SolverVerificationMetrics | None
    cross_solver_relative_l2: float | None
    internal_observed_order: float | None
    openfoam_observed_order: float | None


@dataclass(frozen=True, slots=True)
class HeatVerificationReport:
    alpha: float
    t_end: float
    extent: tuple[float, float]
    records: tuple[HeatGridVerification, ...]
    openfoam_adapter_id: str | None

    @property
    def finest(self) -> HeatGridVerification:
        return self.records[-1]

    def passes(
        self,
        relative_l2_tolerance: float = 0.02,
        minimum_observed_order: float = 1.5,
    ) -> bool:
        finest = self.finest
        if finest.internal.relative_l2 > relative_l2_tolerance:
            return False
        if finest.internal_observed_order is None or finest.internal_observed_order < minimum_observed_order:
            return False
        if finest.openfoam is None:
            return True
        return (
            finest.openfoam.relative_l2 <= relative_l2_tolerance
            and finest.openfoam_observed_order is not None
            and finest.openfoam_observed_order >= minimum_observed_order
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "t_end": self.t_end,
            "extent": list(self.extent),
            "openfoam_adapter_id": self.openfoam_adapter_id,
            "passed": self.passes(),
            "records": [
                {
                    "cells_per_axis": record.cells_per_axis,
                    "internal": _metrics_dict(record.internal),
                    "openfoam": _metrics_dict(record.openfoam) if record.openfoam else None,
                    "cross_solver_relative_l2": record.cross_solver_relative_l2,
                    "internal_observed_order": record.internal_observed_order,
                    "openfoam_observed_order": record.openfoam_observed_order,
                }
                for record in self.records
            ],
        }


def sine_mode_initial_field(
    shape: tuple[int, int],
    extent: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    if len(shape) != 2 or min(shape) < 3:
        raise ValueError("shape must describe at least a 3x3 nodal grid.")
    length_x, length_y = extent
    x = np.linspace(0.0, length_x, shape[1])
    y = np.linspace(length_y, 0.0, shape[0])
    x_grid, y_grid = np.meshgrid(x, y)
    return np.sin(pi * x_grid / length_x) * np.sin(pi * y_grid / length_y)


def exact_sine_mode_cells(
    shape: tuple[int, int],
    alpha: float,
    time: float,
    extent: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must contain positive cell counts.")
    if alpha <= 0 or time < 0:
        raise ValueError("alpha must be positive and time non-negative.")
    rows, columns = shape
    length_x, length_y = extent
    x = (np.arange(columns, dtype=np.float64) + 0.5) * length_x / columns
    y = length_y - (np.arange(rows, dtype=np.float64) + 0.5) * length_y / rows
    x_grid, y_grid = np.meshgrid(x, y)
    decay_rate = alpha * pi * pi * (1.0 / (length_x * length_x) + 1.0 / (length_y * length_y))
    return (
        np.sin(pi * x_grid / length_x)
        * np.sin(pi * y_grid / length_y)
        * np.exp(-decay_rate * time)
    )


def run_heat_verification(
    resolutions: tuple[int, ...] = (8, 16, 32),
    alpha: float = 0.02,
    t_end: float = 0.05,
    extent: tuple[float, float] = (1.0, 1.0),
    openfoam: OpenFOAMAdapter | None = None,
    internal_solver: HeatEquationSolver | None = None,
) -> HeatVerificationReport:
    ordered_resolutions = tuple(sorted(set(resolutions)))
    if len(ordered_resolutions) < 2 or min(ordered_resolutions) < 4:
        raise ValueError("Verification requires at least two resolutions of 4x4 cells or finer.")
    if alpha <= 0 or t_end <= 0:
        raise ValueError("alpha and t_end must be positive.")

    internal_solver = internal_solver or HeatEquationSolver()
    raw_records: list[
        tuple[int, SolverVerificationMetrics, SolverVerificationMetrics | None, float | None]
    ] = []
    for cells_per_axis in ordered_resolutions:
        cell_shape = (cells_per_axis, cells_per_axis)
        nodal_shape = (cells_per_axis + 1, cells_per_axis + 1)
        case = make_heat_case(
            case_id=f"heat-sine-{cells_per_axis}",
            alpha=alpha,
            t_end=t_end,
            initial_field=sine_mode_initial_field(nodal_shape, extent),
            extent=extent,
            metadata={"verification": "sine-mode"},
        )
        exact = exact_sine_mode_cells(cell_shape, alpha, t_end, extent)
        internal_result = internal_solver.solve(case, Fidelity.NOMINAL, nodal_shape)
        internal_cells = sample_cell_centers(internal_result.field, cell_shape)
        internal_metrics = _verification_metrics(
            internal_cells,
            exact,
            extent,
            internal_result.runtime_seconds,
            internal_result.n_steps,
        )

        openfoam_metrics = None
        cross_solver_relative_l2 = None
        if openfoam is not None:
            openfoam_result = openfoam.solve(case, Fidelity.NOMINAL, cell_shape)
            openfoam_metrics = _verification_metrics(
                openfoam_result.field,
                exact,
                extent,
                openfoam_result.runtime_seconds,
                openfoam_result.n_steps,
            )
            cross_solver_relative_l2 = _relative_l2(openfoam_result.field, internal_cells)
        raw_records.append(
            (cells_per_axis, internal_metrics, openfoam_metrics, cross_solver_relative_l2)
        )

    records: list[HeatGridVerification] = []
    for index, (resolution, internal, openfoam_metrics, cross_error) in enumerate(raw_records):
        internal_order = None
        openfoam_order = None
        if index > 0:
            previous_resolution, previous_internal, previous_openfoam, _ = raw_records[index - 1]
            ratio = resolution / previous_resolution
            internal_order = _observed_order(
                previous_internal.relative_l2,
                internal.relative_l2,
                ratio,
            )
            if openfoam_metrics is not None and previous_openfoam is not None:
                openfoam_order = _observed_order(
                    previous_openfoam.relative_l2,
                    openfoam_metrics.relative_l2,
                    ratio,
                )
        records.append(
            HeatGridVerification(
                cells_per_axis=resolution,
                internal=internal,
                openfoam=openfoam_metrics,
                cross_solver_relative_l2=cross_error,
                internal_observed_order=internal_order,
                openfoam_observed_order=openfoam_order,
            )
        )

    return HeatVerificationReport(
        alpha=alpha,
        t_end=t_end,
        extent=extent,
        records=tuple(records),
        openfoam_adapter_id=openfoam.adapter_id if openfoam else None,
    )


def _verification_metrics(
    field: np.ndarray,
    exact: np.ndarray,
    extent: tuple[float, float],
    runtime_seconds: float,
    n_steps: int,
) -> SolverVerificationMetrics:
    difference = np.asarray(field) - np.asarray(exact)
    exact_norm = max(float(np.linalg.norm(exact.ravel())), 1e-15)
    cell_area = extent[0] * extent[1] / exact.size
    exact_energy = float(np.sum(exact) * cell_area)
    numerical_energy = float(np.sum(field) * cell_area)
    return SolverVerificationMetrics(
        relative_l2=float(np.linalg.norm(difference.ravel()) / exact_norm),
        max_abs_error=float(np.max(np.abs(difference))),
        relative_energy_error=abs(numerical_energy - exact_energy) / max(abs(exact_energy), 1e-15),
        runtime_seconds=runtime_seconds,
        n_steps=n_steps,
    )


def _relative_l2(field: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(field) - np.asarray(reference)
    return float(
        np.linalg.norm(difference.ravel())
        / max(float(np.linalg.norm(np.asarray(reference).ravel())), 1e-15)
    )


def _observed_order(coarse_error: float, fine_error: float, refinement_ratio: float) -> float:
    if coarse_error <= 0 or fine_error <= 0 or refinement_ratio <= 1:
        return float("nan")
    return log(coarse_error / fine_error) / log(refinement_ratio)


def _metrics_dict(metrics: SolverVerificationMetrics) -> dict[str, float | int]:
    return {
        "relative_l2": metrics.relative_l2,
        "max_abs_error": metrics.max_abs_error,
        "relative_energy_error": metrics.relative_energy_error,
        "runtime_seconds": metrics.runtime_seconds,
        "n_steps": metrics.n_steps,
    }
