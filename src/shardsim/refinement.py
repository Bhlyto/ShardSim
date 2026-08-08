from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shardsim.contracts import BoundaryConditions, Fidelity, SimulationCase
from shardsim.interpolation import bilinear_resample
from shardsim.solvers.heat import HeatEquationSolver, HeatSimulationTrace


@dataclass(frozen=True, slots=True)
class RefinementRegion:
    row_start: int
    row_end: int
    column_start: int
    column_end: int
    halo: int
    score: float
    domain_shape: tuple[int, int]

    def __post_init__(self) -> None:
        rows, columns = self.domain_shape
        if not 0 <= self.row_start < self.row_end <= rows:
            raise ValueError("Invalid refinement row bounds.")
        if not 0 <= self.column_start < self.column_end <= columns:
            raise ValueError("Invalid refinement column bounds.")
        if self.halo < 1:
            raise ValueError("A refinement region requires a halo of at least one cell.")
        if not np.isfinite(self.score) or self.score < 0:
            raise ValueError("A refinement score must be finite and non-negative.")

    @property
    def core_slice(self) -> tuple[slice, slice]:
        return (
            slice(self.row_start, self.row_end),
            slice(self.column_start, self.column_end),
        )

    @property
    def patch_bounds(self) -> tuple[int, int, int, int]:
        rows, columns = self.domain_shape
        return (
            max(0, self.row_start - self.halo),
            min(rows, self.row_end + self.halo),
            max(0, self.column_start - self.halo),
            min(columns, self.column_end + self.halo),
        )

    @property
    def patch_slice(self) -> tuple[slice, slice]:
        row_start, row_end, column_start, column_end = self.patch_bounds
        return slice(row_start, row_end), slice(column_start, column_end)

    @property
    def core_in_patch_slice(self) -> tuple[slice, slice]:
        patch_row_start, _, patch_column_start, _ = self.patch_bounds
        return (
            slice(self.row_start - patch_row_start, self.row_end - patch_row_start),
            slice(self.column_start - patch_column_start, self.column_end - patch_column_start),
        )

    @property
    def core_cell_count(self) -> int:
        return (self.row_end - self.row_start) * (self.column_end - self.column_start)

    @property
    def patch_shape(self) -> tuple[int, int]:
        row_start, row_end, column_start, column_end = self.patch_bounds
        return row_end - row_start, column_end - column_start


def _partition_bounds(size: int, target_tile_size: int) -> tuple[int, ...]:
    tile_count = max(1, round(size / target_tile_size))
    return tuple(int(value) for value in np.linspace(0, size, tile_count + 1))


def select_refinement_regions(
    score_map: np.ndarray,
    tile_shape: tuple[int, int] = (16, 16),
    max_regions: int = 4,
    halo: int = 4,
    min_score: float = 0.0,
) -> tuple[RefinementRegion, ...]:
    scores = np.asarray(score_map, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 3:
        raise ValueError("score_map must be a two-dimensional grid of at least 3x3.")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("score_map must contain finite non-negative values.")
    if len(tile_shape) != 2 or min(tile_shape) < 1:
        raise ValueError("tile_shape must contain two positive sizes.")
    if max_regions < 0:
        raise ValueError("max_regions cannot be negative.")
    if halo < 1:
        raise ValueError("halo must be at least one cell.")

    row_bounds = _partition_bounds(scores.shape[0], tile_shape[0])
    column_bounds = _partition_bounds(scores.shape[1], tile_shape[1])
    candidates: list[RefinementRegion] = []
    for row_start, row_end in zip(row_bounds[:-1], row_bounds[1:]):
        for column_start, column_end in zip(column_bounds[:-1], column_bounds[1:]):
            tile = scores[row_start:row_end, column_start:column_end]
            score = float(np.sqrt(np.mean(np.square(tile))))
            if score >= min_score:
                candidates.append(
                    RefinementRegion(
                        row_start=row_start,
                        row_end=row_end,
                        column_start=column_start,
                        column_end=column_end,
                        halo=halo,
                        score=score,
                        domain_shape=scores.shape,
                    )
                )
    candidates.sort(key=lambda region: (-region.score, region.row_start, region.column_start))
    return tuple(candidates[:max_regions])


@dataclass(frozen=True, slots=True)
class LocalRefinementResult:
    case_id: str
    base_field: np.ndarray
    merged_field: np.ndarray
    regions: tuple[RefinementRegion, ...]
    coarse_cell_steps: int
    local_cell_steps: int
    nominal_cell_steps: int

    def __post_init__(self) -> None:
        base_field = np.asarray(self.base_field, dtype=np.float64)
        merged_field = np.asarray(self.merged_field, dtype=np.float64)
        if base_field.shape != merged_field.shape or base_field.ndim != 2:
            raise ValueError("Base and merged refinement fields must share a two-dimensional shape.")
        if min(self.coarse_cell_steps, self.local_cell_steps, self.nominal_cell_steps) < 0:
            raise ValueError("Cell-step costs cannot be negative.")
        if self.nominal_cell_steps == 0:
            raise ValueError("Nominal cell-step cost must be positive.")
        base_field = base_field.copy()
        merged_field = merged_field.copy()
        base_field.setflags(write=False)
        merged_field.setflags(write=False)
        object.__setattr__(self, "base_field", base_field)
        object.__setattr__(self, "merged_field", merged_field)

    @property
    def refined_cell_count(self) -> int:
        return sum(region.core_cell_count for region in self.regions)

    @property
    def refined_domain_fraction(self) -> float:
        return self.refined_cell_count / self.merged_field.size

    @property
    def local_compute_fraction(self) -> float:
        return self.local_cell_steps / self.nominal_cell_steps

    @property
    def estimated_total_compute_fraction(self) -> float:
        return (self.coarse_cell_steps + self.local_cell_steps) / self.nominal_cell_steps


def _apply_patch_boundaries(
    patch: np.ndarray,
    guide: np.ndarray,
    region: RefinementRegion,
    boundaries: BoundaryConditions,
) -> None:
    row_start, row_end, column_start, column_end = region.patch_bounds
    domain_rows, domain_columns = region.domain_shape
    patch[0, :] = guide[row_start, column_start:column_end]
    patch[-1, :] = guide[row_end - 1, column_start:column_end]
    patch[:, 0] = guide[row_start:row_end, column_start]
    patch[:, -1] = guide[row_start:row_end, column_end - 1]

    if row_start == 0:
        patch[0, :] = boundaries.top
    if row_end == domain_rows:
        patch[-1, :] = boundaries.bottom
    if column_start == 0:
        patch[:, 0] = boundaries.left
    if column_end == domain_columns:
        patch[:, -1] = boundaries.right
    if row_start == 0 and column_start == 0:
        patch[0, 0] = 0.5 * (boundaries.top + boundaries.left)
    if row_start == 0 and column_end == domain_columns:
        patch[0, -1] = 0.5 * (boundaries.top + boundaries.right)
    if row_end == domain_rows and column_start == 0:
        patch[-1, 0] = 0.5 * (boundaries.bottom + boundaries.left)
    if row_end == domain_rows and column_end == domain_columns:
        patch[-1, -1] = 0.5 * (boundaries.bottom + boundaries.right)


@dataclass(frozen=True, slots=True)
class HeatLocalRefiner:
    solver: HeatEquationSolver

    def refine(
        self,
        case: SimulationCase,
        base_field: np.ndarray,
        score_map: np.ndarray,
        coarse_trace: HeatSimulationTrace,
        nominal_shape: tuple[int, int],
        tile_shape: tuple[int, int] = (16, 16),
        max_regions: int = 4,
        halo: int = 4,
        min_score: float = 0.0,
    ) -> LocalRefinementResult:
        base = np.asarray(base_field, dtype=np.float64)
        scores = np.asarray(score_map, dtype=np.float64)
        if base.shape != nominal_shape or scores.shape != nominal_shape:
            raise ValueError("Base field and score map must use the nominal grid.")
        if coarse_trace.case_id != case.case_id or coarse_trace.fidelity is not Fidelity.COARSE:
            raise ValueError("A matching coarse trace is required for local refinement.")
        if not np.isclose(coarse_trace.t_end, case.problem.t_end):
            raise ValueError("Coarse trace and case use different physical horizons.")

        regions = select_refinement_regions(
            scores,
            tile_shape=tile_shape,
            max_regions=max_regions,
            halo=halo,
            min_score=min_score,
        )
        discretization = self.solver.discretization(case, nominal_shape)
        nominal_interior_cells = max(1, (nominal_shape[0] - 2) * (nominal_shape[1] - 2))
        nominal_cell_steps = nominal_interior_cells * discretization.n_steps
        coarse_interior_cells = max(
            1,
            (coarse_trace.grid_shape[0] - 2) * (coarse_trace.grid_shape[1] - 2),
        )
        coarse_cell_steps = coarse_interior_cells * coarse_trace.n_steps

        if not regions:
            return LocalRefinementResult(
                case_id=case.case_id,
                base_field=base,
                merged_field=base,
                regions=(),
                coarse_cell_steps=coarse_cell_steps,
                local_cell_steps=0,
                nominal_cell_steps=nominal_cell_steps,
            )

        nominal_initial = bilinear_resample(case.initial_field, nominal_shape)
        patches = [nominal_initial[region.patch_slice].copy() for region in regions]
        guide = coarse_trace.field_at(0.0, nominal_shape)
        for patch, region in zip(patches, regions):
            _apply_patch_boundaries(patch, guide, region, case.boundaries)

        for step in range(1, discretization.n_steps + 1):
            guide = coarse_trace.field_at(step * discretization.dt, nominal_shape)
            for patch, region in zip(patches, regions):
                previous = patch.copy()
                center = previous[1:-1, 1:-1]
                laplacian_x = (
                    previous[1:-1, 2:] - 2.0 * center + previous[1:-1, :-2]
                ) / (discretization.dx * discretization.dx)
                laplacian_y = (
                    previous[2:, 1:-1] - 2.0 * center + previous[:-2, 1:-1]
                ) / (discretization.dy * discretization.dy)
                patch[1:-1, 1:-1] = center + discretization.alpha * discretization.dt * (
                    laplacian_x + laplacian_y
                )
                _apply_patch_boundaries(patch, guide, region, case.boundaries)

        merged = base.copy()
        for patch, region in zip(patches, regions):
            merged[region.core_slice] = patch[region.core_in_patch_slice]

        local_cell_steps = sum(
            max(1, (region.patch_shape[0] - 2) * (region.patch_shape[1] - 2))
            * discretization.n_steps
            for region in regions
        )
        return LocalRefinementResult(
            case_id=case.case_id,
            base_field=base,
            merged_field=merged,
            regions=regions,
            coarse_cell_steps=coarse_cell_steps,
            local_cell_steps=local_cell_steps,
            nominal_cell_steps=nominal_cell_steps,
        )
