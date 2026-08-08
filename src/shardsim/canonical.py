from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping

import numpy as np


class FieldRank(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"


class FieldLocation(str, Enum):
    CELL = "cell"
    FACE = "face"
    POINT = "point"


@dataclass(frozen=True, slots=True)
class PhysicalDimensions:
    """SI base-dimension exponents ordered as M, L, T, Theta, N, I, J."""

    exponents: tuple[float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.exponents) != 7 or not all(isfinite(float(value)) for value in self.exponents):
            raise ValueError("Physical dimensions require seven finite SI exponents.")

    def to_openfoam(self) -> str:
        values = " ".join(f"{float(value):g}" for value in self.exponents)
        return f"[{values}]"


DIMENSIONLESS = PhysicalDimensions((0, 0, 0, 0, 0, 0, 0))
LENGTH = PhysicalDimensions((0, 1, 0, 0, 0, 0, 0))
TIME = PhysicalDimensions((0, 0, 1, 0, 0, 0, 0))
TEMPERATURE = PhysicalDimensions((0, 0, 0, 1, 0, 0, 0))
DIFFUSIVITY = PhysicalDimensions((0, 2, -1, 0, 0, 0, 0))


@dataclass(frozen=True, slots=True)
class CanonicalField:
    name: str
    values: np.ndarray
    dimensions: PhysicalDimensions
    location: FieldLocation
    rank: FieldRank = FieldRank.SCALAR
    mesh_id: str | None = None
    time: float | None = None
    unit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A canonical field requires a name.")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim < 1 or not np.isfinite(values).all():
            raise ValueError("Canonical field values must be finite and non-empty.")
        if self.rank is FieldRank.VECTOR and (values.ndim < 2 or values.shape[-1] not in (2, 3)):
            raise ValueError("Vector fields require a final component axis of length 2 or 3.")
        if self.rank is FieldRank.TENSOR and (values.ndim < 2 or values.shape[-1] not in (4, 6, 9)):
            raise ValueError("Tensor fields require 4, 6, or 9 final-axis components.")
        if self.time is not None and (not isfinite(float(self.time)) or self.time < 0):
            raise ValueError("Field time must be finite and non-negative.")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class BoundaryPatch:
    name: str
    entity_ids: tuple[int, ...]
    physical_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.physical_type.strip():
            raise ValueError("Boundary patches require a name and physical type.")
        if any(entity_id < 0 for entity_id in self.entity_ids):
            raise ValueError("Boundary entity identifiers must be non-negative.")


@dataclass(frozen=True, slots=True)
class MeshSpec:
    mesh_id: str
    points: np.ndarray
    cells: tuple[tuple[int, ...], ...]
    cell_types: tuple[str, ...]
    patches: tuple[BoundaryPatch, ...] = ()
    coordinate_dimensions: PhysicalDimensions = LENGTH
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mesh_id.strip():
            raise ValueError("A mesh requires an identifier.")
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3) or not np.isfinite(points).all():
            raise ValueError("Mesh points must be a finite N-by-2 or N-by-3 array.")
        if not self.cells or len(self.cells) != len(self.cell_types):
            raise ValueError("Each mesh cell requires a matching cell type.")
        point_count = points.shape[0]
        if any(len(cell) < 2 or any(index < 0 or index >= point_count for index in cell) for cell in self.cells):
            raise ValueError("Mesh connectivity references an invalid point.")
        patch_names = [patch.name for patch in self.patches]
        if len(patch_names) != len(set(patch_names)):
            raise ValueError("Boundary patch names must be unique.")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "points", points)
