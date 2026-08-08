from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping

import numpy as np

from shardsim.canonical import FieldLocation


class Fidelity(str, Enum):
    COARSE = "coarse"
    NOMINAL = "nominal"


@dataclass(frozen=True, slots=True)
class BoundaryConditions:
    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0

    def __post_init__(self) -> None:
        values = (self.top, self.bottom, self.left, self.right)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Boundary values must be finite.")


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    domain: str
    equation: str
    parameters: Mapping[str, float]
    t_end: float
    extent: tuple[float, float] = (1.0, 1.0)
    input_units: Mapping[str, str] = field(default_factory=dict)
    output_units: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("A problem domain is required.")
        if not self.equation.strip():
            raise ValueError("An equation identifier is required.")
        if not isfinite(float(self.t_end)) or self.t_end <= 0:
            raise ValueError("t_end must be a finite positive value.")
        if len(self.extent) != 2 or any(not isfinite(float(v)) or v <= 0 for v in self.extent):
            raise ValueError("extent must contain two finite positive lengths.")
        for name, value in self.parameters.items():
            if not name:
                raise ValueError("Parameter names cannot be empty.")
            if not isfinite(float(value)):
                raise ValueError(f"Parameter {name!r} must be finite.")

    def parameter(self, name: str) -> float:
        try:
            return float(self.parameters[name])
        except KeyError as error:
            raise ValueError(f"Missing required parameter: {name}") from error


@dataclass(frozen=True, slots=True)
class SimulationCase:
    case_id: str
    problem: ProblemSpec
    initial_field: np.ndarray
    boundaries: BoundaryConditions = field(default_factory=BoundaryConditions)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty.")
        initial_field = np.asarray(self.initial_field, dtype=np.float64)
        if initial_field.ndim != 2 or min(initial_field.shape) < 3:
            raise ValueError("initial_field must be a two-dimensional grid of at least 3x3.")
        if not np.isfinite(initial_field).all():
            raise ValueError("initial_field must contain only finite values.")
        initial_field = initial_field.copy()
        initial_field.setflags(write=False)
        object.__setattr__(self, "initial_field", initial_field)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    case_id: str
    fidelity: Fidelity
    field: np.ndarray
    t_end: float
    dt: float
    n_steps: int
    runtime_seconds: float
    field_location: FieldLocation = FieldLocation.POINT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        result_field = np.asarray(self.field, dtype=np.float64)
        if result_field.ndim != 2 or not np.isfinite(result_field).all():
            raise ValueError("A simulation result must be a finite two-dimensional field.")
        if self.n_steps < 1 or self.dt <= 0 or self.t_end <= 0:
            raise ValueError("Simulation time metadata is invalid.")
        if not isinstance(self.field_location, FieldLocation):
            raise ValueError("Simulation field_location must be a FieldLocation.")
        result_field = result_field.copy()
        result_field.setflags(write=False)
        object.__setattr__(self, "field", result_field)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.field.shape


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_id: str
    domain: str
    equation: str
    schema_version: str
    training_case_ids: tuple[str, ...]
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.domain.strip() or not self.equation.strip():
            raise ValueError("Model identity, domain, and equation are required.")
        if not self.training_case_ids:
            raise ValueError("A model descriptor requires at least one training case.")
        if min(self.input_shape) < 2 or min(self.output_shape) < 2:
            raise ValueError("Model shapes must describe two-dimensional fields.")


@dataclass(frozen=True, slots=True)
class Prediction:
    case_id: str
    model_id: str
    mean: np.ndarray
    uncertainty: np.ndarray
    ood_score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        uncertainty = np.asarray(self.uncertainty, dtype=np.float64)
        if mean.shape != uncertainty.shape or mean.ndim != 2:
            raise ValueError("Prediction mean and uncertainty must share a two-dimensional shape.")
        if not np.isfinite(mean).all() or not np.isfinite(uncertainty).all():
            raise ValueError("Prediction arrays must be finite.")
        if np.any(uncertainty < 0):
            raise ValueError("Prediction uncertainty cannot be negative.")
        if not isfinite(float(self.ood_score)) or self.ood_score < 0:
            raise ValueError("ood_score must be finite and non-negative.")
        mean = mean.copy()
        uncertainty = uncertainty.copy()
        mean.setflags(write=False)
        uncertainty.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "uncertainty", uncertainty)
