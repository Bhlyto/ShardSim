from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping

from shardsim.contracts import BoundaryConditions, SimulationCase
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case


SCENARIO_SCHEMA_VERSION = "1.0"
SUPPORTED_MODEL = "heat-2d"
MODEL_VERSION = "heat-2d.explicit-euler-5-point.v1"
_SCENARIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ScenarioValidationError(ValueError):
    """Raised when a scenario does not satisfy the public V1 contract."""


@dataclass(frozen=True, slots=True)
class SolverConfig:
    backend: str
    grid_shape: tuple[int, int]
    safety_factor: float = 0.9


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    alpha: float
    t_end: float
    extent: tuple[float, float]
    initial_shape: tuple[int, int]
    center: tuple[float, float]
    sigma: tuple[float, float]
    amplitude: float
    baseline: float
    boundaries: BoundaryConditions
    solver: SolverConfig
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCENARIO_SCHEMA_VERSION
    model: str = SUPPORTED_MODEL

    def to_case(self) -> SimulationCase:
        initial_field = gaussian_initial_field(
            self.initial_shape,
            center=self.center,
            sigma=self.sigma,
            amplitude=self.amplitude,
            baseline=self.baseline,
        )
        return make_heat_case(
            case_id=self.scenario_id,
            alpha=self.alpha,
            t_end=self.t_end,
            initial_field=initial_field,
            boundaries=self.boundaries,
            extent=self.extent,
            metadata={**self.metadata, "seed": self.seed, "scenario_schema_version": self.schema_version},
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "model": self.model,
            "parameters": {
                "alpha": self.alpha,
                "t_end": self.t_end,
                "extent": list(self.extent),
            },
            "initial_conditions": {
                "type": "gaussian",
                "shape": list(self.initial_shape),
                "center": list(self.center),
                "sigma": list(self.sigma),
                "amplitude": self.amplitude,
                "baseline": self.baseline,
            },
            "boundary_conditions": {
                "top": self.boundaries.top,
                "bottom": self.boundaries.bottom,
                "left": self.boundaries.left,
                "right": self.boundaries.right,
            },
            "solver": {
                "backend": self.solver.backend,
                "grid_shape": list(self.solver.grid_shape),
                "safety_factor": self.solver.safety_factor,
            },
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


def load_scenario(path: str | Path) -> Scenario:
    scenario_path = Path(path)
    try:
        text = scenario_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ScenarioValidationError(f"Cannot read scenario {scenario_path}: {error}") from error
    try:
        payload = json.loads(
            text,
            parse_constant=lambda value: (_raise_invalid_number(value)),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ScenarioValidationError(f"Invalid JSON in {scenario_path}: {error}") from error
    return parse_scenario(payload)


def parse_scenario(payload: Any) -> Scenario:
    root = _object(payload, "$", required={
        "schema_version",
        "scenario_id",
        "model",
        "parameters",
        "initial_conditions",
        "boundary_conditions",
        "solver",
    }, optional={"seed", "metadata"})
    if root["schema_version"] != SCENARIO_SCHEMA_VERSION:
        raise ScenarioValidationError(
            f"$.schema_version: expected {SCENARIO_SCHEMA_VERSION!r}, got {root['schema_version']!r}"
        )
    scenario_id = root["scenario_id"]
    if not isinstance(scenario_id, str) or not _SCENARIO_ID.fullmatch(scenario_id):
        raise ScenarioValidationError(
            "$.scenario_id: use 1-128 ASCII letters, digits, '.', '_' or '-'"
        )
    if root["model"] != SUPPORTED_MODEL:
        raise ScenarioValidationError(
            f"$.model: V1 supports only {SUPPORTED_MODEL!r}"
        )

    parameters = _object(
        root["parameters"],
        "$.parameters",
        required={"alpha", "t_end"},
        optional={"extent"},
    )
    alpha = _finite_number(parameters["alpha"], "$.parameters.alpha", positive=True)
    t_end = _finite_number(parameters["t_end"], "$.parameters.t_end", positive=True)
    extent = _number_pair(parameters.get("extent", [1.0, 1.0]), "$.parameters.extent", positive=True)

    initial = _object(
        root["initial_conditions"],
        "$.initial_conditions",
        required={"type", "shape"},
        optional={"center", "sigma", "amplitude", "baseline"},
    )
    if initial["type"] != "gaussian":
        raise ScenarioValidationError("$.initial_conditions.type: V1 supports only 'gaussian'")
    initial_shape = _shape(initial["shape"], "$.initial_conditions.shape")
    center = _number_pair(initial.get("center", [0.5, 0.5]), "$.initial_conditions.center")
    if any(value < 0.0 or value > 1.0 for value in center):
        raise ScenarioValidationError("$.initial_conditions.center: values must lie in [0, 1]")
    sigma = _number_pair(initial.get("sigma", [0.1, 0.1]), "$.initial_conditions.sigma", positive=True)
    amplitude = _finite_number(initial.get("amplitude", 1.0), "$.initial_conditions.amplitude")
    baseline = _finite_number(initial.get("baseline", 0.0), "$.initial_conditions.baseline")

    boundary = _object(
        root["boundary_conditions"],
        "$.boundary_conditions",
        required={"top", "bottom", "left", "right"},
    )
    boundaries = BoundaryConditions(
        top=_finite_number(boundary["top"], "$.boundary_conditions.top"),
        bottom=_finite_number(boundary["bottom"], "$.boundary_conditions.bottom"),
        left=_finite_number(boundary["left"], "$.boundary_conditions.left"),
        right=_finite_number(boundary["right"], "$.boundary_conditions.right"),
    )

    solver_payload = _object(
        root["solver"],
        "$.solver",
        required={"backend", "grid_shape"},
        optional={"safety_factor"},
    )
    if solver_payload["backend"] != "internal":
        raise ScenarioValidationError("$.solver.backend: V1 supports only 'internal'")
    solver = SolverConfig(
        backend="internal",
        grid_shape=_shape(solver_payload["grid_shape"], "$.solver.grid_shape"),
        safety_factor=_finite_number(
            solver_payload.get("safety_factor", 0.9),
            "$.solver.safety_factor",
            positive=True,
        ),
    )
    if solver.safety_factor > 1.0:
        raise ScenarioValidationError("$.solver.safety_factor: must be at most 1")

    seed = root.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ScenarioValidationError("$.seed: expected an integer or null")
    metadata = root.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ScenarioValidationError("$.metadata: expected an object")

    return Scenario(
        scenario_id=scenario_id,
        alpha=alpha,
        t_end=t_end,
        extent=extent,
        initial_shape=initial_shape,
        center=center,
        sigma=sigma,
        amplitude=amplitude,
        baseline=baseline,
        boundaries=boundaries,
        solver=solver,
        seed=seed,
        metadata=metadata,
    )


def _object(
    value: Any,
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScenarioValidationError(f"{path}: expected an object")
    optional = optional or set()
    missing = sorted(required - value.keys())
    if missing:
        raise ScenarioValidationError(f"{path}: missing required field(s): {', '.join(missing)}")
    unknown = sorted(value.keys() - required - optional)
    if unknown:
        raise ScenarioValidationError(f"{path}: unknown field(s): {', '.join(unknown)}")
    return value


def _finite_number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioValidationError(f"{path}: expected a number")
    number = float(value)
    if not isfinite(number) or (positive and number <= 0.0):
        qualifier = "a finite positive number" if positive else "a finite number"
        raise ScenarioValidationError(f"{path}: expected {qualifier}")
    return number


def _number_pair(value: Any, path: str, *, positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ScenarioValidationError(f"{path}: expected an array of two numbers")
    return (
        _finite_number(value[0], f"{path}[0]", positive=positive),
        _finite_number(value[1], f"{path}[1]", positive=positive),
    )


def _shape(value: Any, path: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 3 for item in value)
    ):
        raise ScenarioValidationError(f"{path}: expected two integers greater than or equal to 3")
    return int(value[0]), int(value[1])


def _raise_invalid_number(value: str) -> None:
    raise ValueError(f"non-finite number {value!r} is not valid JSON")
