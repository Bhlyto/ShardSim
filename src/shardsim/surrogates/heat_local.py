from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from shardsim.canonical import FieldLocation
from shardsim.contracts import ModelDescriptor, Prediction, ProblemSpec, SimulationCase
from shardsim.pipeline import ReferenceSample
from shardsim.surrogates.features import model_features


LOCAL_FEATURE_NAMES = (
    "temperature",
    "gradient_x",
    "gradient_y",
    "gradient_magnitude",
    "laplacian",
    "x",
    "y",
    "boundary_distance",
    "temperature_x",
    "temperature_y",
    "alpha",
    "t_end",
    "extent_x",
    "extent_y",
    "alpha_t_end",
    "alpha_t_end_laplacian",
)


class HeatLocalResidualSurrogate:
    def __init__(
        self,
        model_id: str = "heat-2d.local-residual.v1",
        ridge: float = 0.03,
        uncertainty_ood_scale: float = 0.25,
    ) -> None:
        if ridge < 0 or uncertainty_ood_scale < 0:
            raise ValueError("Regularization and uncertainty OOD scale must be non-negative.")
        self.model_id = model_id
        self.ridge = float(ridge)
        self.uncertainty_ood_scale = float(uncertainty_ood_scale)
        self._descriptor: ModelDescriptor | None = None
        self._coefficients: np.ndarray | None = None
        self._local_mean: np.ndarray | None = None
        self._local_scale: np.ndarray | None = None
        self._global_mean: np.ndarray | None = None
        self._global_scale: np.ndarray | None = None
        self._uncertainty: np.ndarray | None = None
        self._parameter_names: tuple[str, ...] | None = None
        self._field_location: FieldLocation | None = None

    @property
    def descriptor(self) -> ModelDescriptor:
        if self._descriptor is None:
            raise RuntimeError("The heat local residual surrogate has not been fitted.")
        return self._descriptor

    def fit(self, samples: Sequence[ReferenceSample]) -> ModelDescriptor:
        training_samples = tuple(samples)
        if len(training_samples) < 2:
            raise ValueError("At least two reference samples are required.")
        first = training_samples[0]
        if first.problem.domain != "heat-2d" or first.problem.equation != "du/dt=alpha*laplacian(u)":
            raise ValueError("HeatLocalResidualSurrogate only supports the 2-D heat equation.")
        expected_shape = first.nominal.grid_shape
        parameter_names = tuple(sorted(first.problem.parameters))
        field_location = first.nominal.field_location
        for sample in training_samples:
            if sample.problem.domain != first.problem.domain or sample.problem.equation != first.problem.equation:
                raise ValueError("Training samples must share the heat domain and equation.")
            if sample.problem.schema_version != first.problem.schema_version:
                raise ValueError("Training samples must share a schema version.")
            if sample.nominal.grid_shape != expected_shape:
                raise ValueError("Training samples must share a nominal grid shape.")
            if sample.nominal.field_location is not field_location:
                raise ValueError("Training samples must share a field location.")
            if tuple(sorted(sample.problem.parameters)) != parameter_names:
                raise ValueError("Training samples must share a parameter schema.")

        local_blocks = [
            _local_features(sample.coarse_on_nominal, sample.case_id, sample.problem)
            for sample in training_samples
        ]
        local_matrix = np.concatenate(local_blocks, axis=0)
        targets = np.concatenate([sample.delta.ravel() for sample in training_samples])
        local_mean = np.mean(local_matrix, axis=0)
        local_scale = np.std(local_matrix, axis=0)
        local_scale = np.where(local_scale < 1e-10, 1.0, local_scale)
        standardized = (local_matrix - local_mean) / local_scale
        design = np.column_stack([np.ones(standardized.shape[0]), standardized])
        if self.ridge == 0:
            coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        else:
            penalty = np.eye(design.shape[1]) * self.ridge * design.shape[0]
            penalty[0, 0] = 0.0
            coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)

        global_matrix = np.stack(
            [
                model_features(sample.coarse_on_nominal, sample.problem, parameter_names)
                for sample in training_samples
            ]
        )
        global_mean = np.mean(global_matrix, axis=0)
        global_scale = np.std(global_matrix, axis=0, ddof=1)
        global_floor = 1e-6 + 0.02 * np.maximum(np.abs(global_mean), 1e-3)
        global_scale = np.maximum(global_scale, global_floor)

        predicted_deltas = []
        for block in local_blocks:
            block_design = np.column_stack(
                [np.ones(block.shape[0]), (block - local_mean) / local_scale]
            )
            predicted_deltas.append((block_design @ coefficients).reshape(expected_shape))
        residuals = np.stack(
            [sample.delta - prediction for sample, prediction in zip(training_samples, predicted_deltas)]
        )
        residual_rmse = np.sqrt(np.mean(np.square(residuals), axis=0))
        delta_spread = np.std(np.stack([sample.delta for sample in training_samples]), axis=0, ddof=1)
        uncertainty = np.sqrt(np.square(residual_rmse) + 0.25 * np.square(delta_spread))
        uncertainty_floor = 1e-8 + 0.01 * float(np.mean(np.abs(targets)))

        self._coefficients = coefficients
        self._local_mean = local_mean
        self._local_scale = local_scale
        self._global_mean = global_mean
        self._global_scale = global_scale
        self._uncertainty = np.maximum(uncertainty, uncertainty_floor)
        self._parameter_names = parameter_names
        self._field_location = field_location
        self._descriptor = ModelDescriptor(
            model_id=self.model_id,
            domain=first.problem.domain,
            equation=first.problem.equation,
            schema_version=first.problem.schema_version,
            training_case_ids=tuple(sample.case_id for sample in training_samples),
            input_shape=expected_shape,
            output_shape=expected_shape,
            metadata={
                "algorithm": "shared-local-ridge-residual",
                "ridge": self.ridge,
                "field_location": field_location.value,
                "local_features": LOCAL_FEATURE_NAMES,
                "physical_parameters": parameter_names,
                "uncertainty": "training-residual-and-delta-spread",
            },
        )
        return self._descriptor

    def predict(self, case: SimulationCase, coarse_on_nominal: np.ndarray) -> Prediction:
        descriptor = self.descriptor
        state = (
            self._coefficients,
            self._local_mean,
            self._local_scale,
            self._global_mean,
            self._global_scale,
            self._uncertainty,
            self._parameter_names,
            self._field_location,
        )
        if any(value is None for value in state):
            raise RuntimeError("The heat local residual surrogate state is incomplete.")
        coarse = np.asarray(coarse_on_nominal, dtype=np.float64)
        if coarse.shape != descriptor.input_shape:
            raise ValueError(f"Expected field shape {descriptor.input_shape}, received {coarse.shape}.")
        if case.problem.domain != descriptor.domain or case.problem.equation != descriptor.equation:
            raise ValueError("The surrogate is not compatible with this problem.")
        if tuple(sorted(case.problem.parameters)) != self._parameter_names:
            raise ValueError("The problem parameter schema does not match the surrogate.")

        local = _local_features(coarse, case.case_id, case.problem)
        design = np.column_stack(
            [np.ones(local.shape[0]), (local - self._local_mean) / self._local_scale]
        )
        delta = (design @ self._coefficients).reshape(descriptor.output_shape)
        mean = coarse + delta
        global_vector = model_features(coarse, case.problem, self._parameter_names)
        global_z_score = (global_vector - self._global_mean) / self._global_scale
        ood_score = float(np.linalg.norm(global_z_score) / sqrt(global_z_score.size))
        uncertainty = self._uncertainty * (1.0 + self.uncertainty_ood_scale * ood_score)
        if self._field_location is FieldLocation.POINT:
            mean = mean.copy()
            uncertainty = uncertainty.copy()
            _apply_point_boundaries(mean, case)
            uncertainty[0, :] = 0.0
            uncertainty[-1, :] = 0.0
            uncertainty[:, 0] = 0.0
            uncertainty[:, -1] = 0.0
        return Prediction(
            case_id=case.case_id,
            model_id=descriptor.model_id,
            mean=mean,
            uncertainty=uncertainty,
            ood_score=ood_score,
            metadata={
                "domain": descriptor.domain,
                "equation": descriptor.equation,
                "schema_version": descriptor.schema_version,
                "field_location": self._field_location.value,
            },
        )

    def save(self, path: str | Path) -> Path:
        descriptor = self.descriptor
        target = Path(path)
        if target.suffix != ".npz":
            raise ValueError("HeatLocalResidualSurrogate artifacts must use .npz.")
        state = (
            self._coefficients,
            self._local_mean,
            self._local_scale,
            self._global_mean,
            self._global_scale,
            self._uncertainty,
            self._parameter_names,
            self._field_location,
        )
        if any(value is None for value in state):
            raise RuntimeError("The heat local residual surrogate state is incomplete.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_version": 1,
            "model_id": descriptor.model_id,
            "domain": descriptor.domain,
            "equation": descriptor.equation,
            "schema_version": descriptor.schema_version,
            "training_case_ids": descriptor.training_case_ids,
            "input_shape": descriptor.input_shape,
            "output_shape": descriptor.output_shape,
            "metadata": descriptor.metadata,
            "parameter_names": self._parameter_names,
            "field_location": self._field_location.value,
            "ridge": self.ridge,
            "uncertainty_ood_scale": self.uncertainty_ood_scale,
        }
        np.savez_compressed(
            target,
            coefficients=self._coefficients,
            local_mean=self._local_mean,
            local_scale=self._local_scale,
            global_mean=self._global_mean,
            global_scale=self._global_scale,
            uncertainty=self._uncertainty,
            descriptor_json=np.array(json.dumps(payload)),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "HeatLocalResidualSurrogate":
        with np.load(Path(path), allow_pickle=False) as artifact:
            payload = json.loads(str(artifact["descriptor_json"].item()))
            if payload.get("artifact_version") != 1:
                raise ValueError("Unsupported heat local residual artifact version.")
            model = cls(
                model_id=payload["model_id"],
                ridge=payload["ridge"],
                uncertainty_ood_scale=payload["uncertainty_ood_scale"],
            )
            model._coefficients = artifact["coefficients"].astype(np.float64)
            model._local_mean = artifact["local_mean"].astype(np.float64)
            model._local_scale = artifact["local_scale"].astype(np.float64)
            model._global_mean = artifact["global_mean"].astype(np.float64)
            model._global_scale = artifact["global_scale"].astype(np.float64)
            model._uncertainty = artifact["uncertainty"].astype(np.float64)
            model._parameter_names = tuple(payload["parameter_names"])
            model._field_location = FieldLocation(payload["field_location"])
            model._descriptor = ModelDescriptor(
                model_id=payload["model_id"],
                domain=payload["domain"],
                equation=payload["equation"],
                schema_version=payload["schema_version"],
                training_case_ids=tuple(payload["training_case_ids"]),
                input_shape=tuple(payload["input_shape"]),
                output_shape=tuple(payload["output_shape"]),
                metadata=payload["metadata"],
            )
        return model


def _local_features(field: np.ndarray, case_id: str, problem: ProblemSpec) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 3 or not np.isfinite(values).all():
        raise ValueError(f"Case {case_id!r} requires a finite field of at least 3x3.")
    rows, columns = values.shape
    dy = 1.0 / max(rows - 1, 1)
    dx = 1.0 / max(columns - 1, 1)
    gradient_y, gradient_x = np.gradient(values, dy, dx, edge_order=2)
    laplacian = np.gradient(gradient_x, dx, axis=1, edge_order=2) + np.gradient(
        gradient_y,
        dy,
        axis=0,
        edge_order=2,
    )
    x = np.linspace(0.0, 1.0, columns)
    y = np.linspace(0.0, 1.0, rows)
    x_grid, y_grid = np.meshgrid(x, y)
    boundary_distance = np.minimum.reduce(
        [x_grid, 1.0 - x_grid, y_grid, 1.0 - y_grid]
    )
    alpha = problem.parameter("alpha")
    t_end = problem.t_end
    extent_x, extent_y = problem.extent
    diffusion_time = alpha * t_end
    channels = (
        values,
        gradient_x,
        gradient_y,
        np.sqrt(np.square(gradient_x) + np.square(gradient_y)),
        laplacian,
        x_grid,
        y_grid,
        boundary_distance,
        values * x_grid,
        values * y_grid,
        np.full_like(values, alpha),
        np.full_like(values, t_end),
        np.full_like(values, extent_x),
        np.full_like(values, extent_y),
        np.full_like(values, diffusion_time),
        diffusion_time * laplacian,
    )
    return np.column_stack([channel.ravel() for channel in channels])


def _apply_point_boundaries(field: np.ndarray, case: SimulationCase) -> None:
    boundaries = case.boundaries
    field[0, 1:-1] = boundaries.top
    field[-1, 1:-1] = boundaries.bottom
    field[1:-1, 0] = boundaries.left
    field[1:-1, -1] = boundaries.right
    field[0, 0] = 0.5 * (boundaries.top + boundaries.left)
    field[0, -1] = 0.5 * (boundaries.top + boundaries.right)
    field[-1, 0] = 0.5 * (boundaries.bottom + boundaries.left)
    field[-1, -1] = 0.5 * (boundaries.bottom + boundaries.right)
