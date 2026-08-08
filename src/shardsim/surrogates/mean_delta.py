from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from shardsim.contracts import ModelDescriptor, Prediction, ProblemSpec, SimulationCase
from shardsim.pipeline import ReferenceSample
from shardsim.surrogates.features import field_summary_features, model_features


def _field_features(field: np.ndarray) -> np.ndarray:
    return field_summary_features(field)


def _model_features(
    field: np.ndarray,
    problem: ProblemSpec,
    parameter_names: tuple[str, ...],
) -> np.ndarray:
    return model_features(field, problem, parameter_names)


class MeanDeltaSurrogate:
    def __init__(self, model_id: str = "heat-2d.mean-delta.v1") -> None:
        self.model_id = model_id
        self._descriptor: ModelDescriptor | None = None
        self._mean_delta: np.ndarray | None = None
        self._delta_std: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_scale: np.ndarray | None = None
        self._parameter_names: tuple[str, ...] | None = None

    @property
    def descriptor(self) -> ModelDescriptor:
        if self._descriptor is None:
            raise RuntimeError("The surrogate has not been fitted.")
        return self._descriptor

    def fit(self, samples: Sequence[ReferenceSample]) -> ModelDescriptor:
        training_samples = tuple(samples)
        if len(training_samples) < 2:
            raise ValueError("At least two reference samples are required to estimate uncertainty.")

        first = training_samples[0]
        expected_shape = first.nominal.grid_shape
        parameter_names = tuple(sorted(first.problem.parameters))
        for sample in training_samples:
            if sample.problem.domain != first.problem.domain:
                raise ValueError("Training samples must share a physical domain.")
            if sample.problem.equation != first.problem.equation:
                raise ValueError("Training samples must share an equation.")
            if sample.problem.schema_version != first.problem.schema_version:
                raise ValueError("Training samples must share a schema version.")
            if sample.nominal.grid_shape != expected_shape:
                raise ValueError("Training samples must share a nominal grid shape.")
            if sample.nominal.field_location is not first.nominal.field_location:
                raise ValueError("Training samples must share a nominal field location.")
            if tuple(sorted(sample.problem.parameters)) != parameter_names:
                raise ValueError("Training samples must share a parameter schema.")

        deltas = np.stack([sample.delta for sample in training_samples])
        features = np.stack(
            [
                _model_features(sample.coarse_on_nominal, sample.problem, parameter_names)
                for sample in training_samples
            ]
        )
        feature_mean = np.mean(features, axis=0)
        feature_scale = np.std(features, axis=0, ddof=1)
        scale_floor = 1e-6 + 0.02 * np.maximum(np.abs(feature_mean), 1e-3)

        self._mean_delta = np.mean(deltas, axis=0)
        self._delta_std = np.std(deltas, axis=0, ddof=1)
        self._feature_mean = feature_mean
        self._feature_scale = np.maximum(feature_scale, scale_floor)
        self._parameter_names = parameter_names
        self._descriptor = ModelDescriptor(
            model_id=self.model_id,
            domain=first.problem.domain,
            equation=first.problem.equation,
            schema_version=first.problem.schema_version,
            training_case_ids=tuple(sample.case_id for sample in training_samples),
            input_shape=expected_shape,
            output_shape=expected_shape,
            metadata={
                "algorithm": "mean-delta",
                "uncertainty": "per-cell-sample-standard-deviation",
                "physical_parameters": parameter_names,
                "field_location": first.nominal.field_location.value,
                "ood_features": (
                    *parameter_names,
                    "t_end",
                    "extent_x",
                    "extent_y",
                    "field_mean",
                    "field_std",
                    "field_min",
                    "field_max",
                    "field_rms",
                    "field_center_x",
                    "field_center_y",
                    "field_spread_x",
                    "field_spread_y",
                ),
            },
        )
        return self._descriptor

    def predict(self, case: SimulationCase, coarse_on_nominal: np.ndarray) -> Prediction:
        descriptor = self.descriptor
        if any(
            value is None
            for value in (
                self._mean_delta,
                self._delta_std,
                self._feature_mean,
                self._feature_scale,
                self._parameter_names,
            )
        ):
            raise RuntimeError("The surrogate state is incomplete.")

        coarse = np.asarray(coarse_on_nominal, dtype=np.float64)
        if coarse.shape != descriptor.input_shape:
            raise ValueError(
                f"Expected coarse field shape {descriptor.input_shape}, received {coarse.shape}."
            )

        if case.problem.domain != descriptor.domain or case.problem.equation != descriptor.equation:
            raise ValueError("The surrogate is not compatible with this problem.")
        if tuple(sorted(case.problem.parameters)) != self._parameter_names:
            raise ValueError("The problem parameter schema does not match the surrogate.")

        features = _model_features(coarse, case.problem, self._parameter_names)
        feature_z_score = (features - self._feature_mean) / self._feature_scale
        ood_score = float(np.linalg.norm(feature_z_score) / sqrt(feature_z_score.size))
        return Prediction(
            case_id=case.case_id,
            model_id=descriptor.model_id,
            mean=coarse + self._mean_delta,
            uncertainty=self._delta_std,
            ood_score=ood_score,
            metadata={
                "domain": descriptor.domain,
                "equation": descriptor.equation,
                "schema_version": descriptor.schema_version,
            },
        )

    def save(self, path: str | Path) -> Path:
        descriptor = self.descriptor
        target = Path(path)
        if target.suffix != ".npz":
            raise ValueError("MeanDeltaSurrogate artifacts must use the .npz extension.")
        if any(
            value is None
            for value in (
                self._mean_delta,
                self._delta_std,
                self._feature_mean,
                self._feature_scale,
                self._parameter_names,
            )
        ):
            raise RuntimeError("The surrogate state is incomplete.")

        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor_payload = {
            "model_id": descriptor.model_id,
            "domain": descriptor.domain,
            "equation": descriptor.equation,
            "schema_version": descriptor.schema_version,
            "training_case_ids": descriptor.training_case_ids,
            "input_shape": descriptor.input_shape,
            "output_shape": descriptor.output_shape,
            "metadata": descriptor.metadata,
            "parameter_names": self._parameter_names,
            "artifact_version": 2,
        }
        np.savez_compressed(
            target,
            mean_delta=self._mean_delta,
            delta_std=self._delta_std,
            feature_mean=self._feature_mean,
            feature_scale=self._feature_scale,
            descriptor_json=np.array(json.dumps(descriptor_payload)),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "MeanDeltaSurrogate":
        source = Path(path)
        with np.load(source, allow_pickle=False) as artifact:
            payload = json.loads(str(artifact["descriptor_json"].item()))
            if payload.get("artifact_version") != 2:
                raise ValueError("Unsupported surrogate artifact version.")
            model = cls(model_id=payload["model_id"])
            model._mean_delta = artifact["mean_delta"].astype(np.float64)
            model._delta_std = artifact["delta_std"].astype(np.float64)
            model._feature_mean = artifact["feature_mean"].astype(np.float64)
            model._feature_scale = artifact["feature_scale"].astype(np.float64)
            model._parameter_names = tuple(payload["parameter_names"])
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
