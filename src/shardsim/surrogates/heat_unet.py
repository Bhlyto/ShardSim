from __future__ import annotations

import json
import random
from math import sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from shardsim.canonical import FieldLocation
from shardsim.contracts import ModelDescriptor, Prediction, SimulationCase
from shardsim.pipeline import ReferenceSample
from shardsim.surrogates.features import model_features

try:
    import torch
    from torch import nn
    from torch.nn import functional as functional
except ImportError as error:  # pragma: no cover - exercised only without the optional extra
    raise ImportError(
        "HeatResidualUNetSurrogate requires PyTorch. Install ShardSim with the 'cnn' extra."
    ) from error


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class _ResidualUNet(nn.Module):
    def __init__(self, input_channels: int, width: int) -> None:
        super().__init__()
        self.encoder_one = _ConvBlock(input_channels, width)
        self.encoder_two = _ConvBlock(width, 2 * width)
        self.bottleneck = _ConvBlock(2 * width, 4 * width)
        self.decoder_two = _ConvBlock(6 * width, 2 * width)
        self.decoder_one = _ConvBlock(3 * width, width)
        self.output = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoder_one = self.encoder_one(values)
        encoder_two = self.encoder_two(functional.avg_pool2d(encoder_one, 2))
        bottleneck = self.bottleneck(functional.avg_pool2d(encoder_two, 2))
        decoded_two = functional.interpolate(
            bottleneck,
            size=encoder_two.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_two = self.decoder_two(torch.cat((decoded_two, encoder_two), dim=1))
        decoded_one = functional.interpolate(
            decoded_two,
            size=encoder_one.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_one = self.decoder_one(torch.cat((decoded_one, encoder_one), dim=1))
        return self.output(decoded_one)


class HeatResidualUNetSurrogate:
    minimum_training_cases = 4

    def __init__(
        self,
        model_id: str = "heat-2d.residual-unet.v1",
        *,
        width: int = 8,
        epochs: int = 200,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        gradient_weight: float = 0.05,
        uncertainty_ood_scale: float = 0.25,
        seed: int = 0,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        if width < 4 or epochs < 1 or batch_size < 1:
            raise ValueError("width, epochs, and batch_size must be positive.")
        if min(learning_rate, weight_decay, gradient_weight, uncertainty_ood_scale) < 0:
            raise ValueError("CNN optimization parameters must be non-negative.")
        if device not in ("cpu", "cuda"):
            raise ValueError("CNN device must be 'cpu' or 'cuda'.")
        self.model_id = model_id
        self.width = int(width)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_weight = float(gradient_weight)
        self.uncertainty_ood_scale = float(uncertainty_ood_scale)
        self.seed = int(seed)
        self.device = device
        self.deterministic = bool(deterministic)
        self._descriptor: ModelDescriptor | None = None
        self._network: _ResidualUNet | None = None
        self._optimizer_state: Mapping[str, Any] | None = None
        self._field_mean: float | None = None
        self._field_scale: float | None = None
        self._scalar_mean: np.ndarray | None = None
        self._scalar_scale: np.ndarray | None = None
        self._delta_scale: float | None = None
        self._uncertainty: np.ndarray | None = None
        self._global_mean: np.ndarray | None = None
        self._global_scale: np.ndarray | None = None
        self._parameter_names: tuple[str, ...] | None = None
        self._field_location: FieldLocation | None = None
        self._input_channels: int | None = None
        self._epochs_trained = 0
        self._last_loss: float | None = None

    @property
    def descriptor(self) -> ModelDescriptor:
        if self._descriptor is None:
            raise RuntimeError("The heat residual U-Net has not been fitted.")
        return self._descriptor

    def fit(self, samples: Sequence[ReferenceSample]) -> ModelDescriptor:
        training_samples = tuple(samples)
        if len(training_samples) < self.minimum_training_cases:
            raise ValueError(
                f"HeatResidualUNetSurrogate requires at least {self.minimum_training_cases} cases."
            )
        first = training_samples[0]
        parameter_names = tuple(sorted(first.problem.parameters))
        expected_shape = first.nominal.grid_shape
        for sample in training_samples:
            if sample.problem.domain != "heat-2d":
                raise ValueError("HeatResidualUNetSurrogate supports heat-2d only.")
            if sample.problem.equation != "du/dt=alpha*laplacian(u)":
                raise ValueError("HeatResidualUNetSurrogate requires the heat equation.")
            if sample.nominal.grid_shape != expected_shape:
                raise ValueError("CNN training samples must share a nominal grid shape.")
            if sample.nominal.field_location is not first.nominal.field_location:
                raise ValueError("CNN training samples must share a field location.")
            if tuple(sorted(sample.problem.parameters)) != parameter_names:
                raise ValueError("CNN training samples must share a parameter schema.")

        self._configure_reproducibility()
        self._parameter_names = parameter_names
        self._field_location = first.nominal.field_location
        scalar_matrix = np.asarray(
            [_scalar_features(sample.problem, sample.coarse, parameter_names) for sample in training_samples],
            dtype=np.float64,
        )
        coarse_fields = np.stack([sample.coarse_on_nominal for sample in training_samples])
        targets = np.stack([sample.delta for sample in training_samples])
        if self._field_mean is None:
            self._field_mean = float(np.mean(coarse_fields))
            self._field_scale = max(float(np.std(coarse_fields)), 1e-8)
            self._scalar_mean = np.mean(scalar_matrix, axis=0)
            scalar_scale = np.std(scalar_matrix, axis=0, ddof=1)
            self._scalar_scale = np.maximum(
                scalar_scale,
                1e-8 + 0.01 * np.maximum(np.abs(self._scalar_mean), 1e-3),
            )
            self._delta_scale = max(float(np.sqrt(np.mean(np.square(targets)))), 1e-8)
        self._require_normalization()

        input_array = np.stack(
            [
                self._input_channels_for(sample.coarse_on_nominal, sample.problem, sample.coarse)
                for sample in training_samples
            ]
        )
        input_tensor = torch.from_numpy(input_array.astype(np.float32))
        target_tensor = torch.from_numpy(
            (targets[:, None, :, :] / self._delta_scale).astype(np.float32)
        )
        interior_mask = torch.ones((1, 1, *expected_shape), dtype=torch.float32)
        if first.nominal.field_location is FieldLocation.POINT:
            interior_mask[:, :, 0, :] = 0.0
            interior_mask[:, :, -1, :] = 0.0
            interior_mask[:, :, :, 0] = 0.0
            interior_mask[:, :, :, -1] = 0.0

        input_channels = int(input_tensor.shape[1])
        if self._network is None:
            self._network = _ResidualUNet(input_channels, self.width)
            self._input_channels = input_channels
        elif self._input_channels != input_channels:
            raise ValueError("Warm-start CNN input channels do not match the new dataset.")
        training_device = self._training_device()
        network = self._network.to(training_device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self._optimizer_state is not None:
            optimizer.load_state_dict(self._optimizer_state)
            _optimizer_to(optimizer, training_device)
        inputs = input_tensor.to(training_device)
        target_values = target_tensor.to(training_device)
        mask = interior_mask.to(training_device)
        network.train()
        last_loss = 0.0
        for local_epoch in range(self.epochs):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + self._epochs_trained + local_epoch)
            order = torch.randperm(len(training_samples), generator=generator)
            for start in range(0, len(training_samples), self.batch_size):
                indices = order[start : start + self.batch_size].to(training_device)
                prediction = network(inputs.index_select(0, indices)) * mask
                expected = target_values.index_select(0, indices) * mask
                data_loss = torch.mean(torch.square(prediction - expected))
                gradient_loss = _gradient_loss(prediction, expected)
                loss = data_loss + self.gradient_weight * gradient_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
        self._epochs_trained += self.epochs
        self._last_loss = last_loss
        self._optimizer_state = _cpu_tree(optimizer.state_dict())

        network.eval()
        with torch.no_grad():
            predicted = (
                (network(inputs) * mask).detach().cpu().numpy()[:, 0]
                * self._delta_scale
            )
        residuals = targets - predicted
        residual_rmse = np.sqrt(np.mean(np.square(residuals), axis=0))
        uncertainty_floor = 1e-8 + 0.01 * float(np.mean(np.abs(targets)))
        self._uncertainty = np.maximum(residual_rmse, uncertainty_floor)
        global_matrix = np.stack(
            [model_features(sample.coarse_on_nominal, sample.problem, parameter_names) for sample in training_samples]
        )
        self._global_mean = np.mean(global_matrix, axis=0)
        global_scale = np.std(global_matrix, axis=0, ddof=1)
        self._global_scale = np.maximum(
            global_scale,
            1e-8 + 0.02 * np.maximum(np.abs(self._global_mean), 1e-3),
        )
        self._network = network.cpu()
        self._descriptor = ModelDescriptor(
            model_id=self.model_id,
            domain=first.problem.domain,
            equation=first.problem.equation,
            schema_version=first.problem.schema_version,
            training_case_ids=tuple(sample.case_id for sample in training_samples),
            input_shape=expected_shape,
            output_shape=expected_shape,
            metadata={
                "algorithm": "residual-unet-2d",
                "architecture": "two-level-unet",
                "width": self.width,
                "input_channels": self._input_channels,
                "physical_parameters": parameter_names,
                "field_location": self._field_location.value,
                "epochs_trained": self._epochs_trained,
                "last_training_loss": self._last_loss,
                "uncertainty": "cumulative-training-residual-rmse",
            },
        )
        return self._descriptor

    def predict(self, case: SimulationCase, coarse_on_nominal: np.ndarray) -> Prediction:
        descriptor = self.descriptor
        self._require_state()
        coarse = np.asarray(coarse_on_nominal, dtype=np.float64)
        if coarse.shape != descriptor.input_shape:
            raise ValueError(f"Expected field shape {descriptor.input_shape}, received {coarse.shape}.")
        if case.problem.domain != descriptor.domain or case.problem.equation != descriptor.equation:
            raise ValueError("The CNN surrogate is incompatible with this problem.")
        if tuple(sorted(case.problem.parameters)) != self._parameter_names:
            raise ValueError("The CNN parameter schema does not match the problem.")
        synthetic_coarse = _coarse_metadata(case)
        channels = self._input_channels_for(coarse, case.problem, synthetic_coarse)
        network = self._network
        if network is None:
            raise RuntimeError("CNN network state is missing.")
        network.eval()
        with torch.no_grad():
            normalized_delta = network(
                torch.from_numpy(channels[None].astype(np.float32))
            )[0, 0].numpy()
        delta = normalized_delta.astype(np.float64) * float(self._delta_scale)
        if self._field_location is FieldLocation.POINT:
            delta = delta.copy()
            delta[0, :] = 0.0
            delta[-1, :] = 0.0
            delta[:, 0] = 0.0
            delta[:, -1] = 0.0
        mean = coarse + delta
        if self._field_location is FieldLocation.POINT:
            _apply_point_boundaries(mean, case)
        global_vector = model_features(coarse, case.problem, self._parameter_names)
        global_z_score = (global_vector - self._global_mean) / self._global_scale
        ood_score = float(np.linalg.norm(global_z_score) / sqrt(global_z_score.size))
        uncertainty = self._uncertainty * (1.0 + self.uncertainty_ood_scale * ood_score)
        if self._field_location is FieldLocation.POINT:
            uncertainty = uncertainty.copy()
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
                "algorithm": "residual-unet-2d",
            },
        )

    def save(self, path: str | Path) -> Path:
        self._require_state()
        target = Path(path)
        if target.suffix != ".pt":
            raise ValueError("Heat residual U-Net artifacts must use .pt.")
        target.parent.mkdir(parents=True, exist_ok=True)
        network = self._network
        if network is None:
            raise RuntimeError("CNN network state is missing.")
        payload = {
            "artifact_version": 1,
            "descriptor_json": json.dumps(_descriptor_payload(self.descriptor), sort_keys=True),
            "config_json": json.dumps(self._config_payload(), sort_keys=True),
            "network_state": _cpu_tree(network.state_dict()),
            "optimizer_state": _cpu_tree(self._optimizer_state),
            "field_mean": torch.tensor(float(self._field_mean), dtype=torch.float64),
            "field_scale": torch.tensor(float(self._field_scale), dtype=torch.float64),
            "scalar_mean": torch.from_numpy(self._scalar_mean.astype(np.float64)),
            "scalar_scale": torch.from_numpy(self._scalar_scale.astype(np.float64)),
            "delta_scale": torch.tensor(float(self._delta_scale), dtype=torch.float64),
            "uncertainty": torch.from_numpy(self._uncertainty.astype(np.float64)),
            "global_mean": torch.from_numpy(self._global_mean.astype(np.float64)),
            "global_scale": torch.from_numpy(self._global_scale.astype(np.float64)),
            "parameter_names_json": json.dumps(self._parameter_names),
            "field_location": self._field_location.value,
            "input_channels": int(self._input_channels),
            "epochs_trained": int(self._epochs_trained),
            "last_loss": torch.tensor(float(self._last_loss), dtype=torch.float64),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "HeatResidualUNetSurrogate":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if int(payload.get("artifact_version", 0)) != 1:
            raise ValueError("Unsupported heat residual U-Net artifact version.")
        config = json.loads(payload["config_json"])
        model = cls(**config)
        descriptor = json.loads(payload["descriptor_json"])
        model._input_channels = int(payload["input_channels"])
        model._network = _ResidualUNet(model._input_channels, model.width)
        model._network.load_state_dict(payload["network_state"])
        model._optimizer_state = payload["optimizer_state"]
        model._field_mean = float(payload["field_mean"])
        model._field_scale = float(payload["field_scale"])
        model._scalar_mean = payload["scalar_mean"].numpy().astype(np.float64)
        model._scalar_scale = payload["scalar_scale"].numpy().astype(np.float64)
        model._delta_scale = float(payload["delta_scale"])
        model._uncertainty = payload["uncertainty"].numpy().astype(np.float64)
        model._global_mean = payload["global_mean"].numpy().astype(np.float64)
        model._global_scale = payload["global_scale"].numpy().astype(np.float64)
        model._parameter_names = tuple(json.loads(payload["parameter_names_json"]))
        model._field_location = FieldLocation(payload["field_location"])
        model._epochs_trained = int(payload["epochs_trained"])
        model._last_loss = float(payload["last_loss"])
        model._descriptor = ModelDescriptor(
            model_id=descriptor["model_id"],
            domain=descriptor["domain"],
            equation=descriptor["equation"],
            schema_version=descriptor["schema_version"],
            training_case_ids=tuple(descriptor["training_case_ids"]),
            input_shape=tuple(descriptor["input_shape"]),
            output_shape=tuple(descriptor["output_shape"]),
            metadata=descriptor["metadata"],
        )
        return model

    def _configure_reproducibility(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = self.deterministic
        torch.use_deterministic_algorithms(self.deterministic)
        if self.deterministic and self.device == "cpu":
            torch.set_num_threads(1)

    def _training_device(self) -> torch.device:
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for CNN training but is unavailable.")
        return torch.device(self.device)

    def _input_channels_for(
        self,
        field: np.ndarray,
        problem: Any,
        coarse_metadata: Any,
    ) -> np.ndarray:
        self._require_normalization()
        values = np.asarray(field, dtype=np.float64)
        rows, columns = values.shape
        x = np.linspace(-1.0, 1.0, columns)
        y = np.linspace(-1.0, 1.0, rows)
        x_grid, y_grid = np.meshgrid(x, y)
        scalars = _scalar_features(problem, coarse_metadata, self._parameter_names)
        normalized_scalars = (scalars - self._scalar_mean) / self._scalar_scale
        scalar_channels = [np.full_like(values, scalar) for scalar in normalized_scalars]
        return np.stack(
            [
                (values - self._field_mean) / self._field_scale,
                x_grid,
                y_grid,
                *scalar_channels,
            ]
        )

    def _require_normalization(self) -> None:
        values = (
            self._field_mean,
            self._field_scale,
            self._scalar_mean,
            self._scalar_scale,
            self._delta_scale,
        )
        if any(value is None for value in values):
            raise RuntimeError("CNN normalization state is incomplete.")

    def _require_state(self) -> None:
        self._require_normalization()
        values = (
            self._descriptor,
            self._network,
            self._uncertainty,
            self._global_mean,
            self._global_scale,
            self._parameter_names,
            self._field_location,
            self._input_channels,
            self._last_loss,
        )
        if any(value is None for value in values):
            raise RuntimeError("CNN surrogate state is incomplete.")

    def _config_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "width": self.width,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_weight": self.gradient_weight,
            "uncertainty_ood_scale": self.uncertainty_ood_scale,
            "seed": self.seed,
            "device": self.device,
            "deterministic": self.deterministic,
        }


def _scalar_features(problem: Any, coarse: Any, parameter_names: Sequence[str]) -> np.ndarray:
    boundaries = getattr(coarse, "metadata", {}).get("boundaries")
    if boundaries is None:
        field = getattr(coarse, "field", None)
        if field is None:
            boundaries = (0.0, 0.0, 0.0, 0.0)
        else:
            values = np.asarray(field, dtype=np.float64)
            boundaries = (
                float(np.mean(values[0, 1:-1])),
                float(np.mean(values[-1, 1:-1])),
                float(np.mean(values[1:-1, 0])),
                float(np.mean(values[1:-1, -1])),
            )
    return np.asarray(
        [
            *(problem.parameter(name) for name in parameter_names),
            problem.t_end,
            problem.extent[0],
            problem.extent[1],
            *boundaries,
        ],
        dtype=np.float64,
    )


def _coarse_metadata(case: SimulationCase) -> Any:
    class _Metadata:
        metadata = {
            "boundaries": (
                case.boundaries.top,
                case.boundaries.bottom,
                case.boundaries.left,
                case.boundaries.right,
            )
        }

    return _Metadata()


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    prediction_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (
        torch.mean(torch.square(prediction_x - target_x))
        + torch.mean(torch.square(prediction_y - target_y))
    )


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _descriptor_payload(descriptor: ModelDescriptor) -> dict[str, Any]:
    return {
        "model_id": descriptor.model_id,
        "domain": descriptor.domain,
        "equation": descriptor.equation,
        "schema_version": descriptor.schema_version,
        "training_case_ids": descriptor.training_case_ids,
        "input_shape": descriptor.input_shape,
        "output_shape": descriptor.output_shape,
        "metadata": descriptor.metadata,
    }


def _apply_point_boundaries(field: np.ndarray, case: SimulationCase) -> None:
    field[0, 1:-1] = case.boundaries.top
    field[-1, 1:-1] = case.boundaries.bottom
    field[1:-1, 0] = case.boundaries.left
    field[1:-1, -1] = case.boundaries.right
    field[0, 0] = 0.5 * (case.boundaries.top + case.boundaries.left)
    field[0, -1] = 0.5 * (case.boundaries.top + case.boundaries.right)
    field[-1, 0] = 0.5 * (case.boundaries.bottom + case.boundaries.left)
    field[-1, -1] = 0.5 * (case.boundaries.bottom + case.boundaries.right)
