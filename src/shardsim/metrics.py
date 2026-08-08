from __future__ import annotations

from typing import Mapping

import numpy as np

from shardsim.contracts import BoundaryConditions


def compare_fields(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if predicted.shape != expected.shape:
        raise ValueError("prediction and target must share the same shape.")
    difference = predicted - expected
    target_norm = float(np.linalg.norm(expected.ravel()))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / max(target_norm, 1e-12)),
    }


def boundary_residual(field: np.ndarray, boundaries: BoundaryConditions) -> float:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 3:
        raise ValueError("field must be a two-dimensional grid of at least 3x3.")
    residuals = np.concatenate(
        [
            np.abs(values[0, 1:-1] - boundaries.top),
            np.abs(values[-1, 1:-1] - boundaries.bottom),
            np.abs(values[1:-1, 0] - boundaries.left),
            np.abs(values[1:-1, -1] - boundaries.right),
        ]
    )
    return float(np.max(residuals, initial=0.0))


def compare_gradients(
    prediction: np.ndarray,
    target: np.ndarray,
    extent: tuple[float, float],
) -> Mapping[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 2:
        raise ValueError("prediction and target must share a two-dimensional shape.")
    if len(extent) != 2 or any(float(length) <= 0 for length in extent):
        raise ValueError("extent must contain two positive lengths.")
    spacing_y = float(extent[1]) / max(predicted.shape[0] - 1, 1)
    spacing_x = float(extent[0]) / max(predicted.shape[1] - 1, 1)
    predicted_y, predicted_x = np.gradient(
        predicted, spacing_y, spacing_x, edge_order=2
    )
    expected_y, expected_x = np.gradient(
        expected, spacing_y, spacing_x, edge_order=2
    )
    difference_norm = float(
        np.sqrt(
            np.sum(np.square(predicted_x - expected_x))
            + np.sum(np.square(predicted_y - expected_y))
        )
    )
    target_norm = float(
        np.sqrt(np.sum(np.square(expected_x)) + np.sum(np.square(expected_y)))
    )
    return {
        "gradient_relative_l2": difference_norm / max(target_norm, 1e-12),
        "gradient_rmse": float(
            np.sqrt(
                0.5
                * (
                    np.mean(np.square(predicted_x - expected_x))
                    + np.mean(np.square(predicted_y - expected_y))
                )
            )
        ),
    }


def maximum_principle_violation(
    field: np.ndarray,
    initial_field: np.ndarray,
    boundaries: BoundaryConditions,
) -> Mapping[str, float]:
    values = np.asarray(field, dtype=np.float64)
    initial = np.asarray(initial_field, dtype=np.float64)
    if values.ndim != 2 or initial.ndim != 2:
        raise ValueError("field and initial_field must be two-dimensional.")
    boundary_values = np.asarray(
        [boundaries.top, boundaries.bottom, boundaries.left, boundaries.right],
        dtype=np.float64,
    )
    lower = float(min(np.min(initial), np.min(boundary_values)))
    upper = float(max(np.max(initial), np.max(boundary_values)))
    violation = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
    scale = max(upper - lower, 1e-12)
    return {
        "maximum_principle_max_violation": float(np.max(violation, initial=0.0)),
        "maximum_principle_mean_violation": float(np.mean(violation)),
        "maximum_principle_relative_violation": float(
            np.max(violation, initial=0.0) / scale
        ),
    }
