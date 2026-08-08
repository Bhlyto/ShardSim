from __future__ import annotations

import numpy as np

from shardsim.contracts import ProblemSpec


def field_summary_features(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    y = np.linspace(0.0, 1.0, values.shape[0])
    x = np.linspace(0.0, 1.0, values.shape[1])
    x_grid, y_grid = np.meshgrid(x, y)
    weights = np.abs(values)
    weight_sum = float(np.sum(weights))
    if weight_sum > 1e-12:
        center_x = float(np.sum(weights * x_grid) / weight_sum)
        center_y = float(np.sum(weights * y_grid) / weight_sum)
        spread_x = float(np.sqrt(np.sum(weights * np.square(x_grid - center_x)) / weight_sum))
        spread_y = float(np.sqrt(np.sum(weights * np.square(y_grid - center_y)) / weight_sum))
    else:
        center_x = center_y = 0.5
        spread_x = spread_y = 0.0
    return np.array(
        [
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.sqrt(np.mean(np.square(values))),
            center_x,
            center_y,
            spread_x,
            spread_y,
        ],
        dtype=np.float64,
    )


def model_features(
    field: np.ndarray,
    problem: ProblemSpec,
    parameter_names: tuple[str, ...],
) -> np.ndarray:
    physical_features = np.array(
        [
            *(problem.parameter(name) for name in parameter_names),
            problem.t_end,
            *problem.extent,
        ],
        dtype=np.float64,
    )
    return np.concatenate([physical_features, field_summary_features(field)])
