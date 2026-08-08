from __future__ import annotations

import numpy as np


def bilinear_resample(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(field, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("field must be two-dimensional.")
    if len(target_shape) != 2 or min(target_shape) < 2:
        raise ValueError("target_shape must describe a grid of at least 2x2.")
    if source.shape == target_shape:
        return source.copy()

    source_y = np.linspace(0.0, 1.0, source.shape[0])
    source_x = np.linspace(0.0, 1.0, source.shape[1])
    target_y = np.linspace(0.0, 1.0, target_shape[0])
    target_x = np.linspace(0.0, 1.0, target_shape[1])

    horizontal = np.empty((source.shape[0], target_shape[1]), dtype=np.float64)
    for row_index, row in enumerate(source):
        horizontal[row_index] = np.interp(target_x, source_x, row)

    result = np.empty(target_shape, dtype=np.float64)
    for column_index in range(target_shape[1]):
        result[:, column_index] = np.interp(target_y, source_y, horizontal[:, column_index])
    return result
