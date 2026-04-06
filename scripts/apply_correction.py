#!/usr/bin/env python3
"""
apply_correction.py — Phase 4 ML correction inference script.

Reads a coarse field binary, applies the trained XGBoost correction model
(coarse → discrepancy), adds the correction to the coarse field, and writes
the corrected field back in the same binary format.

Binary format (both input and output):
  magic    : uint32  (0x31434453 = "SDC1")
  version  : uint32  (1)
  nx       : uint64
  ny       : uint64
  count    : uint64  (= nx * ny)
  values   : count * float64 (row-major: values[i*ny + j])

Usage:
  python scripts/apply_correction.py \
      --model models/surrogate_hardened_fullscale.pkl \
      --input /tmp/coarse.bin \
      --output /tmp/corrected.bin
"""

from __future__ import annotations

import argparse
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

from train_surrogate import predict_discrepancy_flat

MAGIC = 0x31434453
VERSION = 1

PATCH_RADIUS = 2  # must match training configuration


def read_field(path: Path) -> tuple[np.ndarray, int, int]:
    """Return (values_flat, nx, ny)."""
    with open(path, "rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"Invalid field binary: magic=0x{magic:08X} version={version}")
        nx, ny, count = struct.unpack("<QQQ", f.read(24))
        if count != nx * ny:
            raise ValueError(f"Field count mismatch: {count} != {nx}*{ny}")
        values = np.frombuffer(f.read(count * 8), dtype=np.float64).copy()
    return values, int(nx), int(ny)


def write_field(path: Path, values: np.ndarray, nx: int, ny: int) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<QQQ", nx, ny, nx * ny))
        f.write(values.astype(np.float64).tobytes())


def extract_patches(field: np.ndarray, nx: int, ny: int, radius: int) -> np.ndarray:
    """
    Extract (2R+1)^2 local patches for every cell; boundary cells are
    clamped (same as the C++ linear policy apply path).
    Returns shape (nx*ny, (2R+1)^2).
    """
    side = 2 * radius + 1
    n_features = side * side
    # Reshape to (nx, ny) for easier indexing
    grid = field.reshape(nx, ny)
    patches = np.empty((nx * ny, n_features), dtype=np.float32)
    feat = 0
    for dj in range(-radius, radius + 1):
        for di in range(-radius, radius + 1):
            ii = np.clip(np.arange(nx) + di, 0, nx - 1)
            jj = np.clip(np.arange(ny) + dj, 0, ny - 1)
            # grid[ii, :][:, jj] would give the patch values
            # but we need all (i,j) pairs:
            patch_vals = grid[np.ix_(ii, jj)]  # (nx, ny)
            patches[:, feat] = patch_vals.reshape(-1)
            feat += 1
    return patches


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply XGBoost correction model to coarse field")
    parser.add_argument("--model", required=True, help="Path to .pkl surrogate model")
    parser.add_argument("--input", required=True, help="Path to coarse field binary")
    parser.add_argument("--output", required=True, help="Path to write corrected field binary")
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    if not input_path.exists():
        sys.exit(f"Input field not found: {input_path}")

    # Load model
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)

    # The surrogate artefact is a dict with keys: model, scaler, patch_radius, ...
    if isinstance(artefact, dict):
        model = artefact.get("model") or artefact.get("xgb_model") or artefact.get("bst")
        scaler = artefact.get("scaler")
        radius = int(artefact.get("patch_radius", PATCH_RADIUS))
    else:
        # Fallback: artefact is the raw model
        model = artefact
        scaler = None
        radius = PATCH_RADIUS

    if model is None:
        sys.exit(f"Could not extract model from artefact keys: {list(artefact.keys())}")

    # Load field
    values, nx, ny = read_field(input_path)

    # Extract patches
    patches = extract_patches(values, nx, ny, radius)

    # Optionally scale
    if scaler is not None:
        patches = scaler.transform(patches).astype(np.float32)

    # Predict discrepancy. The training artefact may store either a raw model
    # object or a model bundle dict (single/two-stage).
    if isinstance(model, dict):
        correction = predict_discrepancy_flat(patches, model).astype(np.float64)
    else:
        try:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(patches)
            correction = model.predict(dmatrix).astype(np.float64)
        except ImportError:
            # Fallback: sklearn-compatible predict
            correction = model.predict(patches).astype(np.float64)

    # Apply correction
    corrected = values + correction

    # Write output
    write_field(output_path, corrected, nx, ny)


if __name__ == "__main__":
    main()
