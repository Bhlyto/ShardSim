#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import struct
from pathlib import Path

import numpy as np

from train_surrogate import extract_patches_fast, predict_discrepancy_flat


COARSE_MAGIC = 0x31434453  # SDC1
COARSE_VERSION = 1
MASK_MAGIC = 0x314D4453    # SDM1
MASK_VERSION = 1


def _read_exact(handle, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise ValueError(f"Expected {n} bytes, got {len(data)}")
    return data


def load_coarse_field(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, version = struct.unpack("<II", _read_exact(handle, 8))
        if magic != COARSE_MAGIC or version != COARSE_VERSION:
            raise ValueError(f"Unsupported coarse field file: {path}")
        nx = struct.unpack("<Q", _read_exact(handle, 8))[0]
        ny = struct.unpack("<Q", _read_exact(handle, 8))[0]
        count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        data = np.frombuffer(_read_exact(handle, count * 8), dtype="<f8")
    if count != nx * ny:
        raise ValueError(f"Invalid coarse field payload size in {path}")
    return data.reshape((int(ny), int(nx))).astype(np.float32)


def write_mask(path: Path, mask: np.ndarray) -> None:
    mask = np.asarray(mask, dtype=np.uint8)
    ny, nx = mask.shape
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", MASK_MAGIC, MASK_VERSION))
        handle.write(struct.pack("<Q", nx))
        handle.write(struct.pack("<Q", ny))
        handle.write(struct.pack("<Q", mask.size))
        handle.write(mask.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-fraction", type=float, default=0.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.model, "rb") as handle:
        bundle = pickle.load(handle)

    coarse = load_coarse_field(Path(args.input))
    radius = int(bundle["metadata"].get("patch_radius", 2))
    X = extract_patches_fast(coarse[None, :, :], radius=radius)
    Xs = bundle["scaler"].transform(X)
    pred = predict_discrepancy_flat(Xs, bundle["model"]).reshape(coarse.shape)
    scores = np.abs(pred)

    score_threshold = float(args.score_threshold)
    if score_threshold > 0.0:
        mask = (scores > score_threshold).astype(np.uint8)
    else:
        mask = np.zeros_like(scores, dtype=np.uint8)
    min_fraction = float(np.clip(args.min_fraction, 0.0, 1.0))
    min_required = int(np.ceil(min_fraction * mask.size))
    current = int(mask.sum())
    if current < min_required:
        flat_scores = scores.reshape(-1)
        top_idx = np.argpartition(flat_scores, -min_required)[-min_required:]
        flat_mask = mask.reshape(-1)
        flat_mask[top_idx] = 1
        mask = flat_mask.reshape(mask.shape)

    write_mask(Path(args.output), mask)


if __name__ == "__main__":
    main()