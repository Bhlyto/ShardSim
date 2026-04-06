#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import sys
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import xgboost as xgb

from train_surrogate_3d import extract_patches_3d


COARSE_MAGIC = 0x33434453  # SDC3
COARSE_VERSION = 1
MASK_MAGIC = 0x334D4453    # SDM3
MASK_VERSION = 1


def _read_exact(handle: BinaryIO, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise ValueError(f"Expected {n} bytes, got {len(data)}")
    return data


def load_coarse_field(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, version = struct.unpack("<II", _read_exact(handle, 8))
        if magic != COARSE_MAGIC or version != COARSE_VERSION:
            raise ValueError(f"Unsupported 3D coarse field file: {path}")
        nx = struct.unpack("<Q", _read_exact(handle, 8))[0]
        ny = struct.unpack("<Q", _read_exact(handle, 8))[0]
        nz = struct.unpack("<Q", _read_exact(handle, 8))[0]
        count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        data = np.frombuffer(_read_exact(handle, count * 8), dtype="<f8")
    if count != nx * ny * nz:
        raise ValueError(f"Invalid 3D coarse payload size in {path}")
    return data.reshape((int(nz), int(ny), int(nx))).astype(np.float32)


def write_mask(path: Path, mask: np.ndarray) -> None:
    mask = np.asarray(mask, dtype=np.uint8)
    nz, ny, nx = mask.shape
    with path.open("wb") as handle:
        handle.write(struct.pack("<II", MASK_MAGIC, MASK_VERSION))
        handle.write(struct.pack("<Q", nx))
        handle.write(struct.pack("<Q", ny))
        handle.write(struct.pack("<Q", nz))
        handle.write(struct.pack("<Q", mask.size))
        handle.write(mask.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--min-fraction", type=float, default=0.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--worker-stdio", action="store_true")
    args = parser.parse_args()

    model_cache: dict[str, Any] = {}

    def compute_mask(input_path: Path, min_fraction: float, score_threshold: float) -> np.ndarray:
        model_key = str(Path(args.model).resolve())
        bundle: Any = model_cache.get(model_key)
        if bundle is None:
            with open(args.model, "rb") as handle:
                bundle = pickle.load(handle)
            model_cache[model_key] = bundle

        coarse = load_coarse_field(input_path)
        radius = int(bundle["metadata"].get("patch_radius", 1))
        X = extract_patches_3d(coarse[None, :, :, :], radius=radius)
        Xs = bundle["scaler"].transform(X)

        dtest = xgb.QuantileDMatrix(Xs)
        reg = bundle["model"]["regressor"] if isinstance(bundle["model"], dict) else bundle["model"]
        pred = reg.predict(dtest).reshape(coarse.shape)
        scores = np.abs(pred)

        score_threshold = float(score_threshold)
        if score_threshold > 0.0:
            mask = (scores > score_threshold).astype(np.uint8)
        else:
            mask = np.zeros_like(scores, dtype=np.uint8)

        min_fraction = float(np.clip(min_fraction, 0.0, 1.0))
        min_required = int(np.ceil(min_fraction * mask.size))
        current = int(mask.sum())
        if min_required > 0 and current < min_required:
            flat_scores = scores.reshape(-1)
            top_idx = np.argpartition(flat_scores, -min_required)[-min_required:]
            flat_mask = mask.reshape(-1)
            flat_mask[top_idx] = 1
            mask = flat_mask.reshape(mask.shape)
        return mask

    if args.worker_stdio:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if req.get("command") == "shutdown":
                    print(json.dumps({"ok": True}), flush=True)
                    return

                input_path = Path(req["input"])
                output_path = Path(req["output"])
                min_fraction = float(req.get("min_fraction", 0.0))
                score_threshold = float(req.get("score_threshold", 0.0))

                mask = compute_mask(input_path, min_fraction, score_threshold)
                write_mask(output_path, mask)
                print(json.dumps({"ok": True}), flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required unless --worker-stdio is used")

    mask = compute_mask(Path(args.input), args.min_fraction, args.score_threshold)
    write_mask(Path(args.output), mask)


if __name__ == "__main__":
    main()
