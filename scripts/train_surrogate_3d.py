#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import struct
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATCH_RADIUS = 1
TRAINING_MAGIC_3D = 0x33534453  # SDS3
TRAINING_VERSION_3D = 1


def _read_exact(handle, nbytes: int) -> bytes:
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"Unexpected EOF while reading {nbytes} bytes")
    return data


def _load_single_solver_bin_3d(path: Path):
    with path.open('rb') as f:
        magic, version = struct.unpack('<II', _read_exact(f, 8))
        if magic != TRAINING_MAGIC_3D or version != TRAINING_VERSION_3D:
            raise ValueError(f"Unsupported 3D training export format in {path}")

        nx = struct.unpack('<Q', _read_exact(f, 8))[0]
        ny = struct.unpack('<Q', _read_exact(f, 8))[0]
        nz = struct.unpack('<Q', _read_exact(f, 8))[0]
        _coarse_steps = struct.unpack('<Q', _read_exact(f, 8))[0]
        _fine_steps = struct.unpack('<Q', _read_exact(f, 8))[0]
        _critical_fraction = struct.unpack('<d', _read_exact(f, 8))[0]

        coarse_count = struct.unpack('<Q', _read_exact(f, 8))[0]
        coarse_flat = np.frombuffer(_read_exact(f, coarse_count * 8), dtype='<f8')
        fine_count = struct.unpack('<Q', _read_exact(f, 8))[0]
        fine_flat = np.frombuffer(_read_exact(f, fine_count * 8), dtype='<f8')

    if coarse_count != nx * ny * nz or fine_count != nx * ny * nz:
        raise ValueError(f"Invalid 3D payload sizes in {path}")

    coarse = coarse_flat.reshape((int(nz), int(ny), int(nx))).astype(np.float32)
    fine = fine_flat.reshape((int(nz), int(ny), int(nx))).astype(np.float32)
    return coarse, fine


def load_paired_solver_training_data_3d(data_dir: str):
    root = Path(data_dir)
    coarse_files = sorted(root.glob('scenario_*_coarse.bin'))
    if not coarse_files:
        raise ValueError(f"No paired 3D coarse files found in {root}")

    coarse_list = []
    disc_list = []
    for cpath in coarse_files:
        stem = cpath.name.replace('_coarse.bin', '')
        fpath = root / f"{stem}_fullfine.bin"
        if not fpath.exists():
            raise ValueError(f"Missing pair file for {cpath.name}: expected {fpath.name}")

        c_coarse, _ = _load_single_solver_bin_3d(cpath)
        _, f_fine = _load_single_solver_bin_3d(fpath)
        coarse_list.append(c_coarse)
        disc_list.append((f_fine - c_coarse).astype(np.float32))

    coarse_fields = np.stack(coarse_list, axis=0).astype(np.float32)
    disc_fields = np.stack(disc_list, axis=0).astype(np.float32)
    logger.info(
        "Loaded paired 3D dataset: samples=%d, grid=%dx%dx%d",
        coarse_fields.shape[0],
        coarse_fields.shape[3],
        coarse_fields.shape[2],
        coarse_fields.shape[1],
    )
    return coarse_fields, disc_fields


def extract_patches_3d(fields: np.ndarray, radius: int = PATCH_RADIUS) -> np.ndarray:
    k = 2 * radius + 1
    padded = np.pad(fields, ((0, 0), (radius, radius), (radius, radius), (radius, radius)), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k, k), axis=(1, 2, 3))
    return windows.reshape(fields.shape[0] * fields.shape[1] * fields.shape[2] * fields.shape[3], k ** 3).astype(np.float32)


def predict_discrepancy_flat(X_s: np.ndarray, model_bundle):
    dtest = xgb.QuantileDMatrix(X_s)
    if isinstance(model_bundle, dict):
        return model_bundle['regressor'].predict(dtest)
    return model_bundle.predict(dtest)


def evaluate_model(coarse_test, disc_test, scaler, model, radius=PATCH_RADIUS, active_threshold=1e-8):
    X = extract_patches_3d(coarse_test, radius)
    y_true = disc_test.reshape(-1).astype(np.float32)
    y_pred = predict_discrepancy_flat(scaler.transform(X), model)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    active = np.abs(y_true) > active_threshold
    active_mae = float(np.mean(np.abs(y_pred[active] - y_true[active]))) if np.any(active) else 0.0
    active_rmse = float(np.sqrt(np.mean((y_pred[active] - y_true[active]) ** 2))) if np.any(active) else 0.0
    zero_mae = float(np.mean(np.abs(y_true)))
    zero_rmse = float(np.sqrt(np.mean(y_true ** 2)))

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'baseline_zero_mae': zero_mae,
        'baseline_zero_rmse': zero_rmse,
        'active_fraction': float(active.mean()),
        'active_mae': active_mae,
        'active_rmse': active_rmse,
        'active_threshold': float(active_threshold),
    }


def save_model(scaler, model, output_path: str, grid_shape, patch_radius: int, metrics: dict, training_options: dict):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'grid_x': int(grid_shape[2]),
        'grid_y': int(grid_shape[1]),
        'grid_z': int(grid_shape[0]),
        'patch_radius': int(patch_radius),
        'patch_size': int((2 * patch_radius + 1) ** 3),
        'model_type': 'xgboost_local_patch_3d_cuda',
        'metrics': metrics,
        'training_options': training_options,
    }
    with output.open('wb') as f:
        pickle.dump({'metadata': metadata, 'scaler': scaler, 'model': model}, f)
    with output.with_suffix('.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-data-dir', required=True)
    parser.add_argument('--model-output', default='models/surrogate_3d.pkl')
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--num-rounds', type=int, default=180)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--active-threshold', type=float, default=1e-8)
    args = parser.parse_args()

    logger.info("3D ML SURROGATE — local-patch XGBoost (device=cuda)")
    coarse, disc = load_paired_solver_training_data_3d(args.paired_data_dir)
    n = coarse.shape[0]
    n_test = max(1, int(round(n * args.test_split)))
    n_train = max(1, n - n_test)
    coarse_train, coarse_test = coarse[:n_train], coarse[n_train:]
    disc_train, disc_test = disc[:n_train], disc[n_train:]
    if coarse_test.shape[0] == 0:
        coarse_test, disc_test = coarse_train, disc_train

    t0 = time.perf_counter()
    X_train = extract_patches_3d(coarse_train)
    y_train = disc_train.reshape(-1).astype(np.float32)
    logger.info("3D patch extraction: %.2fs", time.perf_counter() - t0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    weights = np.ones_like(y_train, dtype=np.float32)
    weights[np.abs(y_train) > args.active_threshold] = 8.0

    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': args.max_depth,
        'eta': args.learning_rate,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'verbosity': 1,
    }
    dtrain = xgb.QuantileDMatrix(Xs, label=y_train, weight=weights)
    model = {'regressor': xgb.train(params, dtrain, num_boost_round=args.num_rounds, verbose_eval=False)}

    metrics = evaluate_model(coarse_test, disc_test, scaler, model, active_threshold=args.active_threshold)
    logger.info(
        "3D holdout: MAE=%.6f RMSE=%.6f R²=%.4f | active_MAE=%.6f active_RMSE=%.6f",
        metrics['mae'],
        metrics['rmse'],
        metrics['r2'],
        metrics['active_mae'],
        metrics['active_rmse'],
    )
    save_model(
        scaler,
        model,
        args.model_output,
        coarse.shape[1:],
        PATCH_RADIUS,
        metrics,
        {
            'num_rounds': args.num_rounds,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'active_threshold': args.active_threshold,
        },
    )


if __name__ == '__main__':
    main()