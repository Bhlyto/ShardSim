#!/usr/bin/env python3
"""
ML Surrogate Training: Local-Patch Coarse-to-Fine Discrepancy Model (GPU-Accelerated)

Trains an XGBoost model to predict cell-wise discrepancies using LOCAL PATCH features
rather than the full solution grid. This is the standard physics-surrogate design:
  - Input:  5x5 local neighborhood of coarse temperature around each cell (25 features)
  - Output: discrepancy at that cell = fine[i,j] - coarse[i,j]
  - Scale:  1 model trained on all cells from all samples (13M instances, 25 features)

This collapses the input from 65K -> 25 features per instance, making GPU training
fast and memory-efficient on an RTX 4070.

GPU Requirements:
    XGBoost from NVIDIA index (USE_CUDA=True):
    pip install xgboost --index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple/

Usage:
    source <venv>/bin/activate
    python3 scripts/train_surrogate.py --grid-size 256 --n-samples 200 \
                                        --model-output models/surrogate_256x256_cuda.pkl
"""

import argparse
import json
import numpy as np
import pickle
from pathlib import Path
import struct
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATCH_RADIUS = 2   # 5x5 patch = 25 features
TRAINING_MAGIC = 0x31534453  # "SDS1"
TRAINING_VERSION = 1


def generate_synthetic_training_data(grid_size=256, n_samples=200, seed=42):
    """
    Generate synthetic (coarse, fine) solution pairs for a 2D heat problem.
    Coarse: smooth low-frequency superposition.
    Fine:   coarse + gradient-correlated local noise (the hard-to-capture detail).
    """
    rng = np.random.default_rng(seed)
    logger.info(f"Generating {n_samples} synthetic training pairs ({grid_size}x{grid_size})")

    coarse_fields, disc_fields = [], []
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x, y)

    for _ in range(n_samples):
        coarse = np.zeros((grid_size, grid_size))
        for _ in range(4):
            kx, ky = rng.integers(1, 6), rng.integers(1, 6)
            coarse += rng.uniform(0.5, 1.5) * np.sin(kx * np.pi * xx) * np.sin(ky * np.pi * yy)
        coarse /= 4.0

        dTdx = np.gradient(coarse, axis=1)
        dTdy = np.gradient(coarse, axis=0)
        grad_mag = np.sqrt(dTdx**2 + dTdy**2)
        scale = 0.3 * grad_mag / (grad_mag.max() + 1e-8)
        discrepancy = scale * rng.standard_normal((grid_size, grid_size))

        coarse_fields.append(coarse)
        disc_fields.append(discrepancy)

    return np.array(coarse_fields, dtype=np.float32), np.array(disc_fields, dtype=np.float32)


def _read_exact(handle, nbytes):
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"Unexpected EOF while reading {nbytes} bytes")
    return data


def load_solver_training_data(data_dir, expected_grid_size=None):
    """
    Load solver-exported binary coarse/fine samples and return:
      coarse_fields: (N, H, W)
      disc_fields:   (N, H, W) where disc = fine - coarse
    """
    root = Path(data_dir)
    files = sorted(root.glob('*.bin'))
    if not files:
        raise ValueError(f"No .bin training samples found in {root}")

    coarse_list = []
    disc_list = []
    dims_seen = None

    logger.info(f"Loading {len(files)} solver-exported training samples from {root}")
    for path in files:
        with open(path, 'rb') as f:
            magic, version = struct.unpack('<II', _read_exact(f, 8))
            if magic != TRAINING_MAGIC:
                raise ValueError(f"{path}: invalid magic (expected {TRAINING_MAGIC:#x})")
            if version != TRAINING_VERSION:
                raise ValueError(f"{path}: unsupported version {version}")

            nx, ny, coarse_steps, fine_steps = struct.unpack('<QQQQ', _read_exact(f, 32))
            _critical_fraction = struct.unpack('<d', _read_exact(f, 8))[0]

            coarse_count = struct.unpack('<Q', _read_exact(f, 8))[0]
            coarse_raw = _read_exact(f, coarse_count * 8)
            coarse_flat = np.frombuffer(coarse_raw, dtype='<f8')

            fine_count = struct.unpack('<Q', _read_exact(f, 8))[0]
            fine_raw = _read_exact(f, fine_count * 8)
            fine_flat = np.frombuffer(fine_raw, dtype='<f8')

            if coarse_count != nx * ny or fine_count != nx * ny:
                raise ValueError(
                    f"{path}: count mismatch (nx={nx}, ny={ny}, coarse_count={coarse_count}, fine_count={fine_count})"
                )

            if coarse_steps == 0 or fine_steps == 0:
                logger.warning(f"{path}: zero solver steps found in header")

            if dims_seen is None:
                dims_seen = (int(nx), int(ny))
            elif dims_seen != (int(nx), int(ny)):
                raise ValueError(f"Mixed grid sizes in dataset: saw {dims_seen} and {(int(nx), int(ny))}")

            coarse = coarse_flat.reshape((int(ny), int(nx))).astype(np.float32)
            fine = fine_flat.reshape((int(ny), int(nx))).astype(np.float32)

            coarse_list.append(coarse)
            disc_list.append(fine - coarse)

    coarse_fields = np.stack(coarse_list, axis=0)
    disc_fields = np.stack(disc_list, axis=0)

    _, H, W = coarse_fields.shape
    logger.info(f"Loaded real dataset: samples={coarse_fields.shape[0]}, grid={W}x{H}")

    if expected_grid_size is not None and (H != expected_grid_size or W != expected_grid_size):
        raise ValueError(
            f"Loaded grid is {W}x{H} but --grid-size={expected_grid_size}. "
            "Use matching exports or adjust --grid-size."
        )

    return coarse_fields, disc_fields


def _load_single_solver_bin(path):
    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', _read_exact(f, 8))
        if magic != TRAINING_MAGIC:
            raise ValueError(f"{path}: invalid magic (expected {TRAINING_MAGIC:#x})")
        if version != TRAINING_VERSION:
            raise ValueError(f"{path}: unsupported version {version}")

        nx, ny, coarse_steps, fine_steps = struct.unpack('<QQQQ', _read_exact(f, 32))
        _critical_fraction = struct.unpack('<d', _read_exact(f, 8))[0]

        coarse_count = struct.unpack('<Q', _read_exact(f, 8))[0]
        coarse_raw = _read_exact(f, coarse_count * 8)
        coarse_flat = np.frombuffer(coarse_raw, dtype='<f8')

        fine_count = struct.unpack('<Q', _read_exact(f, 8))[0]
        fine_raw = _read_exact(f, fine_count * 8)
        fine_flat = np.frombuffer(fine_raw, dtype='<f8')

    if coarse_count != nx * ny or fine_count != nx * ny:
        raise ValueError(
            f"{path}: count mismatch (nx={nx}, ny={ny}, coarse_count={coarse_count}, fine_count={fine_count})"
        )

    coarse = coarse_flat.reshape((int(ny), int(nx))).astype(np.float32)
    fine = fine_flat.reshape((int(ny), int(nx))).astype(np.float32)
    return coarse, fine


def load_paired_solver_training_data(data_dir, expected_grid_size=None):
    """
    Load paired samples named as:
      scenario_XXXX_coarse.bin
      scenario_XXXX_fullfine.bin

    Returns coarse_fields and discrepancy fields where discrepancy = fullfine - coarse.
    """
    root = Path(data_dir)
    coarse_files = sorted(root.glob('scenario_*_coarse.bin'))
    if not coarse_files:
        raise ValueError(f"No paired coarse files found in {root}")

    coarse_list = []
    disc_list = []
    for cpath in coarse_files:
        stem = cpath.name.replace('_coarse.bin', '')
        fpath = root / f"{stem}_fullfine.bin"
        if not fpath.exists():
            raise ValueError(f"Missing pair file for {cpath.name}: expected {fpath.name}")

        c_coarse, _ = _load_single_solver_bin(cpath)
        _, f_fine = _load_single_solver_bin(fpath)
        if c_coarse.shape != f_fine.shape:
            raise ValueError(f"Shape mismatch for pair {stem}: {c_coarse.shape} vs {f_fine.shape}")

        coarse_list.append(c_coarse)
        disc_list.append((f_fine - c_coarse).astype(np.float32))

    coarse_fields = np.stack(coarse_list, axis=0).astype(np.float32)
    disc_fields = np.stack(disc_list, axis=0).astype(np.float32)
    _, H, W = coarse_fields.shape
    logger.info(f"Loaded paired real dataset: samples={coarse_fields.shape[0]}, grid={W}x{H}")

    if expected_grid_size is not None and (H != expected_grid_size or W != expected_grid_size):
        raise ValueError(
            f"Loaded grid is {W}x{H} but --grid-size={expected_grid_size}. "
            "Use matching exports or adjust --grid-size."
        )

    return coarse_fields, disc_fields


def extract_patches_fast(fields, radius=PATCH_RADIUS):
    """
    Vectorised patch extraction via stride tricks.
    fields: (N, H, W)  →  returns (N*H*W, (2R+1)^2)
    """
    n, H, W = fields.shape
    k = 2 * radius + 1
    padded = np.pad(fields, ((0,0),(radius,radius),(radius,radius)), mode='edge')
    shape   = (n, H, W, k, k)
    strides = (padded.strides[0], padded.strides[1], padded.strides[2],
               padded.strides[1], padded.strides[2])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return windows.reshape(n * H * W, k * k).astype(np.float32)


def train_surrogate_model(
    coarse_fields,
    disc_fields,
    radius=PATCH_RADIUS,
    num_rounds=200,
    max_depth=6,
    learning_rate=0.1,
    active_threshold=1e-8,
    active_weight=8.0,
    two_stage=False,
    clf_rounds=120,
):
    """
    Train XGBoost on local-patch -> discrepancy.
    One model, 25 features in, scalar discrepancy out.
    """
    n, H, W = coarse_fields.shape
    patch_size = (2 * radius + 1) ** 2
    n_instances = n * H * W

    logger.info(f"Extracting patches: {n_instances:,} instances x {patch_size} features")
    t0 = time.perf_counter()
    X = extract_patches_fast(coarse_fields, radius)
    y = disc_fields.reshape(-1).astype(np.float32)
    logger.info(f"Patch extraction: {time.perf_counter()-t0:.2f}s")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Building QuantileDMatrix on CUDA device...")
    t1 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(X_scaled, label=y)
    logger.info(f"QuantileDMatrix ready: {time.perf_counter()-t1:.2f}s")

    common = {
        'tree_method':      'hist',
        'device':           'cuda',
        'max_depth':        max_depth,
        'learning_rate':    learning_rate,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'verbosity':        1,
    }

    if two_stage:
        y_active = (np.abs(y) > active_threshold).astype(np.float32)
        dcls = xgb.QuantileDMatrix(X_scaled, label=y_active)
        cls_params = {
            **common,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
        }
        logger.info(f"Training classifier (active threshold={active_threshold:.2e}, rounds={clf_rounds})...")
        classifier = xgb.train(cls_params, dcls, num_boost_round=clf_rounds, verbose_eval=False)

        active_idx = np.where(y_active > 0.5)[0]
        if active_idx.size < 256:
            logger.warning("Too few active cells for two-stage regressor; falling back to weighted single-stage")
            two_stage = False
        else:
            dreg = xgb.QuantileDMatrix(X_scaled[active_idx], label=y[active_idx])
            reg_params = {
                **common,
                'objective': 'reg:squarederror',
            }
            logger.info(f"Training active-cell regressor (active cells={active_idx.size:,}, rounds={num_rounds})...")
            regressor = xgb.train(reg_params, dreg, num_boost_round=num_rounds, verbose_eval=False)
            return scaler, {
                'mode': 'two-stage',
                'active_threshold': float(active_threshold),
                'classifier': classifier,
                'regressor': regressor,
            }

    params = {
        'objective':        'reg:squarederror',
        **common,
    }

    weights = np.ones_like(y, dtype=np.float32)
    weights[np.abs(y) > active_threshold] = float(active_weight)
    dtrain = xgb.QuantileDMatrix(X_scaled, label=y, weight=weights)

    logger.info(f"Training weighted XGBoost (rounds={num_rounds}, active_weight={active_weight})...")
    t2 = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=num_rounds, verbose_eval=False)
    logger.info(f"Training complete: {time.perf_counter()-t2:.2f}s")

    return scaler, {
        'mode': 'single',
        'active_threshold': float(active_threshold),
        'regressor': model,
    }


def predict_discrepancy_flat(X_s, model_bundle):
    dtest = xgb.QuantileDMatrix(X_s)
    if isinstance(model_bundle, dict):
        mode = model_bundle.get('mode', 'single')
        if mode == 'two-stage':
            p_active = model_bundle['classifier'].predict(dtest)
            y_reg = model_bundle['regressor'].predict(dtest)
            return p_active * y_reg
        return model_bundle['regressor'].predict(dtest)
    return model_bundle.predict(dtest)


def evaluate_model(coarse_test, disc_test, scaler, model, radius=PATCH_RADIUS, active_threshold=1e-8):
    X      = extract_patches_fast(coarse_test, radius)
    y_true = disc_test.reshape(-1).astype(np.float32)
    X_s    = scaler.transform(X)
    y_pred = predict_discrepancy_flat(X_s, model)

    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    y_zero = np.zeros_like(y_true)
    mae0 = float(np.mean(np.abs(y_zero - y_true)))
    rmse0 = float(np.sqrt(np.mean((y_zero - y_true)**2)))

    active = np.abs(y_true) > active_threshold
    if np.any(active):
        ya = y_true[active]
        pa = y_pred[active]
        active_mae = float(np.mean(np.abs(pa - ya)))
        active_rmse = float(np.sqrt(np.mean((pa - ya)**2)))
    else:
        active_mae = 0.0
        active_rmse = 0.0

    n = y_true.size
    k1 = max(1, n // 100)
    k5 = max(1, n // 20)
    t1 = set(np.argpartition(np.abs(y_true), -k1)[-k1:])
    p1 = set(np.argpartition(np.abs(y_pred), -k1)[-k1:])
    t5 = set(np.argpartition(np.abs(y_true), -k5)[-k5:])
    p5 = set(np.argpartition(np.abs(y_pred), -k5)[-k5:])
    top1_recall = float(len(t1 & p1) / k1)
    top5_recall = float(len(t5 & p5) / k5)

    logger.info(
        f"Test Metrics: MAE={mae:.6f} RMSE={rmse:.6f} R²={r2:.4f} | "
        f"active_MAE={active_mae:.6f} active_RMSE={active_rmse:.6f} | "
        f"top1={top1_recall:.3f} top5={top5_recall:.3f}")

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'baseline_zero_mae': mae0,
        'baseline_zero_rmse': rmse0,
        'active_threshold': float(active_threshold),
        'active_fraction': float(active.mean()),
        'active_mae': active_mae,
        'active_rmse': active_rmse,
        'top1_recall': top1_recall,
        'top5_recall': top5_recall,
    }


def save_model(scaler, model, output_path, grid_width, grid_height, patch_radius, metrics=None, training_options=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        'grid_size': max(grid_width, grid_height),
        'grid_width': grid_width,
        'grid_height': grid_height,
        'patch_radius': patch_radius,
        'patch_size': (2*patch_radius+1)**2,
        'model_type': 'xgboost_local_patch_cuda',
        'metrics': metrics or {},
        'training_options': training_options or {},
    }
    with open(output_path, 'wb') as f:
        pickle.dump({'metadata': meta, 'scaler': scaler, 'model': model}, f)
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved: {output_path}  ({output_path.stat().st_size/1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-size',    type=int,   default=256)
    parser.add_argument('--n-samples',    type=int,   default=200)
    parser.add_argument('--patch-radius', type=int,   default=PATCH_RADIUS)
    parser.add_argument('--test-split',   type=float, default=0.2)
    parser.add_argument('--model-output', type=str,   default='models/surrogate.pkl')
    parser.add_argument('--data-dir',     type=str,   default='',
                        help='Directory with solver-exported *.bin samples. If provided, synthetic data is skipped.')
    parser.add_argument('--paired-data-dir', type=str, default='',
                        help='Directory with scenario_XXXX_coarse.bin and scenario_XXXX_fullfine.bin pairs.')
    parser.add_argument('--allow-non-square-grid', action='store_true',
                        help='Allow rectangular real data (otherwise grid must match --grid-size x --grid-size).')
    parser.add_argument('--num-rounds', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--active-threshold', type=float, default=1e-8)
    parser.add_argument('--active-weight', type=float, default=8.0)
    parser.add_argument('--two-stage', action='store_true')
    parser.add_argument('--clf-rounds', type=int, default=120)
    args = parser.parse_args()

    build  = xgb.build_info()
    gpu_ok = build.get('USE_CUDA', False)
    logger.info("=" * 70)
    logger.info("ML SURROGATE — local-patch XGBoost (device=cuda)")
    logger.info("=" * 70)
    logger.info(f"XGBoost {xgb.__version__}  USE_CUDA={gpu_ok}  CUDA={build.get('CUDA_VERSION')}")
    logger.info(f"grid={args.grid_size}x{args.grid_size}  samples={args.n_samples}  patch_radius={args.patch_radius}")
    if not gpu_ok:
        logger.warning("USE_CUDA=False — install from NVIDIA index for GPU acceleration")

    if args.data_dir and args.paired_data_dir:
        raise ValueError("Use either --data-dir or --paired-data-dir, not both")

    if args.paired_data_dir:
        expected = None if args.allow_non_square_grid else args.grid_size
        coarse, disc = load_paired_solver_training_data(args.paired_data_dir, expected_grid_size=expected)
        if coarse.shape[0] < 2:
            raise ValueError("Need at least 2 paired real samples for train/test split")
    elif args.data_dir:
        expected = None if args.allow_non_square_grid else args.grid_size
        coarse, disc = load_solver_training_data(args.data_dir, expected_grid_size=expected)
        if coarse.shape[0] < 2:
            raise ValueError("Need at least 2 real samples for train/test split")
    else:
        coarse, disc = generate_synthetic_training_data(args.grid_size, args.n_samples)

    total_samples = coarse.shape[0]
    n_test = max(1, int(total_samples * args.test_split))
    if n_test >= total_samples:
        n_test = total_samples - 1
    idx = np.random.default_rng(0).permutation(total_samples)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    logger.info(f"Train/test: {len(train_idx)} / {len(test_idx)} samples")

    scaler, model = train_surrogate_model(
        coarse[train_idx],
        disc[train_idx],
        radius=args.patch_radius,
        num_rounds=args.num_rounds,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        active_threshold=args.active_threshold,
        active_weight=args.active_weight,
        two_stage=args.two_stage,
        clf_rounds=args.clf_rounds,
    )

    metrics = evaluate_model(
        coarse[test_idx],
        disc[test_idx],
        scaler,
        model,
        args.patch_radius,
        active_threshold=args.active_threshold,
    )

    save_model(
        scaler,
        model,
        args.model_output,
        grid_width=int(coarse.shape[2]),
        grid_height=int(coarse.shape[1]),
        patch_radius=args.patch_radius,
        metrics=metrics,
        training_options={
            'num_rounds': args.num_rounds,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'active_threshold': args.active_threshold,
            'active_weight': args.active_weight,
            'two_stage': args.two_stage,
            'clf_rounds': args.clf_rounds,
        },
    )

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
