#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from train_surrogate import extract_patches_fast, load_paired_solver_training_data


def fit_weighted_ridge(X: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, float]:
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    weighted_X = X_aug * weights[:, None]
    gram = X_aug.T @ weighted_X
    reg = np.eye(gram.shape[0], dtype=X.dtype) * ridge_lambda
    reg[-1, -1] = 0.0
    rhs = X_aug.T @ (y * weights)
    beta = np.linalg.solve(gram + reg, rhs)
    return beta[:-1], float(beta[-1])


def evaluate_linear(coarse: np.ndarray,
                    disc: np.ndarray,
                    scaler: StandardScaler,
                    weights: np.ndarray,
                    bias: float,
                    radius: int,
                    active_threshold: float) -> dict[str, float]:
    X = extract_patches_fast(coarse, radius=radius)
    y_true = disc.reshape(-1).astype(np.float32)
    y_pred = scaler.transform(X) @ weights + bias
    y_pred = y_pred.astype(np.float32)

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


def save_linear_model(path: Path,
                      scaler: StandardScaler,
                      weights: np.ndarray,
                      bias: float,
                      patch_radius: int,
                      metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        'model_type: linear_patch_2d',
        f'patch_radius: {patch_radius}',
        f'bias: {bias:.17g}',
        'feature_mean: ' + ','.join(f'{value:.17g}' for value in scaler.mean_),
        'feature_scale: ' + ','.join(f'{value:.17g}' for value in scaler.scale_),
        'weights: ' + ','.join(f'{value:.17g}' for value in weights),
        f"active_threshold: {metrics['active_threshold']:.17g}",
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    metadata = {
        'model_type': 'linear_patch_2d',
        'patch_radius': patch_radius,
        'patch_size': int((2 * patch_radius + 1) ** 2),
        'metrics': metrics,
    }
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-data-dir', required=True)
    parser.add_argument('--model-output', default='models/surrogate_linear_policy.txt')
    parser.add_argument('--patch-radius', type=int, default=2)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--active-threshold', type=float, default=1e-8)
    parser.add_argument('--active-weight', type=float, default=8.0)
    parser.add_argument('--ridge-lambda', type=float, default=1e-3)
    parser.add_argument('--max-rows', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=20260404)
    args = parser.parse_args()

    coarse, disc = load_paired_solver_training_data(args.paired_data_dir)
    total = coarse.shape[0]
    n_test = max(1, int(total * args.test_split))
    if n_test >= total:
        n_test = total - 1
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(total)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train = extract_patches_fast(coarse[train_idx], radius=args.patch_radius)
    y_train = disc[train_idx].reshape(-1).astype(np.float64)
    if args.max_rows > 0 and X_train.shape[0] > args.max_rows:
        row_idx = rng.choice(X_train.shape[0], size=args.max_rows, replace=False)
        X_train = X_train[row_idx]
        y_train = y_train[row_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    sample_weights = np.ones_like(y_train, dtype=np.float64)
    sample_weights[np.abs(y_train) > args.active_threshold] = float(args.active_weight)
    weights, bias = fit_weighted_ridge(X_scaled.astype(np.float64), y_train, sample_weights, args.ridge_lambda)

    metrics = evaluate_linear(
        coarse[test_idx],
        disc[test_idx],
        scaler,
        weights,
        bias,
        radius=args.patch_radius,
        active_threshold=args.active_threshold,
    )
    save_linear_model(Path(args.model_output), scaler, weights, bias, args.patch_radius, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()