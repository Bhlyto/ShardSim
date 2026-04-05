#!/usr/bin/env python3
"""
Surrogate Model Visualization

Generates a comprehensive dashboard showing:
  1. Scatter: predicted vs actual discrepancy (all test cells)
  2. Error distribution histogram (with percentiles)
  3. Spatial maps for one test sample: coarse / true discrepancy / predicted / abs error
  4. Cell criticality repartition: how the model ranks cells (histogram of |predicted disc|)
  5. Cumulative error curve: sorted abs error across cells

Usage:
    source /home/bhlyto/.venv/bin/activate
    python3 scripts/visualize_surrogate.py --model models/surrogate_256x256_cuda.pkl \
                                            --grid-size 256 --n-samples 40 \
                                            --output reports/surrogate_eval.png
"""

import argparse
import pickle
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import logging

from train_surrogate import (
    load_solver_training_data,
    load_paired_solver_training_data,
    predict_discrepancy_flat,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATCH_RADIUS = 2


# ── data helpers (same as train_surrogate.py) ──────────────────────────────────

def generate_test_data(grid_size, n_samples, seed=99):
    """Generate fresh test data (different seed from training)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x, y)
    coarse_list, disc_list = [], []
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
        coarse_list.append(coarse)
        disc_list.append(discrepancy)
    return np.array(coarse_list, dtype=np.float32), np.array(disc_list, dtype=np.float32)


def extract_patches_fast(fields, radius=PATCH_RADIUS):
    n, H, W = fields.shape
    k = 2 * radius + 1
    padded = np.pad(fields, ((0,0),(radius,radius),(radius,radius)), mode='edge')
    shape   = (n, H, W, k, k)
    strides = (padded.strides[0], padded.strides[1], padded.strides[2],
               padded.strides[1], padded.strides[2])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return windows.reshape(n * H * W, k * k).astype(np.float32)


def predict_discrepancy(coarse_fields, scaler, model, radius=PATCH_RADIUS):
    X      = extract_patches_fast(coarse_fields, radius)
    X_s    = scaler.transform(X)
    y_pred = predict_discrepancy_flat(X_s, model)
    n, H, W = coarse_fields.shape
    return y_pred.reshape(n, H, W)


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_scatter(ax, y_true_flat, y_pred_flat, subsample=50000):
    """Predicted vs actual scatter, subsampled for readability."""
    if len(y_true_flat) > subsample:
        idx = np.random.default_rng(0).choice(len(y_true_flat), subsample, replace=False)
        yt, yp = y_true_flat[idx], y_pred_flat[idx]
    else:
        yt, yp = y_true_flat, y_pred_flat

    vmin = min(yt.min(), yp.min())
    vmax = max(yt.max(), yp.max())

    ax.scatter(yt, yp, s=1, alpha=0.15, c='steelblue', rasterized=True)
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1.5, label='ideal')

    mae  = np.mean(np.abs(y_pred_flat - y_true_flat))
    rmse = np.sqrt(np.mean((y_pred_flat - y_true_flat)**2))
    ss_res = np.sum((y_true_flat - y_pred_flat)**2)
    ss_tot = np.sum((y_true_flat - y_true_flat.mean())**2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    ax.set_xlabel('True discrepancy')
    ax.set_ylabel('Predicted discrepancy')
    ax.set_title(f'Predicted vs Actual  (MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.3f})',
                 fontsize=9)
    ax.legend(fontsize=8)


def plot_error_histogram(ax, y_true_flat, y_pred_flat):
    """Absolute error distribution with percentile lines."""
    abs_err = np.abs(y_pred_flat - y_true_flat)
    p50, p90, p99 = np.percentile(abs_err, [50, 90, 99])

    ax.hist(abs_err, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    for p, label, color in [(p50, f'P50={p50:.4f}', 'limegreen'),
                             (p90, f'P90={p90:.4f}', 'orange'),
                             (p99, f'P99={p99:.4f}', 'red')]:
        ax.axvline(p, color=color, linestyle='--', lw=1.5, label=label)

    ax.set_xlabel('Absolute error')
    ax.set_ylabel('Cell count')
    ax.set_title('Error Distribution (|pred − true|)', fontsize=9)
    ax.legend(fontsize=8)


def plot_spatial(axes, coarse, disc_true, disc_pred):
    """Four spatial maps for one sample."""
    abs_err = np.abs(disc_pred - disc_true)
    disc_max = max(np.abs(disc_true).max(), np.abs(disc_pred).max(), 1e-6)
    disc_norm = TwoSlopeNorm(vmin=-disc_max, vcenter=0, vmax=disc_max)

    im0 = axes[0].imshow(coarse,     cmap='hot',    origin='lower')
    im1 = axes[1].imshow(disc_true,  cmap='RdBu_r', norm=disc_norm, origin='lower')
    im2 = axes[2].imshow(disc_pred,  cmap='RdBu_r', norm=disc_norm, origin='lower')
    im3 = axes[3].imshow(abs_err,    cmap='inferno', origin='lower')

    for ax, im, title in zip(axes,
                              [im0, im1, im2, im3],
                              ['Coarse T', 'True discrepancy',
                               'Predicted discrepancy', 'Abs error']):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.axis('off')


def plot_criticality_repartition(ax, disc_pred_all, y_true_flat, percentiles=(50, 75, 90, 95, 99)):
    """Histogram of |predicted discrepancy| with threshold lines showing cell selection rates."""
    abs_pred = np.abs(disc_pred_all.ravel())
    ax.hist(abs_pred, bins=120, color='mediumpurple', edgecolor='none', alpha=0.75)

    for p in percentiles:
        thresh = np.percentile(abs_pred, p)
        frac   = (abs_pred > thresh).mean() * 100
        ax.axvline(thresh, linestyle=':', lw=1.3,
                   label=f'P{p}: τ={thresh:.3f} → selects {frac:.1f}% cells')

    ax.set_xlabel('|Predicted discrepancy|  (model uncertainty proxy)')
    ax.set_ylabel('Cell count')
    ax.set_title('Cell Criticality Repartition\n(threshold → selected fraction)', fontsize=9)
    ax.legend(fontsize=7)


def plot_cumulative_error(ax, y_true_flat, y_pred_flat):
    """Cumulative sorted absolute error: shows how error is concentrated."""
    abs_err = np.abs(y_pred_flat - y_true_flat)
    sorted_err = np.sort(abs_err)
    cumfrac = np.linspace(0, 1, len(sorted_err))

    ax.plot(sorted_err, cumfrac * 100, color='darkorange', lw=1.5)

    for frac, color in [(50, 'limegreen'), (90, 'orange'), (99, 'red')]:
        v = np.percentile(sorted_err, frac)
        ax.axvline(v, linestyle='--', color=color, lw=1.2, label=f'P{frac}={v:.4f}')

    ax.set_xlabel('Absolute error')
    ax.set_ylabel('Cumulative % of cells')
    ax.set_title('Cumulative Error Distribution\n(fraction of cells ≤ error)', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, default='models/surrogate_256x256_cuda.pkl')
    parser.add_argument('--grid-size',  type=int, default=256)
    parser.add_argument('--n-samples',  type=int, default=40)
    parser.add_argument('--output',     type=str, default='reports/surrogate_eval.png')
    parser.add_argument('--data-dir', type=str, default='',
                        help='Evaluate on real solver exports (*.bin) instead of synthetic test generation.')
    parser.add_argument('--paired-data-dir', type=str, default='',
                        help='Evaluate on paired scenario_XXXX_coarse.bin and scenario_XXXX_fullfine.bin data.')
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    with open(args.model, 'rb') as f:
        payload = pickle.load(f)
    scaler = payload['scaler']
    model  = payload['model']
    meta   = payload.get('metadata', {})
    radius = meta.get('patch_radius', PATCH_RADIUS)
    logger.info(f"Loaded: {meta}")

    # Fresh test data or real exports
    G = args.grid_size
    if args.data_dir and args.paired_data_dir:
        raise ValueError("Use either --data-dir or --paired-data-dir, not both")

    if args.paired_data_dir:
        logger.info(f"Loading paired real evaluation data from: {args.paired_data_dir}")
        coarse, disc_true = load_paired_solver_training_data(args.paired_data_dir, expected_grid_size=G)
    elif args.data_dir:
        logger.info(f"Loading real evaluation data from: {args.data_dir}")
        coarse, disc_true = load_solver_training_data(args.data_dir, expected_grid_size=G)
    else:
        logger.info(f"Generating {args.n_samples} synthetic test samples ({G}x{G})...")
        coarse, disc_true = generate_test_data(G, args.n_samples, seed=99)

    if coarse.shape[0] > args.n_samples:
        coarse = coarse[:args.n_samples]
        disc_true = disc_true[:args.n_samples]

    # Inference
    logger.info("Running inference...")
    disc_pred = predict_discrepancy(coarse, scaler, model, radius)

    y_true_flat = disc_true.ravel()
    y_pred_flat = disc_pred.ravel()

    # Global metrics
    mae  = float(np.mean(np.abs(y_pred_flat - y_true_flat)))
    rmse = float(np.sqrt(np.mean((y_pred_flat - y_true_flat)**2)))
    ss_res = float(np.sum((y_true_flat - y_pred_flat)**2))
    ss_tot = float(np.sum((y_true_flat - y_true_flat.mean())**2))
    r2   = 1.0 - ss_res / (ss_tot + 1e-12)
    logger.info(f"Global:  MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}")

    # ── build dashboard ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    fig.suptitle(
        f'Surrogate Model Evaluation  —  {G}×{G} grid, {args.n_samples} test samples\n'
        f'MAE={mae:.5f}  RMSE={rmse:.5f}  R²={r2:.4f}  |  model: {Path(args.model).name}',
        fontsize=11, fontweight='bold'
    )

    gs = gridspec.GridSpec(3, 4, figure=fig)

    ax_scatter  = fig.add_subplot(gs[0, 0])
    ax_hist     = fig.add_subplot(gs[0, 1])
    ax_cumul    = fig.add_subplot(gs[0, 2])
    ax_crit     = fig.add_subplot(gs[0, 3])

    ax_coarse   = fig.add_subplot(gs[1, 0])
    ax_true     = fig.add_subplot(gs[1, 1])
    ax_pred     = fig.add_subplot(gs[1, 2])
    ax_abserr   = fig.add_subplot(gs[1, 3])

    ax_coarse2  = fig.add_subplot(gs[2, 0])
    ax_true2    = fig.add_subplot(gs[2, 1])
    ax_pred2    = fig.add_subplot(gs[2, 2])
    ax_abserr2  = fig.add_subplot(gs[2, 3])

    # Row 0: global statistics
    plot_scatter(ax_scatter, y_true_flat, y_pred_flat)
    plot_error_histogram(ax_hist, y_true_flat, y_pred_flat)
    plot_cumulative_error(ax_cumul, y_true_flat, y_pred_flat)
    plot_criticality_repartition(ax_crit, disc_pred, y_true_flat)

    # Row 1: spatial maps for sample 0 (worst/best chosen by median error)
    sample_mae = np.mean(np.abs(disc_pred - disc_true), axis=(1,2))
    idx_median = int(np.argsort(sample_mae)[len(sample_mae)//2])
    plot_spatial([ax_coarse, ax_true, ax_pred, ax_abserr],
                 coarse[idx_median], disc_true[idx_median], disc_pred[idx_median])
    ax_coarse.set_title(f'Sample #{idx_median} (median error={sample_mae[idx_median]:.4f})\nCoarse T',
                        fontsize=8)

    # Row 2: spatial maps for worst sample
    idx_worst = int(np.argmax(sample_mae))
    plot_spatial([ax_coarse2, ax_true2, ax_pred2, ax_abserr2],
                 coarse[idx_worst], disc_true[idx_worst], disc_pred[idx_worst])
    ax_coarse2.set_title(f'Sample #{idx_worst} (worst error={sample_mae[idx_worst]:.4f})\nCoarse T',
                         fontsize=8)

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    logger.info(f"Dashboard saved: {out}  ({out.stat().st_size/1024:.0f} KB)")
    plt.close(fig)


if __name__ == '__main__':
    main()
