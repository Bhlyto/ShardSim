#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from train_surrogate import _load_single_solver_bin


def compare(reference_fine: np.ndarray, candidate_fine: np.ndarray) -> dict[str, float]:
    diff = candidate_fine - reference_fine
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((reference_fine - reference_fine.mean()) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    ref_sq = float(np.sum(reference_fine ** 2))
    global_error_norm = float(np.sqrt(ss_res) / np.sqrt(ref_sq + 1e-12))
    return {
        'mae_vs_reference': mae,
        'rmse_vs_reference': rmse,
        'r2_vs_reference': r2,
        'global_error_norm_vs_reference': global_error_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', action='append', default=[], help='label=path')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    _, reference_fine = _load_single_solver_bin(Path(args.reference))

    rows: list[dict[str, float | str]] = []
    for item in args.candidate:
        if '=' not in item:
            raise ValueError(f'Invalid candidate spec: {item}')
        label, file_path = item.split('=', 1)
        _, candidate_fine = _load_single_solver_bin(Path(file_path))
        metrics = compare(reference_fine, candidate_fine)
        rows.append({'label': label, **metrics})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == '__main__':
    main()