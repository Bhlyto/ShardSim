#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from train_surrogate_3d import evaluate_model, load_paired_solver_training_data_3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--paired-data-dir', required=True)
    parser.add_argument('--output', default='reports/surrogate_3d_eval.json')
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        bundle = pickle.load(f)

    coarse, disc = load_paired_solver_training_data_3d(args.paired_data_dir)
    metrics = evaluate_model(
        coarse,
        disc,
        bundle['scaler'],
        bundle['model'],
        radius=bundle['metadata']['patch_radius'],
        active_threshold=bundle['metadata']['metrics'].get('active_threshold', 1e-8),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()