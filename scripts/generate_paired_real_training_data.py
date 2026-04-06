#!/usr/bin/env python3
"""
Generate paired real training samples by sweeping source parameters.

For each scenario:
- Run adaptive config and keep its coarse field export.
- Run full-fine config and keep its fine-field reference export.
- Rename outputs to:
    scenario_XXXX_coarse.bin
    scenario_XXXX_fullfine.bin

Usage:
    python scripts/generate_paired_real_training_data.py \
    --n-scenarios 32 \
    --output-dir runs/training_pairs

Works for both 2D and 3D solver paths. Use --grid-z > 1 to emit paired 3D exports.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import time
from pathlib import Path


def latest_bin(path: Path) -> Path:
    bins = sorted(path.glob("*.bin"), key=lambda p: p.stat().st_mtime)
    if not bins:
        raise RuntimeError(f"No .bin files found in {path}")
    return bins[-1]


def write_config(path: Path, cfg: dict[str, str | int | float | bool]) -> None:
    lines: list[str] = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_solver(repo_root: Path, config_path: Path) -> None:
    cmd = ["./build/shardsim_cli", str(config_path)]
    subprocess.run(cmd, cwd=repo_root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenarios", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="runs/training_pairs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--grid-z", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--complexity",
        type=str,
        default="standard",
        choices=["standard", "complex", "stress"],
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="id",
        choices=["id", "ood", "far_ood"],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / args.output_dir
    raw_dir = out_dir / "raw"
    cfg_dir = out_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    common = {
        "grid_x": args.grid_size,
        "grid_y": args.grid_size,
        "grid_z": args.grid_z,
        "steps": args.steps,
        "dt": 0.001,
        "alpha": 1.0,
        "coarse_tolerance": 0.05,
        "fine_tolerance": 0.03,
        "refine_local_error_tau": 0.05,
        "refine_uncertainty_tau": 0.20,
        "memory_ceiling_gb": 16,
        "wallclock_limit_minutes": 0,
        "halo_overhead_ratio_max": 0.95,
        "deterministic_mode": True,
        "partitioning_policy": "strict_geometric",
        "export_training_data": True,
        "training_data_export_dir": raw_dir.as_posix(),
    }

    for sid in range(args.n_scenarios):
        x_frac = rng.uniform(0.10, 0.90)
        y_frac = rng.uniform(0.10, 0.90)
        temp = rng.uniform(30.0, 260.0)

        alpha = 1.0
        dt = 0.001
        source2_enabled = False
        source2_x = 0.75
        source2_y = 0.25
        source2_z = 0.50
        source2_temp = 50.0
        z_frac = 0.50 if args.grid_z <= 1 else rng.uniform(0.10, 0.90)

        if args.complexity in {"complex", "stress"}:
            if args.distribution == "id":
                alpha = rng.uniform(0.7, 1.5)
                dt = rng.uniform(0.0005, 0.0014)
                source2_enabled = (rng.random() < (0.6 if args.complexity == "complex" else 0.85))
            elif args.distribution == "ood":
                # Intentional OOD regime: colder/hotter diffusivity, shifted dt, denser multi-source usage.
                alpha = rng.choice([
                    rng.uniform(0.35, 0.65),
                    rng.uniform(1.6, 2.1),
                ])
                dt = rng.choice([
                    rng.uniform(0.00025, 0.00045),
                    rng.uniform(0.0017, 0.0022),
                ])
                source2_enabled = (rng.random() < 0.9)
            else:
                # Far-OOD regime: much more extreme parameters and nearly-always multi-source.
                alpha = rng.choice([
                    rng.uniform(0.15, 0.30),
                    rng.uniform(2.3, 3.0),
                ])
                dt = rng.choice([
                    rng.uniform(0.00010, 0.00022),
                    rng.uniform(0.0026, 0.0035),
                ])
                source2_enabled = True
            source2_x = rng.uniform(0.10, 0.90)
            source2_y = rng.uniform(0.10, 0.90)
            source2_z = 0.50 if args.grid_z <= 1 else rng.uniform(0.10, 0.90)
            if args.complexity == "stress":
                temp = rng.uniform(20.0, 320.0)
                source2_temp = rng.uniform(10.0, 280.0)
            else:
                source2_temp = rng.uniform(15.0, 220.0)

        scenario = {
            "source_x_fraction": round(x_frac, 6),
            "source_y_fraction": round(y_frac, 6),
            "source_z_fraction": round(z_frac, 6),
            "source_temperature": round(temp, 6),
            "alpha": round(alpha, 6),
            "dt": round(dt, 6),
            "source2_enabled": source2_enabled,
            "source2_x_fraction": round(source2_x, 6),
            "source2_y_fraction": round(source2_y, 6),
            "source2_z_fraction": round(source2_z, 6),
            "source2_temperature": round(source2_temp, 6),
        }

        adaptive_cfg = {
            **common,
            **scenario,
            "min_critical_fraction": 0.0,
        }
        fullfine_cfg = {
            **common,
            **scenario,
            "min_critical_fraction": 1.0,
        }

        adaptive_cfg_path = cfg_dir / f"scenario_{sid:04d}_adaptive.yaml"
        fullfine_cfg_path = cfg_dir / f"scenario_{sid:04d}_fullfine.yaml"
        write_config(adaptive_cfg_path, adaptive_cfg)
        write_config(fullfine_cfg_path, fullfine_cfg)

        run_solver(repo_root, adaptive_cfg_path)
        coarse_src = latest_bin(raw_dir)
        coarse_dst = out_dir / f"scenario_{sid:04d}_coarse.bin"
        coarse_src.replace(coarse_dst)

        # Ensure mtimes differ even on coarse timestamp filesystems.
        time.sleep(0.01)

        run_solver(repo_root, fullfine_cfg_path)
        fine_src = latest_bin(raw_dir)
        fine_dst = out_dir / f"scenario_{sid:04d}_fullfine.bin"
        fine_src.replace(fine_dst)

        print(f"generated scenario {sid:04d}: {coarse_dst.name}, {fine_dst.name}")

    print(f"Done. Paired dataset in: {out_dir}")


if __name__ == "__main__":
    main()
