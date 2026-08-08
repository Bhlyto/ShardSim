from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from shardsim.campaign.core import ReproducibleCampaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shardsim-campaign",
        description="Create and manually run reproducible ShardSim campaigns.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init", help="Create an editable campaign.json.")
    initialize.add_argument("root", type=Path)
    initialize.add_argument("--name", required=True)
    initialize.add_argument("--seed", type=int, required=True)
    initialize.add_argument(
        "--profile",
        choices=("standard", "heat-wide"),
        default="standard",
    )
    initialize.add_argument("--force", action="store_true")

    lock = subparsers.add_parser("lock", help="Generate and hash the immutable case list.")
    lock.add_argument("root", type=Path)
    lock.add_argument("--force", action="store_true")

    list_cases = subparsers.add_parser("list", help="List locked cases without running them.")
    list_cases.add_argument("root", type=Path)
    _add_case_filters(list_cases, require_scope=False)

    run = subparsers.add_parser("run", help="Run selected pending nominal simulations.")
    run.add_argument("root", type=Path)
    _add_case_filters(run, require_scope=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--continue-on-error", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    status = subparsers.add_parser("status", help="Show completed and pending case counts.")
    status.add_argument("root", type=Path)

    train = subparsers.add_parser("train", help="Train a model from the locked training split.")
    train.add_argument("root", type=Path)
    train.add_argument("--allow-partial", action="store_true")
    train.add_argument(
        "--algorithm",
        choices=("heat-local-residual", "mean-delta", "heat-residual-unet"),
    )
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--width", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--device", choices=("cpu", "cuda"))

    full = subparsers.add_parser(
        "full",
        help="Run resumable simulation, training, and validation batches.",
    )
    full.add_argument("root", type=Path)
    full.add_argument("--cases-per-batch", type=int, default=50)
    full.add_argument("--max-batches", type=int)
    full.add_argument(
        "--algorithm",
        choices=("heat-local-residual", "mean-delta", "heat-residual-unet"),
        default="heat-residual-unet",
    )
    full.add_argument("--epochs", type=int)
    full.add_argument("--batch-size", type=int)
    full.add_argument("--width", type=int)
    full.add_argument("--learning-rate", type=float)
    full.add_argument("--device", choices=("cpu", "cuda"))
    full.add_argument("--test-at-end", action="store_true")
    full.add_argument("--continue-on-error", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate the model on a locked split.")
    evaluate.add_argument("root", type=Path)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--allow-partial", action="store_true")

    export = subparsers.add_parser(
        "export", help="Create concatenated CSV and NPZ result files."
    )
    export.add_argument("root", type=Path)

    dashboard = subparsers.add_parser(
        "dashboard", help="Create the standalone campaign result dashboard."
    )
    dashboard.add_argument("root", type=Path)
    dashboard.add_argument("--open", action="store_true", dest="open_browser")

    models = subparsers.add_parser("models", help="List reusable model versions.")
    models.add_argument("root", type=Path)

    activate = subparsers.add_parser(
        "activate", help="Select the model version used by evaluations."
    )
    activate.add_argument("root", type=Path)
    activate.add_argument("--key", required=True, help="Full model key or unique prefix.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "init":
        campaign = ReproducibleCampaign.initialize(
            arguments.root,
            arguments.name,
            arguments.seed,
            force=arguments.force,
            profile=arguments.profile,
        )
        _print({"created": str(campaign.spec_path)})
        return 0

    campaign = ReproducibleCampaign(arguments.root)
    if arguments.command == "lock":
        _print(campaign.lock(force=arguments.force))
    elif arguments.command == "list":
        definitions = campaign.load_definitions(
            split=arguments.split,
            family=arguments.family,
            case_ids=arguments.case_id,
        )
        for definition in definitions:
            print(json.dumps(definition.to_payload(), sort_keys=True))
    elif arguments.command == "run":
        if not (arguments.all or arguments.split or arguments.family or arguments.case_id):
            parser.error("run requires --split, --family, --case-id, or explicit --all")
        result = campaign.run(
            split=arguments.split,
            family=arguments.family,
            case_ids=arguments.case_id,
            limit=arguments.limit,
            continue_on_error=arguments.continue_on_error,
            dry_run=arguments.dry_run,
            progress=None if arguments.dry_run else print,
        )
        _print(result)
    elif arguments.command == "status":
        _print(campaign.status().to_payload())
    elif arguments.command == "train":
        _print(
            campaign.train(
                allow_partial=arguments.allow_partial,
                algorithm=arguments.algorithm,
                model_overrides={
                    "epochs": arguments.epochs,
                    "batch_size": arguments.batch_size,
                    "width": arguments.width,
                    "learning_rate": arguments.learning_rate,
                    "device": arguments.device,
                },
            )
        )
    elif arguments.command == "full":
        _print(
            campaign.run_full_training_campaign(
                cases_per_batch=arguments.cases_per_batch,
                algorithm=arguments.algorithm,
                model_overrides={
                    "epochs": arguments.epochs,
                    "batch_size": arguments.batch_size,
                    "width": arguments.width,
                    "learning_rate": arguments.learning_rate,
                    "device": arguments.device,
                },
                max_batches=arguments.max_batches,
                test_at_end=arguments.test_at_end,
                continue_on_error=arguments.continue_on_error,
                progress=print,
            )
        )
    elif arguments.command == "evaluate":
        _print(
            campaign.evaluate(
                split=arguments.split,
                allow_partial=arguments.allow_partial,
            )
        )
    elif arguments.command == "export":
        _print(campaign.export_results())
    elif arguments.command == "dashboard":
        dashboard_path = campaign.dashboard(open_browser=arguments.open_browser)
        _print({"dashboard": str(dashboard_path)})
    elif arguments.command == "models":
        _print({"models": campaign.list_models()})
    elif arguments.command == "activate":
        _print(campaign.activate_model(arguments.key))
    return 0


def _add_case_filters(parser: argparse.ArgumentParser, require_scope: bool) -> None:
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--family")
    parser.add_argument("--case-id", action="append", default=[])
    if require_scope:
        parser.add_argument("--all", action="store_true", help="Explicitly select all cases.")


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
