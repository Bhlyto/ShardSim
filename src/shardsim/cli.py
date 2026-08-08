from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from shardsim.execution import inspect_result, run_scenario_file
from shardsim.scenario import ScenarioValidationError, load_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shardsim",
        description="Validate and run reproducible physical scenarios.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate a V1 scenario JSON file.")
    validate.add_argument("scenario", type=Path)

    run = commands.add_parser("run", help="Run one independent physical scenario.")
    run.add_argument("scenario", type=Path)
    run.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser("inspect", help="Verify a result and its artifact checksums.")
    inspect.add_argument("result", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            scenario = load_scenario(arguments.scenario)
            _print({
                "valid": True,
                "schema_version": scenario.schema_version,
                "scenario_id": scenario.scenario_id,
                "model": scenario.model,
            })
            return 0
        if arguments.command == "run":
            result = run_scenario_file(arguments.scenario, arguments.output)
            _print(result)
            return int(result["exit_code"])
        _print(inspect_result(arguments.result))
        return 0
    except (ScenarioValidationError, FileExistsError, ValueError) as error:
        print(f"shardsim: error: {error}", file=sys.stderr)
        return 2


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
