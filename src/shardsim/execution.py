from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from shardsim.contracts import Fidelity
from shardsim.metrics import boundary_residual, maximum_principle_violation
from shardsim.scenario import MODEL_VERSION, Scenario, load_scenario
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.version import __version__


RESULT_SCHEMA_VERSION = "1.0"
_RESERVED_OUTPUTS = ("scenario.json", "result.json", "field.npy", "run.log")


def run_scenario_file(path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    return run_scenario(load_scenario(path), output_dir)


def run_scenario(scenario: Scenario, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    _prepare_output(output)
    scenario_payload = scenario.to_payload()
    scenario_text = _pretty_json(scenario_payload)
    scenario_path = output / "scenario.json"
    _atomic_write_text(scenario_path, scenario_text)
    scenario_sha256 = _sha256_bytes(_canonical_json(scenario_payload).encode("utf-8"))
    started_at = _utc_now()
    log_path = output / "run.log"

    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        _log(log, "INFO", f"starting scenario={scenario.scenario_id} model={scenario.model}")
        try:
            solver = HeatEquationSolver(safety_factor=scenario.solver.safety_factor)
            case = scenario.to_case()
            result = solver.solve(
                case,
                Fidelity.NOMINAL,
                scenario.solver.grid_shape,
            )
            field_path = output / "field.npy"
            _atomic_save_array(field_path, result.field)
            field_sha256 = _sha256_file(field_path)
            metrics = {
                "field_min": float(np.min(result.field)),
                "field_max": float(np.max(result.field)),
                "field_mean": float(np.mean(result.field)),
                "field_l2_norm": float(np.linalg.norm(result.field.ravel())),
                "boundary_residual": boundary_residual(result.field, scenario.boundaries),
                **maximum_principle_violation(result.field, case.initial_field, scenario.boundaries),
            }
            finished_at = _utc_now()
            record = {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "scenario_id": scenario.scenario_id,
                "status": "success",
                "exit_code": 0,
                "started_at": started_at,
                "finished_at": finished_at,
                "runtime_seconds": result.runtime_seconds,
                "shardsim_version": __version__,
                "model": scenario.model,
                "model_version": MODEL_VERSION,
                "seed": scenario.seed,
                "scenario_sha256": scenario_sha256,
                "reproducibility_key": _sha256_bytes(
                    f"{scenario_sha256}:{MODEL_VERSION}:{field_sha256}".encode("utf-8")
                ),
                "solver": {
                    "backend": scenario.solver.backend,
                    "algorithm": result.metadata["solver"],
                    "grid_shape": list(result.grid_shape),
                    "dt": result.dt,
                    "n_steps": result.n_steps,
                    "t_end": result.t_end,
                    "safety_factor": scenario.solver.safety_factor,
                    "stability_number": result.metadata["stability_number"],
                },
                "metrics": metrics,
                "artifacts": [
                    _artifact("scenario", scenario_path, output),
                    _artifact("field", field_path, output, sha256=field_sha256),
                    _artifact("log", log_path, output),
                ],
                "environment": environment_snapshot(),
            }
            _log(
                log,
                "INFO",
                f"completed scenario={scenario.scenario_id} steps={result.n_steps} "
                f"runtime_seconds={result.runtime_seconds:.9f}",
            )
        except Exception as error:
            finished_at = _utc_now()
            _log(log, "ERROR", f"{type(error).__name__}: {error}")
            record = {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "scenario_id": scenario.scenario_id,
                "status": "failed",
                "exit_code": 1,
                "started_at": started_at,
                "finished_at": finished_at,
                "shardsim_version": __version__,
                "model": scenario.model,
                "model_version": MODEL_VERSION,
                "seed": scenario.seed,
                "scenario_sha256": scenario_sha256,
                "error": {"type": type(error).__name__, "message": str(error)},
                "artifacts": [
                    _artifact("scenario", scenario_path, output),
                    _artifact("log", log_path, output),
                ],
                "environment": environment_snapshot(),
            }

    # Recompute the log checksum after its handle has been flushed and closed.
    for artifact in record["artifacts"]:
        if artifact["kind"] == "log":
            artifact["sha256"] = _sha256_file(log_path)
            artifact["bytes"] = log_path.stat().st_size
    _atomic_write_text(output / "result.json", _pretty_json(record))
    return record


def inspect_result(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read result {result_path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed ShardSim result")
    if payload.get("status") not in ("success", "failed"):
        raise ValueError("Result status must be 'success' or 'failed'")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Result artifacts must be an array")
    root = result_path.resolve().parent
    checked: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError("Malformed artifact entry")
        artifact_path = (root / artifact["path"]).resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Artifact escapes the result directory: {artifact['path']}") from error
        if not artifact_path.is_file():
            raise ValueError(f"Missing artifact: {artifact['path']}")
        actual = _sha256_file(artifact_path)
        if actual != artifact.get("sha256"):
            raise ValueError(f"Checksum mismatch for artifact: {artifact['path']}")
        checked.append({"path": artifact["path"], "sha256": actual})
    return {
        "valid": True,
        "scenario_id": payload.get("scenario_id"),
        "status": payload["status"],
        "model_version": payload.get("model_version"),
        "reproducibility_key": payload.get("reproducibility_key"),
        "artifacts": checked,
    }


@lru_cache(maxsize=1)
def environment_snapshot() -> dict[str, Any]:
    try:
        installed_version = importlib.metadata.version("shardsim")
    except importlib.metadata.PackageNotFoundError:
        installed_version = __version__
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "shardsim_version": installed_version,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain", "--untracked-files=no")),
    }


def _prepare_output(output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    conflicts = [name for name in _RESERVED_OUTPUTS if (output / name).exists()]
    if conflicts:
        raise FileExistsError(
            f"Output directory already contains ShardSim files: {', '.join(conflicts)}"
        )


def _artifact(
    kind: str,
    path: Path,
    root: Path,
    *,
    sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256 or _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _pretty_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _log(stream: Any, level: str, message: str) -> None:
    stream.write(f"{_utc_now()} {level} {message}\n")
    stream.flush()
