from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from shardsim.canonical import FieldLocation
from shardsim.contracts import Fidelity, ProblemSpec, SimulationResult
from shardsim.pipeline import ReferenceSample


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"))


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceDatasetStore:
    manifest_version = 1

    def __init__(self, root: str | Path, dataset_id: str = "shardsim-reference") -> None:
        self.root = Path(root)
        self.dataset_id = dataset_id
        if not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty.")

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def samples_path(self) -> Path:
        return self.root / "samples"

    def _empty_manifest(self, sample: ReferenceSample) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "dataset_id": self.dataset_id,
            "domain": sample.problem.domain,
            "equation": sample.problem.equation,
            "problem_schema_version": sample.problem.schema_version,
            "coarse_shape": list(sample.coarse.grid_shape),
            "nominal_shape": list(sample.nominal.grid_shape),
            "samples": [],
        }

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("manifest_version") != self.manifest_version:
            raise ValueError("Unsupported reference dataset manifest version.")
        if manifest.get("dataset_id") != self.dataset_id:
            raise ValueError("Dataset identifier does not match the existing manifest.")
        return manifest

    def _validate_sample(self, manifest: Mapping[str, Any], sample: ReferenceSample) -> None:
        expected = (
            manifest["domain"],
            manifest["equation"],
            manifest["problem_schema_version"],
            tuple(manifest["coarse_shape"]),
            tuple(manifest["nominal_shape"]),
        )
        observed = (
            sample.problem.domain,
            sample.problem.equation,
            sample.problem.schema_version,
            sample.coarse.grid_shape,
            sample.nominal.grid_shape,
        )
        if observed != expected:
            raise ValueError("Reference sample is incompatible with the dataset manifest.")

    def add(self, sample: ReferenceSample, replace: bool = False) -> Path:
        self.samples_path.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest() or self._empty_manifest(sample)
        self._validate_sample(manifest, sample)

        existing = {entry["case_id"]: entry for entry in manifest["samples"]}
        if sample.case_id in existing and not replace:
            raise ValueError(f"Case {sample.case_id!r} already exists in the dataset.")

        file_stem = hashlib.sha256(sample.case_id.encode("utf-8")).hexdigest()[:20]
        relative_path = Path("samples") / f"{file_stem}.npz"
        target = self.root / relative_path
        temporary = target.with_suffix(".tmp.npz")
        payload = {
            "case_id": sample.case_id,
            "case_metadata": sample.case_metadata,
            "problem": {
                "domain": sample.problem.domain,
                "equation": sample.problem.equation,
                "parameters": sample.problem.parameters,
                "t_end": sample.problem.t_end,
                "extent": sample.problem.extent,
                "input_units": sample.problem.input_units,
                "output_units": sample.problem.output_units,
                "schema_version": sample.problem.schema_version,
            },
            "coarse": {
                "fidelity": sample.coarse.fidelity.value,
                "t_end": sample.coarse.t_end,
                "dt": sample.coarse.dt,
                "n_steps": sample.coarse.n_steps,
                "runtime_seconds": sample.coarse.runtime_seconds,
                "field_location": sample.coarse.field_location.value,
                "metadata": sample.coarse.metadata,
            },
            "nominal": {
                "fidelity": sample.nominal.fidelity.value,
                "t_end": sample.nominal.t_end,
                "dt": sample.nominal.dt,
                "n_steps": sample.nominal.n_steps,
                "runtime_seconds": sample.nominal.runtime_seconds,
                "field_location": sample.nominal.field_location.value,
                "metadata": sample.nominal.metadata,
            },
            "metrics": sample.metrics,
        }
        np.savez_compressed(
            temporary,
            coarse_field=sample.coarse.field,
            nominal_field=sample.nominal.field,
            coarse_on_nominal=sample.coarse_on_nominal,
            delta=sample.delta,
            error_map=sample.error_map,
            metadata_json=np.array(_canonical_json(payload)),
        )
        temporary.replace(target)
        entry = {
            "case_id": sample.case_id,
            "path": relative_path.as_posix(),
            "sha256": _file_checksum(target),
        }
        manifest["samples"] = sorted(
            [item for item in manifest["samples"] if item["case_id"] != sample.case_id]
            + [entry],
            key=lambda item: item["case_id"],
        )
        manifest_temporary = self.manifest_path.with_suffix(".tmp.json")
        manifest_temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_temporary.replace(self.manifest_path)
        return target

    def case_ids(self) -> tuple[str, ...]:
        manifest = self._read_manifest()
        if manifest is None:
            return ()
        return tuple(entry["case_id"] for entry in manifest["samples"])

    def load_all(self, verify_checksums: bool = True) -> tuple[ReferenceSample, ...]:
        manifest = self._read_manifest()
        if manifest is None:
            return ()
        samples: list[ReferenceSample] = []
        for entry in manifest["samples"]:
            path = self.root / entry["path"]
            if verify_checksums and _file_checksum(path) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch for reference case {entry['case_id']!r}.")
            with np.load(path, allow_pickle=False) as archive:
                payload = json.loads(str(archive["metadata_json"].item()))
                problem_payload = payload["problem"]
                problem = ProblemSpec(
                    domain=problem_payload["domain"],
                    equation=problem_payload["equation"],
                    parameters=problem_payload["parameters"],
                    t_end=problem_payload["t_end"],
                    extent=tuple(problem_payload["extent"]),
                    input_units=problem_payload["input_units"],
                    output_units=problem_payload["output_units"],
                    schema_version=problem_payload["schema_version"],
                )
                coarse_payload = payload["coarse"]
                nominal_payload = payload["nominal"]
                coarse = SimulationResult(
                    case_id=payload["case_id"],
                    fidelity=Fidelity(coarse_payload["fidelity"]),
                    field=archive["coarse_field"],
                    t_end=coarse_payload["t_end"],
                    dt=coarse_payload["dt"],
                    n_steps=coarse_payload["n_steps"],
                    runtime_seconds=coarse_payload["runtime_seconds"],
                    field_location=FieldLocation(coarse_payload.get("field_location", "point")),
                    metadata=coarse_payload["metadata"],
                )
                nominal = SimulationResult(
                    case_id=payload["case_id"],
                    fidelity=Fidelity(nominal_payload["fidelity"]),
                    field=archive["nominal_field"],
                    t_end=nominal_payload["t_end"],
                    dt=nominal_payload["dt"],
                    n_steps=nominal_payload["n_steps"],
                    runtime_seconds=nominal_payload["runtime_seconds"],
                    field_location=FieldLocation(nominal_payload.get("field_location", "point")),
                    metadata=nominal_payload["metadata"],
                )
                samples.append(
                    ReferenceSample(
                        case_id=payload["case_id"],
                        problem=problem,
                        coarse=coarse,
                        nominal=nominal,
                        coarse_on_nominal=archive["coarse_on_nominal"],
                        delta=archive["delta"],
                        error_map=archive["error_map"],
                        metrics=payload["metrics"],
                        case_metadata=payload.get("case_metadata", {}),
                    )
                )
        return tuple(samples)
