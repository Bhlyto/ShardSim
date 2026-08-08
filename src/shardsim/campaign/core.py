from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from shardsim.contracts import BoundaryConditions, SimulationCase
from shardsim.dataset import ReferenceDatasetStore
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.metrics import (
    boundary_residual,
    compare_fields,
    compare_gradients,
    maximum_principle_violation,
)
from shardsim.pipeline import BootstrapPipeline, FidelityPlan, ReferenceSample
from shardsim.solvers.base import Solver
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.solvers.openfoam import OPENFOAM_IMAGE, OpenFOAMAdapter
from shardsim.surrogates.heat_local import HeatLocalResidualSurrogate
from shardsim.surrogates.mean_delta import MeanDeltaSurrogate


CAMPAIGN_SCHEMA_VERSION = 1
CASE_GENERATOR_VERSION = 1
VALID_SPLITS = ("train", "validation", "test")


class CampaignLockError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HeatCaseDefinition:
    case_id: str
    family: str
    split: str
    alpha: float
    t_end: float
    extent: tuple[float, float]
    center: tuple[float, float]
    sigma: tuple[float, float]
    amplitude: float
    baseline: float
    boundaries: BoundaryConditions
    initial_shape: tuple[int, int]
    definition_sha256: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HeatCaseDefinition":
        definition_payload = {
            key: payload[key]
            for key in (
                "family",
                "split",
                "alpha",
                "t_end",
                "extent",
                "center",
                "sigma",
                "amplitude",
                "baseline",
                "boundaries",
                "initial_shape",
            )
        }
        expected_hash = _sha256_text(_canonical_json(definition_payload))
        if payload.get("definition_sha256") != expected_hash:
            raise CampaignLockError(f"Case definition hash mismatch for {payload.get('case_id')!r}.")
        family = str(payload["family"])
        case_id = str(payload["case_id"])
        if not case_id.endswith(expected_hash[:10]) or not case_id.startswith(_slug(family)):
            raise CampaignLockError(f"Case identifier is inconsistent for {case_id!r}.")
        boundaries = payload["boundaries"]
        return cls(
            case_id=case_id,
            family=family,
            split=str(payload["split"]),
            alpha=float(payload["alpha"]),
            t_end=float(payload["t_end"]),
            extent=tuple(float(value) for value in payload["extent"]),
            center=tuple(float(value) for value in payload["center"]),
            sigma=tuple(float(value) for value in payload["sigma"]),
            amplitude=float(payload["amplitude"]),
            baseline=float(payload["baseline"]),
            boundaries=BoundaryConditions(
                top=float(boundaries["top"]),
                bottom=float(boundaries["bottom"]),
                left=float(boundaries["left"]),
                right=float(boundaries["right"]),
            ),
            initial_shape=tuple(int(value) for value in payload["initial_shape"]),
            definition_sha256=expected_hash,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "split": self.split,
            "alpha": self.alpha,
            "t_end": self.t_end,
            "extent": list(self.extent),
            "center": list(self.center),
            "sigma": list(self.sigma),
            "amplitude": self.amplitude,
            "baseline": self.baseline,
            "boundaries": {
                "top": self.boundaries.top,
                "bottom": self.boundaries.bottom,
                "left": self.boundaries.left,
                "right": self.boundaries.right,
            },
            "initial_shape": list(self.initial_shape),
            "definition_sha256": self.definition_sha256,
        }

    def to_case(self) -> SimulationCase:
        return make_heat_case(
            case_id=self.case_id,
            alpha=self.alpha,
            t_end=self.t_end,
            extent=self.extent,
            boundaries=self.boundaries,
            initial_field=gaussian_initial_field(
                self.initial_shape,
                center=self.center,
                sigma=self.sigma,
                amplitude=self.amplitude,
                baseline=self.baseline,
            ),
            metadata={
                "campaign_family": self.family,
                "campaign_split": self.split,
                "case_definition_sha256": self.definition_sha256,
                "case_generator_version": CASE_GENERATOR_VERSION,
                "initial_center": self.center,
                "initial_sigma": self.sigma,
                "initial_amplitude": self.amplitude,
                "initial_baseline": self.baseline,
            },
        )


@dataclass(frozen=True, slots=True)
class CampaignStatus:
    campaign_name: str
    spec_sha256: str
    cases_sha256: str
    output_key: str
    split_counts: Mapping[str, Mapping[str, int]]
    total_cases: int
    completed_cases: int
    pending_cases: int
    model_exists: bool
    model_count: int
    active_model_key: str | None
    evaluations: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "campaign_name": self.campaign_name,
            "spec_sha256": self.spec_sha256,
            "cases_sha256": self.cases_sha256,
            "output_key": self.output_key,
            "split_counts": self.split_counts,
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "pending_cases": self.pending_cases,
            "model_exists": self.model_exists,
            "model_count": self.model_count,
            "active_model_key": self.active_model_key,
            "evaluations": list(self.evaluations),
        }


def default_heat_campaign_spec(name: str, seed: int) -> dict[str, Any]:
    if not name.strip():
        raise ValueError("Campaign name cannot be empty.")
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_name": name,
        "seed": int(seed),
        "domain": "heat-2d",
        "equation": "du/dt=alpha*laplacian(u)",
        "fidelity": {
            "coarse_shape": [17, 17],
            "nominal_shape": [32, 32],
        },
        "solvers": {
            "coarse": "internal",
            "nominal": "openfoam",
            "openfoam_image": OPENFOAM_IMAGE,
        },
        "design": {
            "extent": [1.0, 1.0],
            "initial_shape": [65, 65],
            "boundaries": {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
            "families": {
                "train-centered": {
                    "split": "train",
                    "count": 12,
                    "alpha_range": [0.015, 0.030],
                    "t_end_range": [0.015, 0.050],
                    "center_x_range": [0.35, 0.65],
                    "center_y_range": [0.35, 0.65],
                    "sigma_x_range": [0.070, 0.130],
                    "sigma_y_range": [0.070, 0.130],
                    "amplitude_range": [0.8, 1.2],
                    "baseline_range": [0.0, 0.0],
                },
                "validation-centered": {
                    "split": "validation",
                    "count": 4,
                    "alpha_range": [0.016, 0.029],
                    "t_end_range": [0.018, 0.045],
                    "center_x_range": [0.38, 0.62],
                    "center_y_range": [0.38, 0.62],
                    "sigma_x_range": [0.075, 0.125],
                    "sigma_y_range": [0.075, 0.125],
                    "amplitude_range": [0.85, 1.15],
                    "baseline_range": [0.0, 0.0],
                },
                "test-corner-nw": {
                    "split": "test",
                    "count": 3,
                    "alpha_range": [0.015, 0.030],
                    "t_end_range": [0.015, 0.050],
                    "center_x_range": [0.15, 0.30],
                    "center_y_range": [0.70, 0.85],
                    "sigma_x_range": [0.060, 0.110],
                    "sigma_y_range": [0.060, 0.110],
                    "amplitude_range": [0.8, 1.2],
                    "baseline_range": [0.0, 0.0],
                },
                "test-fast": {
                    "split": "test",
                    "count": 3,
                    "alpha_range": [0.035, 0.060],
                    "t_end_range": [0.015, 0.050],
                    "center_x_range": [0.35, 0.65],
                    "center_y_range": [0.35, 0.65],
                    "sigma_x_range": [0.070, 0.130],
                    "sigma_y_range": [0.070, 0.130],
                    "amplitude_range": [0.8, 1.2],
                    "baseline_range": [0.0, 0.0],
                },
            },
        },
        "model": {
            "algorithm": "heat-local-residual",
            "artifact_name": "heat-local-residual.npz",
            "ridge": 0.03,
            "uncertainty_ood_scale": 0.25,
        },
        "quality": {
            "max_relative_l2": 0.20,
            "max_error_ratio_vs_coarse": 1.0,
            "min_coverage_2sigma": 0.50,
        },
    }


def wide_heat_campaign_spec(name: str, seed: int) -> dict[str, Any]:
    spec = default_heat_campaign_spec(name, seed)
    spec["design"]["families"] = {
        "train-global": _heat_family("train", 240, (0.008, 0.080), (0.005, 0.120), (0.10, 0.90), (0.10, 0.90), (0.035, 0.200)),
        "train-corner-nw": _heat_family("train", 60, (0.008, 0.080), (0.005, 0.120), (0.08, 0.30), (0.70, 0.92), (0.035, 0.160)),
        "train-corner-ne": _heat_family("train", 60, (0.008, 0.080), (0.005, 0.120), (0.70, 0.92), (0.70, 0.92), (0.035, 0.160)),
        "train-corner-sw": _heat_family("train", 60, (0.008, 0.080), (0.005, 0.120), (0.08, 0.30), (0.08, 0.30), (0.035, 0.160)),
        "train-corner-se": _heat_family("train", 60, (0.008, 0.080), (0.005, 0.120), (0.70, 0.92), (0.08, 0.30), (0.035, 0.160)),
        "train-fast": _heat_family("train", 60, (0.060, 0.120), (0.005, 0.060), (0.10, 0.90), (0.10, 0.90), (0.035, 0.180)),
        "train-long": _heat_family("train", 60, (0.005, 0.040), (0.080, 0.200), (0.10, 0.90), (0.10, 0.90), (0.035, 0.180)),
        "validation-global": _heat_family("validation", 40, (0.009, 0.078), (0.008, 0.115), (0.12, 0.88), (0.12, 0.88), (0.040, 0.190)),
        "validation-corners": _heat_family("validation", 20, (0.009, 0.078), (0.008, 0.115), (0.06, 0.94), (0.06, 0.94), (0.040, 0.150)),
        "validation-fast": _heat_family("validation", 10, (0.065, 0.115), (0.008, 0.055), (0.12, 0.88), (0.12, 0.88), (0.040, 0.170)),
        "validation-long": _heat_family("validation", 10, (0.007, 0.038), (0.085, 0.190), (0.12, 0.88), (0.12, 0.88), (0.040, 0.170)),
        "test-global": _heat_family("test", 30, (0.008, 0.080), (0.005, 0.120), (0.10, 0.90), (0.10, 0.90), (0.035, 0.200)),
        "test-corners": _heat_family("test", 20, (0.008, 0.080), (0.005, 0.120), (0.05, 0.95), (0.05, 0.95), (0.035, 0.160)),
        "test-fast": _heat_family("test", 15, (0.060, 0.120), (0.005, 0.060), (0.10, 0.90), (0.10, 0.90), (0.035, 0.180)),
        "test-long": _heat_family("test", 15, (0.005, 0.040), (0.080, 0.200), (0.10, 0.90), (0.10, 0.90), (0.035, 0.180)),
    }
    spec["model"] = {
        "algorithm": "heat-residual-unet",
        "artifact_name": "heat-residual-unet.pt",
        "width": 16,
        "epochs": 150,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 1e-6,
        "gradient_weight": 0.05,
        "uncertainty_ood_scale": 0.25,
        "seed": int(seed),
        "device": "cpu",
        "deterministic": True,
    }
    return spec


def _heat_family(
    split: str,
    count: int,
    alpha_range: tuple[float, float],
    t_end_range: tuple[float, float],
    center_x_range: tuple[float, float],
    center_y_range: tuple[float, float],
    sigma_range: tuple[float, float],
) -> dict[str, Any]:
    return {
        "split": split,
        "count": count,
        "alpha_range": list(alpha_range),
        "t_end_range": list(t_end_range),
        "center_x_range": list(center_x_range),
        "center_y_range": list(center_y_range),
        "sigma_x_range": list(sigma_range),
        "sigma_y_range": list(sigma_range),
        "amplitude_range": [0.4, 1.8],
        "baseline_range": [0.0, 0.2],
    }


class ReproducibleCampaign:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def spec_path(self) -> Path:
        return self.root / "campaign.json"

    @property
    def lock_path(self) -> Path:
        return self.root / "campaign.lock.json"

    @property
    def cases_path(self) -> Path:
        return self.root / "cases.jsonl"

    @classmethod
    def initialize(
        cls,
        root: str | Path,
        name: str,
        seed: int,
        force: bool = False,
        profile: str = "standard",
    ) -> "ReproducibleCampaign":
        campaign = cls(root)
        campaign.root.mkdir(parents=True, exist_ok=True)
        if campaign.spec_path.exists() and not force:
            raise FileExistsError(f"Campaign specification already exists: {campaign.spec_path}")
        if profile == "standard":
            spec = default_heat_campaign_spec(name, seed)
        elif profile == "heat-wide":
            spec = wide_heat_campaign_spec(name, seed)
        else:
            raise ValueError(f"Unknown campaign profile: {profile}")
        _atomic_write_json(campaign.spec_path, spec)
        return campaign

    def lock(self, force: bool = False) -> Mapping[str, Any]:
        spec = self._read_and_validate_spec()
        spec_sha256 = _sha256_file(self.spec_path)
        if self.lock_path.exists() and not force:
            existing = _read_json(self.lock_path)
            if existing.get("spec_sha256") == spec_sha256:
                self.verify_lock()
                return existing
            raise CampaignLockError(
                "campaign.json changed after locking; use lock --force to create a new lock."
            )

        definitions = self._generate_definitions(spec)
        cases_text = "".join(_canonical_json(definition.to_payload()) + "\n" for definition in definitions)
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.cases_path, cases_text)
        split_counts = {
            split: sum(definition.split == split for definition in definitions)
            for split in VALID_SPLITS
        }
        lock_payload = {
            "lock_schema_version": 1,
            "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
            "case_generator_version": CASE_GENERATOR_VERSION,
            "design_algorithm": "family-independent-latin-hypercube",
            "bit_generator": "PCG64",
            "campaign_name": spec["campaign_name"],
            "seed": spec["seed"],
            "spec_sha256": spec_sha256,
            "cases_sha256": _sha256_file(self.cases_path),
            "case_count": len(definitions),
            "split_counts": split_counts,
            "numpy_version": np.__version__,
            "solver_contract": spec["solvers"],
            "fidelity_contract": spec["fidelity"],
        }
        _atomic_write_json(self.lock_path, lock_payload)
        return lock_payload

    def verify_lock(self) -> Mapping[str, Any]:
        if not self.lock_path.is_file() or not self.cases_path.is_file():
            raise CampaignLockError("Campaign is not locked. Run the lock command first.")
        lock_payload = _read_json(self.lock_path)
        if lock_payload.get("lock_schema_version") != 1:
            raise CampaignLockError("Unsupported campaign lock version.")
        if _sha256_file(self.spec_path) != lock_payload.get("spec_sha256"):
            raise CampaignLockError("campaign.json does not match campaign.lock.json.")
        if _sha256_file(self.cases_path) != lock_payload.get("cases_sha256"):
            raise CampaignLockError("cases.jsonl does not match campaign.lock.json.")
        definitions = self.load_definitions(verify=False)
        if len(definitions) != lock_payload.get("case_count"):
            raise CampaignLockError("Locked case count does not match cases.jsonl.")
        return lock_payload

    def load_definitions(
        self,
        split: str | None = None,
        family: str | None = None,
        case_ids: Sequence[str] | None = None,
        verify: bool = True,
    ) -> tuple[HeatCaseDefinition, ...]:
        if verify:
            self.verify_lock()
        requested_ids = set(case_ids or ())
        definitions: list[HeatCaseDefinition] = []
        with self.cases_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                definition = HeatCaseDefinition.from_payload(json.loads(line))
                if split is not None and definition.split != split:
                    continue
                if family is not None and definition.family != family:
                    continue
                if requested_ids and definition.case_id not in requested_ids:
                    continue
                definitions.append(definition)
        if requested_ids - {definition.case_id for definition in definitions}:
            missing = sorted(requested_ids - {definition.case_id for definition in definitions})
            raise ValueError(f"Unknown or filtered case identifiers: {', '.join(missing)}")
        return tuple(definitions)

    def run(
        self,
        split: str | None = None,
        family: str | None = None,
        case_ids: Sequence[str] | None = None,
        limit: int | None = None,
        continue_on_error: bool = False,
        dry_run: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> Mapping[str, Any]:
        lock_payload = self.verify_lock()
        spec = self._read_and_validate_spec()
        definitions = self.load_definitions(split, family, case_ids)
        store = self._store(lock_payload)
        completed_ids = set(store.case_ids())
        pending = tuple(definition for definition in definitions if definition.case_id not in completed_ids)
        if limit is not None:
            if limit < 0:
                raise ValueError("limit cannot be negative.")
            pending = pending[:limit]
        if dry_run:
            return {
                "dry_run": True,
                "selected": len(definitions),
                "pending": len(pending),
                "case_ids": [definition.case_id for definition in pending],
            }

        coarse_solver, nominal_solver = self._solvers(spec)
        pipeline = BootstrapPipeline(
            coarse_solver,
            self._plan(spec),
            nominal_solver=nominal_solver,
        )
        run_record = self._new_run_record(lock_payload, split, family, case_ids, pending, pending)
        run_path = self._runs_path(lock_payload) / f"{run_record['run_id']}.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(run_path, run_record)

        for index, definition in enumerate(pending, start=1):
            if progress is not None:
                progress(f"[{index}/{len(pending)}] {definition.case_id}")
            try:
                sample = pipeline.run_case(definition.to_case())
                store.add(sample)
                run_record["completed_case_ids"].append(definition.case_id)
            except Exception as error:
                run_record["failures"].append(
                    {
                        "case_id": definition.case_id,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                run_record["status"] = "failed"
                run_record["finished_at_utc"] = _utc_now()
                _atomic_write_json(run_path, run_record)
                if not continue_on_error:
                    raise
            _atomic_write_json(run_path, run_record)

        run_record["status"] = "completed" if not run_record["failures"] else "completed-with-errors"
        run_record["finished_at_utc"] = _utc_now()
        run_record["dataset_manifest_sha256"] = (
            _sha256_file(store.manifest_path) if store.manifest_path.is_file() else None
        )
        _atomic_write_json(run_path, run_record)
        return run_record

    def run_full_training_campaign(
        self,
        cases_per_batch: int = 50,
        algorithm: str = "heat-residual-unet",
        model_overrides: Mapping[str, Any] | None = None,
        max_batches: int | None = None,
        test_at_end: bool = False,
        continue_on_error: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> Mapping[str, Any]:
        if cases_per_batch < 1:
            raise ValueError("cases_per_batch must be positive.")
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be positive.")
        lock_payload = self.verify_lock()
        spec = self._read_and_validate_spec()
        training_spec = _with_model_spec(spec, algorithm, model_overrides)
        lineage_id = _model_lineage_id(training_spec)
        minimum_cases = 4 if algorithm == "heat-residual-unet" else 2
        configuration = {
            "algorithm": algorithm,
            "cases_per_batch": cases_per_batch,
            "max_batches": max_batches,
            "test_at_end": test_at_end,
            "continue_on_error": continue_on_error,
            "model_parameters": _model_parameters(training_spec),
        }
        orchestration_key = _sha256_text(
            _canonical_json(
                {
                    "spec_sha256": lock_payload["spec_sha256"],
                    "configuration": configuration,
                }
            )
        )
        report_path = (
            self._reports_path(lock_payload)
            / f"full-campaign-{orchestration_key[:12]}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.is_file():
            report = _read_json(report_path)
            report["status"] = "running"
            report.pop("error", None)
        else:
            report = {
                "report_version": 1,
                "orchestration_key": orchestration_key,
                "campaign_name": spec["campaign_name"],
                "spec_sha256": lock_payload["spec_sha256"],
                "cases_sha256": lock_payload["cases_sha256"],
                "configuration": configuration,
                "status": "running",
                "started_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "batches": [],
            }
        _atomic_write_json(report_path, report)

        def emit(message: str) -> None:
            if progress is not None:
                progress(message)

        def persist() -> None:
            report["updated_at_utc"] = _utc_now()
            report["campaign_status"] = self.status().to_payload()
            _atomic_write_json(report_path, report)

        def require_completed_split(split: str) -> None:
            split_status = self.status().split_counts[split]
            if split_status["pending"]:
                raise RuntimeError(
                    f"{split} split remains incomplete after execution "
                    f"({split_status['pending']} pending)."
                )

        def checkpoint_current_training() -> None:
            completed = self.status().split_counts["train"]["completed"]
            lineage_models = [
                model
                for model in self.list_models()
                if model.get("lineage_id") == lineage_id
            ]
            checkpoint_count = max(
                (
                    int(model.get("training_case_count", 0))
                    for model in lineage_models
                ),
                default=0,
            )
            if completed < minimum_cases:
                return
            if checkpoint_count >= completed:
                latest = max(
                    (
                        model
                        for model in lineage_models
                        if int(model.get("training_case_count", 0)) == completed
                    ),
                    key=lambda model: int(model.get("checkpoint_index", 0)),
                    default=None,
                )
                if latest is None:
                    return
                if not latest.get("active"):
                    self.activate_model(str(latest["reproducibility_key"]))
                evaluation_path = (
                    self._reports_path(lock_payload)
                    / "models"
                    / str(latest["reproducibility_key"])
                    / "evaluation-validation.json"
                )
                evaluation = (
                    _read_json(evaluation_path)
                    if evaluation_path.is_file()
                    else self.evaluate(split="validation")
                )
                recorded_keys = {
                    batch["model_reproducibility_key"]
                    for batch in report["batches"]
                }
                if latest["reproducibility_key"] not in recorded_keys:
                    report["batches"].append(
                        {
                            "checkpoint_index": latest["checkpoint_index"],
                            "model_reproducibility_key": latest[
                                "reproducibility_key"
                            ],
                            "training_case_count": latest[
                                "training_case_count"
                            ],
                            "validation_metrics": evaluation["metrics"],
                            "completed_at_utc": _utc_now(),
                        }
                    )
                    self.dashboard()
                    persist()
                return
            emit(f"[checkpoint] entraînement cumulatif sur {completed} cas")
            manifest = self.train(
                allow_partial=True,
                algorithm=algorithm,
                model_overrides=model_overrides,
            )
            evaluation = self.evaluate(split="validation")
            self.dashboard()
            report["batches"].append(
                {
                    "checkpoint_index": manifest["checkpoint_index"],
                    "model_reproducibility_key": manifest["reproducibility_key"],
                    "training_case_count": manifest["training_case_count"],
                    "validation_metrics": evaluation["metrics"],
                    "completed_at_utc": _utc_now(),
                }
            )
            persist()

        try:
            emit("[validation] calcul des références verrouillées")
            self.run(
                split="validation",
                continue_on_error=continue_on_error,
                progress=progress,
            )
            require_completed_split("validation")
            checkpoint_current_training()

            batches_run = 0
            while self.status().split_counts["train"]["pending"]:
                if max_batches is not None and batches_run >= max_batches:
                    break
                emit(
                    f"[lot {batches_run + 1}] jusqu'à {cases_per_batch} références OpenFOAM"
                )
                run_record = self.run(
                    split="train",
                    limit=cases_per_batch,
                    continue_on_error=continue_on_error,
                    progress=progress,
                )
                if not run_record["completed_case_ids"]:
                    raise RuntimeError(
                        "The training batch completed no case; inspect its run record before resuming."
                    )
                batches_run += 1
                checkpoint_current_training()

            training_complete = (
                self.status().split_counts["train"]["pending"] == 0
            )
            if test_at_end:
                if not training_complete:
                    raise RuntimeError(
                        "The test split cannot run before the training split is complete."
                    )
                emit("[test] évaluation finale, sans réentraînement")
                self.run(
                    split="test",
                    continue_on_error=continue_on_error,
                    progress=progress,
                )
                require_completed_split("test")
                report["test_evaluation"] = self.evaluate(split="test")
            self.export_results()
            self.dashboard()
            report["status"] = "completed" if training_complete else "paused"
            persist()
        except Exception:
            report["status"] = "failed"
            report["error"] = traceback.format_exc()
            persist()
            raise
        report["report_path"] = report_path.relative_to(self.root).as_posix()
        return report

    def train(
        self,
        allow_partial: bool = False,
        algorithm: str | None = None,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        lock_payload = self.verify_lock()
        spec = self._read_and_validate_spec()
        training_spec = _with_model_spec(spec, algorithm, model_overrides)
        definitions = self.load_definitions(split="train")
        store = self._store(lock_payload)
        samples_by_id = {sample.case_id: sample for sample in store.load_all()}
        missing = [definition.case_id for definition in definitions if definition.case_id not in samples_by_id]
        if missing and not allow_partial:
            raise RuntimeError(
                f"Training split is incomplete ({len(missing)} pending); run it or pass --allow-partial."
            )
        samples = tuple(
            samples_by_id[definition.case_id]
            for definition in definitions
            if definition.case_id in samples_by_id
        )
        if len(samples) < 2:
            raise RuntimeError("At least two completed training cases are required.")
        lineage_id = _model_lineage_id(training_spec)
        lineage_models = [
            registered_model
            for registered_model in self.list_models()
            if registered_model.get("lineage_id") == lineage_id
            or (
                not registered_model.get("lineage_id")
                and registered_model.get("algorithm")
                == training_spec["model"]["algorithm"]
            )
        ]
        parent_manifest = max(
            lineage_models,
            key=lambda registered_model: int(
                registered_model.get("checkpoint_index", 0)
            ),
            default=None,
        )
        training_case_ids = tuple(sample.case_id for sample in samples)
        parent_training_case_ids = set(
            parent_manifest.get("training_case_ids", ()) if parent_manifest else ()
        )
        if parent_manifest and parent_training_case_ids == set(training_case_ids):
            return self.activate_model(str(parent_manifest["reproducibility_key"]))
        warm_start = (
            training_spec["model"]["algorithm"] == "heat-residual-unet"
            and parent_manifest is not None
        )
        parent_artifact_path = (
            self.root / str(parent_manifest["artifact_path"])
            if warm_start and parent_manifest
            else None
        )
        model = _new_model(training_spec, parent_artifact_path)
        descriptor = model.fit(samples)
        artifact_path = self._model_path(lock_payload, training_spec)
        model.save(artifact_path)
        dataset_manifest = _read_json(store.manifest_path)
        sample_entries = {entry["case_id"]: entry for entry in dataset_manifest["samples"]}
        training_inputs = [
            {"case_id": case_id, "sha256": sample_entries[case_id]["sha256"]}
            for case_id in descriptor.training_case_ids
        ]
        reproducibility_key = _sha256_text(
            _canonical_json(
                {
                    "spec_sha256": lock_payload["spec_sha256"],
                    "cases_sha256": lock_payload["cases_sha256"],
                    "algorithm": training_spec["model"]["algorithm"],
                    "model_parameters": _model_parameters(training_spec),
                    "training_inputs": training_inputs,
                    "parent_reproducibility_key": (
                        parent_manifest["reproducibility_key"] if warm_start else None
                    ),
                }
            )
        )
        artifact_sha256 = _sha256_file(artifact_path)
        artifact_content_sha256 = _model_artifact_content_sha256(artifact_path)
        registry_path = self._model_registry_path(lock_payload) / reproducibility_key
        registry_artifact_path = registry_path / artifact_path.name
        registry_manifest_path = registry_path / "model.manifest.json"
        manifest = {
            "manifest_version": 2,
            "campaign_name": training_spec["campaign_name"],
            "model_id": descriptor.model_id,
            "lineage_id": lineage_id,
            "lineage_name": f"{training_spec['campaign_name']}.{training_spec['domain']}.{training_spec['model']['algorithm']}",
            "checkpoint_index": 1
            + max(
                (
                    int(registered_model.get("checkpoint_index", 0))
                    for registered_model in lineage_models
                ),
                default=0,
            ),
            "parent_reproducibility_key": (
                parent_manifest["reproducibility_key"] if parent_manifest else None
            ),
            "training_mode": (
                "cumulative-warm-start"
                if warm_start
                else "cumulative-cold-start"
                if training_spec["model"]["algorithm"] == "heat-residual-unet"
                else "cumulative-refit"
            ),
            "algorithm": training_spec["model"]["algorithm"],
            "model_parameters": _model_parameters(training_spec),
            "implementation_algorithm": descriptor.metadata["algorithm"],
            "artifact_path": registry_artifact_path.relative_to(self.root).as_posix(),
            "artifact_sha256": artifact_sha256,
            "artifact_content_sha256": artifact_content_sha256,
            "reproducibility_key": reproducibility_key,
            "training_case_ids": list(descriptor.training_case_ids),
            "new_training_case_ids": [
                case_id
                for case_id in descriptor.training_case_ids
                if case_id not in parent_training_case_ids
            ],
            "training_case_count": len(descriptor.training_case_ids),
            "training_inputs": training_inputs,
            "spec_sha256": lock_payload["spec_sha256"],
            "cases_sha256": lock_payload["cases_sha256"],
            "dataset_manifest_sha256": _sha256_file(store.manifest_path),
            "environment": _environment_snapshot(training_spec),
            "trained_at_utc": _utc_now(),
        }
        if registry_manifest_path.is_file():
            registered_manifest = _read_json(registry_manifest_path)
            if (
                registered_manifest.get("artifact_content_sha256")
                != artifact_content_sha256
            ):
                raise CampaignLockError(
                    "A registered model has the same reproducibility key but different content."
                )
            manifest = registered_manifest
        else:
            registry_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact_path, registry_artifact_path)
            _atomic_write_json(registry_manifest_path, manifest)
        _atomic_write_json(self._model_manifest_path(lock_payload), manifest)
        self._write_current_model(lock_payload, manifest)
        return manifest

    def list_models(self) -> tuple[dict[str, Any], ...]:
        lock_payload = self.verify_lock()
        active_key = self._active_model_key(lock_payload)
        registry_path = self._model_registry_path(lock_payload)
        records: list[dict[str, Any]] = []
        if not registry_path.exists():
            return ()
        for manifest_path in sorted(registry_path.glob("*/model.manifest.json")):
            manifest = _read_json(manifest_path)
            artifact_path = self.root / manifest["artifact_path"]
            if not artifact_path.is_file():
                raise CampaignLockError(f"Registered model artifact is missing: {artifact_path}")
            if _model_artifact_content_sha256(artifact_path) != manifest["artifact_content_sha256"]:
                raise CampaignLockError(
                    f"Registered model content does not match: {artifact_path}"
                )
            record = dict(manifest)
            record["active"] = manifest["reproducibility_key"] == active_key
            records.append(record)
        return tuple(records)

    def activate_model(self, reproducibility_key: str) -> Mapping[str, Any]:
        lock_payload = self.verify_lock()
        normalized_key = reproducibility_key.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{8,64}", normalized_key):
            raise ValueError("A model key must be a hexadecimal prefix of at least 8 characters.")
        manifest_paths = [
            path
            for path in self._model_registry_path(lock_payload).glob(
                "*/model.manifest.json"
            )
            if path.parent.name.startswith(normalized_key)
        ]
        if not manifest_paths:
            raise ValueError(f"Unknown model reproducibility key: {reproducibility_key}")
        if len(manifest_paths) > 1:
            raise ValueError("The model key prefix is ambiguous; provide more characters.")
        manifest_path = manifest_paths[0]
        manifest = _read_json(manifest_path)
        artifact_path = self.root / manifest["artifact_path"]
        if not artifact_path.is_file():
            raise CampaignLockError("The selected model artifact is missing.")
        if _model_artifact_content_sha256(artifact_path) != manifest["artifact_content_sha256"]:
            raise CampaignLockError("The selected model artifact does not match its manifest.")
        _atomic_write_json(self._model_manifest_path(lock_payload), manifest)
        self._write_current_model(lock_payload, manifest)
        return manifest

    def evaluate(self, split: str = "validation", allow_partial: bool = False) -> Mapping[str, Any]:
        if split not in ("validation", "test"):
            raise ValueError("Evaluation split must be validation or test.")
        lock_payload = self.verify_lock()
        spec = self._read_and_validate_spec()
        model_manifest_path = self._model_manifest_path(lock_payload)
        if not model_manifest_path.is_file():
            raise RuntimeError("No trained model manifest found. Run the train command first.")
        model_manifest = _read_json(model_manifest_path)
        artifact_path = self.root / model_manifest["artifact_path"]
        if _model_artifact_content_sha256(artifact_path) != model_manifest["artifact_content_sha256"]:
            raise CampaignLockError("Model artifact content does not match its manifest.")
        model = _load_model(model_manifest["algorithm"], artifact_path)

        definitions = self.load_definitions(split=split)
        store = self._store(lock_payload)
        samples_by_id = {sample.case_id: sample for sample in store.load_all()}
        missing = [definition.case_id for definition in definitions if definition.case_id not in samples_by_id]
        if missing and not allow_partial:
            raise RuntimeError(
                f"{split} split is incomplete ({len(missing)} pending); run it or pass --allow-partial."
            )
        rows: list[dict[str, Any]] = []
        for definition in definitions:
            sample = samples_by_id.get(definition.case_id)
            if sample is None:
                continue
            case = definition.to_case()
            inference_started = perf_counter()
            prediction = model.predict(case, sample.coarse_on_nominal)
            inference_seconds = max(perf_counter() - inference_started, 1e-12)
            preview_metrics = compare_fields(prediction.mean, sample.nominal.field)
            coarse_metrics = compare_fields(sample.coarse_on_nominal, sample.nominal.field)
            gradient_metrics = compare_gradients(
                prediction.mean,
                sample.nominal.field,
                case.problem.extent,
            )
            principle_metrics = maximum_principle_violation(
                prediction.mean,
                case.initial_field,
                case.boundaries,
            )
            absolute_error = np.abs(prediction.mean - sample.nominal.field)
            preview_runtime_seconds = sample.coarse.runtime_seconds + inference_seconds
            rows.append(
                {
                    "case_id": definition.case_id,
                    "family": definition.family,
                    "preview_relative_l2": preview_metrics["relative_l2"],
                    "coarse_relative_l2": coarse_metrics["relative_l2"],
                    "error_ratio_vs_coarse": preview_metrics["relative_l2"]
                    / max(coarse_metrics["relative_l2"], 1e-15),
                    "preview_mae": preview_metrics["mae"],
                    "preview_rmse": preview_metrics["rmse"],
                    "preview_max_abs_error": preview_metrics["max_abs_error"],
                    "gradient_relative_l2": gradient_metrics["gradient_relative_l2"],
                    "gradient_rmse": gradient_metrics["gradient_rmse"],
                    "relative_gain_vs_coarse": 1.0
                    - preview_metrics["relative_l2"]
                    / max(coarse_metrics["relative_l2"], 1e-15),
                    "boundary_residual": boundary_residual(
                        prediction.mean, case.boundaries
                    ),
                    **principle_metrics,
                    "coverage_1sigma": float(
                        np.mean(absolute_error <= prediction.uncertainty)
                    ),
                    "coverage_2sigma": float(
                        np.mean(absolute_error <= 2.0 * prediction.uncertainty)
                    ),
                    "mean_uncertainty": float(np.mean(prediction.uncertainty)),
                    "ood_score": prediction.ood_score,
                    "inference_seconds": inference_seconds,
                    "preview_runtime_seconds": preview_runtime_seconds,
                    "nominal_runtime_seconds": sample.nominal.runtime_seconds,
                    "preview_speedup": sample.nominal.runtime_seconds
                    / preview_runtime_seconds,
                }
            )
        if not rows:
            raise RuntimeError(f"No completed cases are available in the {split} split.")
        metrics = {
            "mean_preview_relative_l2": float(
                np.mean([row["preview_relative_l2"] for row in rows])
            ),
            "mean_coarse_relative_l2": float(
                np.mean([row["coarse_relative_l2"] for row in rows])
            ),
            "worst_preview_relative_l2": float(
                np.max([row["preview_relative_l2"] for row in rows])
            ),
            "worst_error_ratio_vs_coarse": float(
                np.max([row["error_ratio_vs_coarse"] for row in rows])
            ),
            "minimum_coverage_2sigma": float(
                np.min([row["coverage_2sigma"] for row in rows])
            ),
            "median_preview_relative_l2": _percentile(
                rows, "preview_relative_l2", 50
            ),
            "p95_preview_relative_l2": _percentile(
                rows, "preview_relative_l2", 95
            ),
            "mean_gradient_relative_l2": _mean(rows, "gradient_relative_l2"),
            "worst_gradient_relative_l2": _maximum(
                rows, "gradient_relative_l2"
            ),
            "mean_relative_gain_vs_coarse": _mean(
                rows, "relative_gain_vs_coarse"
            ),
            "minimum_relative_gain_vs_coarse": _minimum(
                rows, "relative_gain_vs_coarse"
            ),
            "mean_coverage_1sigma": _mean(rows, "coverage_1sigma"),
            "mean_coverage_2sigma": _mean(rows, "coverage_2sigma"),
            "mean_uncertainty": _mean(rows, "mean_uncertainty"),
            "mean_ood_score": _mean(rows, "ood_score"),
            "worst_boundary_residual": _maximum(rows, "boundary_residual"),
            "worst_maximum_principle_relative_violation": _maximum(
                rows, "maximum_principle_relative_violation"
            ),
            "median_inference_seconds": _percentile(
                rows, "inference_seconds", 50
            ),
            "p95_inference_seconds": _percentile(rows, "inference_seconds", 95),
            "median_preview_speedup": _percentile(rows, "preview_speedup", 50),
        }
        quality = spec["quality"]
        passed = (
            metrics["worst_preview_relative_l2"] <= quality["max_relative_l2"]
            and metrics["worst_error_ratio_vs_coarse"]
            <= quality["max_error_ratio_vs_coarse"]
            and metrics["minimum_coverage_2sigma"] >= quality["min_coverage_2sigma"]
        )
        report = {
            "report_version": 1,
            "campaign_name": spec["campaign_name"],
            "split": split,
            "passed": passed,
            "quality_policy": quality,
            "metrics": metrics,
            "cases": rows,
            "metrics_by_family": _metrics_by_family(rows),
            "model_reproducibility_key": model_manifest["reproducibility_key"],
            "model_lineage_id": model_manifest.get("lineage_id"),
            "model_checkpoint_index": model_manifest.get("checkpoint_index"),
            "parent_reproducibility_key": model_manifest.get(
                "parent_reproducibility_key"
            ),
            "model_artifact_content_sha256": model_manifest["artifact_content_sha256"],
            "spec_sha256": lock_payload["spec_sha256"],
            "cases_sha256": lock_payload["cases_sha256"],
            "dataset_manifest_sha256": _sha256_file(store.manifest_path),
            "evaluated_at_utc": _utc_now(),
            "environment": _environment_snapshot(spec),
        }
        report["checkpoint_comparison"] = self._checkpoint_comparison(
            lock_payload,
            model_manifest,
            split,
            rows,
        )
        reports_path = self._reports_path(lock_payload)
        report_path = reports_path / f"evaluation-{split}.json"
        versioned_report_path = (
            reports_path
            / "models"
            / model_manifest["reproducibility_key"]
            / f"evaluation-{split}.json"
        )
        _atomic_write_json(versioned_report_path, report)
        _atomic_write_json(report_path, report)
        return report

    def _checkpoint_comparison(
        self,
        lock_payload: Mapping[str, Any],
        model_manifest: Mapping[str, Any],
        split: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | None:
        parent_key = model_manifest.get("parent_reproducibility_key")
        if not parent_key:
            return None
        parent_report_path = (
            self._reports_path(lock_payload)
            / "models"
            / str(parent_key)
            / f"evaluation-{split}.json"
        )
        if not parent_report_path.is_file():
            return None
        parent_report = _read_json(parent_report_path)
        parent_rows = {row["case_id"]: row for row in parent_report["cases"]}
        paired = [
            (
                float(row["preview_relative_l2"]),
                float(parent_rows[row["case_id"]]["preview_relative_l2"]),
            )
            for row in rows
            if row["case_id"] in parent_rows
        ]
        if not paired:
            return None
        deltas = np.asarray([current - parent for current, parent in paired])
        return {
            "parent_reproducibility_key": parent_key,
            "paired_case_count": len(paired),
            "mean_relative_l2_delta": float(np.mean(deltas)),
            "worst_case_regression": float(np.max(deltas)),
            "improved_case_fraction": float(np.mean(deltas < 0.0)),
        }

    def export_results(self) -> Mapping[str, Any]:
        from shardsim.campaign.reporting import export_campaign_results

        lock_payload = self.verify_lock()
        spec = self._read_and_validate_spec()
        definitions = self.load_definitions()
        store = self._store(lock_payload)
        return export_campaign_results(
            root=self.root,
            reports_path=self._reports_path(lock_payload),
            spec=spec,
            lock_payload=lock_payload,
            definitions=definitions,
            samples=store.load_all(),
            dataset_manifest_path=store.manifest_path,
        )

    def dashboard(self, open_browser: bool = False) -> Path:
        from shardsim.campaign.reporting import render_campaign_dashboard

        lock_payload = self.verify_lock()
        export = self.export_results()
        reports_path = self._reports_path(lock_payload)
        run_records = [
            _read_json(path)
            for path in sorted(self._runs_path(lock_payload).glob("*.json"))
        ]
        dashboard_path = render_campaign_dashboard(
            reports_path=reports_path,
            campaign_status=self.status().to_payload(),
            samples=self._store(lock_payload).load_all(),
            definitions=self.load_definitions(),
            models=self.list_models(),
            run_records=run_records,
            export=export,
        )
        if open_browser:
            import webbrowser

            webbrowser.open(dashboard_path.resolve().as_uri())
        return dashboard_path

    def status(self) -> CampaignStatus:
        lock_payload = self.verify_lock()
        spec = self._read_and_validate_spec()
        definitions = self.load_definitions()
        store = self._store(lock_payload)
        completed = set(store.case_ids())
        split_counts: dict[str, dict[str, int]] = {}
        for split in VALID_SPLITS:
            split_definitions = [definition for definition in definitions if definition.split == split]
            completed_count = sum(definition.case_id in completed for definition in split_definitions)
            split_counts[split] = {
                "total": len(split_definitions),
                "completed": completed_count,
                "pending": len(split_definitions) - completed_count,
            }
        reports_path = self._reports_path(lock_payload)
        evaluations = tuple(
            sorted(path.stem.removeprefix("evaluation-") for path in reports_path.glob("evaluation-*.json"))
        ) if reports_path.exists() else ()
        models = self.list_models()
        active_model_key = next(
            (model["reproducibility_key"] for model in models if model["active"]),
            None,
        )
        return CampaignStatus(
            campaign_name=spec["campaign_name"],
            spec_sha256=lock_payload["spec_sha256"],
            cases_sha256=lock_payload["cases_sha256"],
            output_key=self._output_key(lock_payload),
            split_counts=split_counts,
            total_cases=len(definitions),
            completed_cases=len(completed),
            pending_cases=len(definitions) - len(completed),
            model_exists=self._model_manifest_path(lock_payload).is_file(),
            model_count=len(models),
            active_model_key=active_model_key,
            evaluations=evaluations,
        )

    def _read_and_validate_spec(self) -> dict[str, Any]:
        if not self.spec_path.is_file():
            raise FileNotFoundError(f"Campaign specification not found: {self.spec_path}")
        spec = _read_json(self.spec_path)
        if spec.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError("Unsupported campaign schema version.")
        if spec.get("domain") != "heat-2d" or spec.get("equation") != "du/dt=alpha*laplacian(u)":
            raise ValueError("This campaign runner currently supports the heat-2d equation only.")
        if not isinstance(spec.get("seed"), int):
            raise ValueError("Campaign seed must be an integer.")
        plan = self._plan(spec)
        if plan.coarse_shape == plan.nominal_shape:
            raise ValueError("Campaign fidelity levels must differ.")
        solvers = spec.get("solvers", {})
        if solvers.get("coarse") != "internal":
            raise ValueError("The current campaign coarse solver must be internal.")
        if solvers.get("nominal") not in ("internal", "openfoam"):
            raise ValueError("Nominal solver must be internal or openfoam.")
        if solvers.get("nominal") == "openfoam" and not solvers.get("openfoam_image"):
            raise ValueError("OpenFOAM campaigns require a pinned image reference.")
        design = spec.get("design", {})
        if len(design.get("extent", ())) != 2 or len(design.get("initial_shape", ())) != 2:
            raise ValueError("Campaign extent and initial_shape must be two-dimensional.")
        families = design.get("families")
        if not isinstance(families, Mapping) or not families:
            raise ValueError("Campaign design requires at least one family.")
        for family_name, family in families.items():
            if not str(family_name).strip() or family.get("split") not in VALID_SPLITS:
                raise ValueError("Every campaign family requires a valid name and split.")
            if not isinstance(family.get("count"), int) or family["count"] < 1:
                raise ValueError(f"Family {family_name!r} requires a positive count.")
            for key in _RANGE_KEYS:
                _validate_range(family.get(key), f"{family_name}.{key}")
        quality = spec.get("quality", {})
        if not 0 <= quality.get("min_coverage_2sigma", -1) <= 1:
            raise ValueError("Quality coverage threshold must lie in [0, 1].")
        if quality.get("max_relative_l2", 0) <= 0 or quality.get(
            "max_error_ratio_vs_coarse", 0
        ) <= 0:
            raise ValueError("Quality error thresholds must be positive.")
        if spec.get("model", {}).get("algorithm") not in (
            "heat-local-residual",
            "mean-delta",
            "heat-residual-unet",
        ):
            raise ValueError(
                "Model algorithm must be heat-local-residual, mean-delta, or heat-residual-unet."
            )
        return spec

    def _generate_definitions(self, spec: Mapping[str, Any]) -> tuple[HeatCaseDefinition, ...]:
        design = spec["design"]
        boundaries = design["boundaries"]
        definitions: list[HeatCaseDefinition] = []
        for family_name in sorted(design["families"]):
            family = design["families"][family_name]
            rng = np.random.Generator(
                np.random.PCG64(_derived_seed(int(spec["seed"]), family_name))
            )
            unit_design = _latin_hypercube(rng, family["count"], len(_RANGE_KEYS))
            for index in range(family["count"]):
                values = {
                    key: _scale_unit(unit_design[index, column], family[key])
                    for column, key in enumerate(_RANGE_KEYS)
                }
                definition_payload = {
                    "family": family_name,
                    "split": family["split"],
                    "alpha": values["alpha_range"],
                    "t_end": values["t_end_range"],
                    "extent": [float(value) for value in design["extent"]],
                    "center": [
                        values["center_x_range"],
                        values["center_y_range"],
                    ],
                    "sigma": [
                        values["sigma_x_range"],
                        values["sigma_y_range"],
                    ],
                    "amplitude": values["amplitude_range"],
                    "baseline": values["baseline_range"],
                    "boundaries": {
                        key: float(boundaries[key])
                        for key in ("top", "bottom", "left", "right")
                    },
                    "initial_shape": [int(value) for value in design["initial_shape"]],
                }
                definition_hash = _sha256_text(_canonical_json(definition_payload))
                payload = {
                    "case_id": f"{_slug(family_name)}-{index:04d}-{definition_hash[:10]}",
                    **definition_payload,
                    "definition_sha256": definition_hash,
                }
                definitions.append(HeatCaseDefinition.from_payload(payload))
        return tuple(definitions)

    def _plan(self, spec: Mapping[str, Any]) -> FidelityPlan:
        fidelity = spec["fidelity"]
        return FidelityPlan(
            coarse_shape=tuple(int(value) for value in fidelity["coarse_shape"]),
            nominal_shape=tuple(int(value) for value in fidelity["nominal_shape"]),
        )

    def _solvers(self, spec: Mapping[str, Any]) -> tuple[Solver, Solver]:
        coarse_solver: Solver = HeatEquationSolver()
        if spec["solvers"]["nominal"] == "internal":
            nominal_solver: Solver = HeatEquationSolver()
        else:
            nominal_solver = OpenFOAMAdapter(image=spec["solvers"]["openfoam_image"])
        return coarse_solver, nominal_solver

    def _store(self, lock_payload: Mapping[str, Any]) -> ReferenceDatasetStore:
        return ReferenceDatasetStore(
            self._output_path(lock_payload) / "dataset",
            dataset_id=f"{lock_payload['campaign_name']}-{self._output_key(lock_payload)}",
        )

    def _output_key(self, lock_payload: Mapping[str, Any]) -> str:
        return str(lock_payload["spec_sha256"])[:12]

    def _output_path(self, lock_payload: Mapping[str, Any]) -> Path:
        return self.root / "outputs" / self._output_key(lock_payload)

    def _runs_path(self, lock_payload: Mapping[str, Any]) -> Path:
        return self._output_path(lock_payload) / "runs"

    def _reports_path(self, lock_payload: Mapping[str, Any]) -> Path:
        return self._output_path(lock_payload) / "reports"

    def _model_path(self, lock_payload: Mapping[str, Any], spec: Mapping[str, Any]) -> Path:
        return self._output_path(lock_payload) / "models" / spec["model"]["artifact_name"]

    def _model_manifest_path(self, lock_payload: Mapping[str, Any]) -> Path:
        return self._output_path(lock_payload) / "models" / "model.manifest.json"

    def _model_registry_path(self, lock_payload: Mapping[str, Any]) -> Path:
        return self._output_path(lock_payload) / "models" / "registry"

    def _current_model_path(self, lock_payload: Mapping[str, Any]) -> Path:
        return self._output_path(lock_payload) / "models" / "current.json"

    def _active_model_key(self, lock_payload: Mapping[str, Any]) -> str | None:
        current_path = self._current_model_path(lock_payload)
        if current_path.is_file():
            return str(_read_json(current_path)["reproducibility_key"])
        manifest_path = self._model_manifest_path(lock_payload)
        if manifest_path.is_file():
            return str(_read_json(manifest_path)["reproducibility_key"])
        return None

    def _write_current_model(
        self,
        lock_payload: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        manifest_path = (
            self._model_registry_path(lock_payload)
            / str(manifest["reproducibility_key"])
            / "model.manifest.json"
        )
        _atomic_write_json(
            self._current_model_path(lock_payload),
            {
                "reproducibility_key": manifest["reproducibility_key"],
                "manifest_path": manifest_path.relative_to(self.root).as_posix(),
                "activated_at_utc": _utc_now(),
            },
        )

    def _new_run_record(
        self,
        lock_payload: Mapping[str, Any],
        split: str | None,
        family: str | None,
        case_ids: Sequence[str] | None,
        selected: Sequence[HeatCaseDefinition],
        pending: Sequence[HeatCaseDefinition],
    ) -> dict[str, Any]:
        started_at = _utc_now()
        run_id = f"{started_at.replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:8]}"
        spec = self._read_and_validate_spec()
        return {
            "run_record_version": 1,
            "run_id": run_id,
            "status": "running",
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "selection": {
                "split": split,
                "family": family,
                "case_ids": list(case_ids or ()),
            },
            "selected_case_ids": [definition.case_id for definition in selected],
            "pending_at_start_case_ids": [definition.case_id for definition in pending],
            "completed_case_ids": [],
            "failures": [],
            "spec_sha256": lock_payload["spec_sha256"],
            "cases_sha256": lock_payload["cases_sha256"],
            "environment": _environment_snapshot(spec),
        }


_RANGE_KEYS = (
    "alpha_range",
    "t_end_range",
    "center_x_range",
    "center_y_range",
    "sigma_x_range",
    "sigma_y_range",
    "amplitude_range",
    "baseline_range",
)


def _validate_range(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value JSON array.")
    lower, upper = (float(item) for item in value)
    if not np.isfinite((lower, upper)).all() or lower > upper:
        raise ValueError(f"{name} must contain a finite ordered range.")


def _scale_unit(unit_value: float, bounds: Sequence[float]) -> float:
    lower, upper = (float(value) for value in bounds)
    if lower == upper:
        return lower
    return float(lower + unit_value * (upper - lower))


def _latin_hypercube(
    rng: np.random.Generator,
    count: int,
    dimensions: int,
) -> np.ndarray:
    design = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        design[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return design


def _new_model(
    spec: Mapping[str, Any],
    parent_artifact_path: Path | None = None,
) -> Any:
    model_spec = spec["model"]
    if model_spec["algorithm"] == "heat-local-residual":
        return HeatLocalResidualSurrogate(
            model_id=f"{spec['campaign_name']}.heat-local-residual.v1",
            ridge=float(model_spec.get("ridge", 0.03)),
            uncertainty_ood_scale=float(model_spec.get("uncertainty_ood_scale", 0.25)),
        )
    if model_spec["algorithm"] == "mean-delta":
        return MeanDeltaSurrogate(model_id=f"{spec['campaign_name']}.mean-delta.v1")
    from shardsim.surrogates.heat_unet import HeatResidualUNetSurrogate

    if parent_artifact_path is not None:
        return HeatResidualUNetSurrogate.load(parent_artifact_path)
    return HeatResidualUNetSurrogate(
        model_id=f"{spec['campaign_name']}.heat-residual-unet.v1",
        width=int(model_spec["width"]),
        epochs=int(model_spec["epochs"]),
        batch_size=int(model_spec["batch_size"]),
        learning_rate=float(model_spec["learning_rate"]),
        weight_decay=float(model_spec["weight_decay"]),
        gradient_weight=float(model_spec["gradient_weight"]),
        uncertainty_ood_scale=float(model_spec["uncertainty_ood_scale"]),
        seed=int(model_spec["seed"]),
        device=str(model_spec["device"]),
        deterministic=bool(model_spec["deterministic"]),
    )


def _load_model(
    algorithm: str,
    path: Path,
) -> Any:
    if algorithm == "heat-local-residual":
        return HeatLocalResidualSurrogate.load(path)
    if algorithm == "mean-delta":
        return MeanDeltaSurrogate.load(path)
    if algorithm == "heat-residual-unet":
        from shardsim.surrogates.heat_unet import HeatResidualUNetSurrogate

        return HeatResidualUNetSurrogate.load(path)
    raise ValueError(f"Unsupported model algorithm: {algorithm}")


def _model_parameters(spec: Mapping[str, Any]) -> dict[str, Any]:
    model_spec = spec["model"]
    if model_spec["algorithm"] == "heat-local-residual":
        return {
            "ridge": float(model_spec.get("ridge", 0.03)),
            "uncertainty_ood_scale": float(model_spec.get("uncertainty_ood_scale", 0.25)),
        }
    if model_spec["algorithm"] == "mean-delta":
        return {}
    return {
        "width": int(model_spec["width"]),
        "epochs": int(model_spec["epochs"]),
        "batch_size": int(model_spec["batch_size"]),
        "learning_rate": float(model_spec["learning_rate"]),
        "weight_decay": float(model_spec["weight_decay"]),
        "gradient_weight": float(model_spec["gradient_weight"]),
        "uncertainty_ood_scale": float(model_spec["uncertainty_ood_scale"]),
        "seed": int(model_spec["seed"]),
        "device": str(model_spec["device"]),
        "deterministic": bool(model_spec["deterministic"]),
    }


def _with_model_spec(
    spec: Mapping[str, Any],
    algorithm: str | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected_algorithm = algorithm or str(spec["model"]["algorithm"])
    if selected_algorithm not in (
        "heat-local-residual",
        "mean-delta",
        "heat-residual-unet",
    ):
        raise ValueError(f"Unsupported model algorithm: {selected_algorithm}")
    selected_overrides = {
        key: value for key, value in (overrides or {}).items() if value is not None
    }
    if selected_algorithm != "heat-residual-unet" and selected_overrides:
        raise ValueError("CNN training overrides require --algorithm heat-residual-unet.")
    if selected_algorithm == "heat-residual-unet":
        model_spec = {
            "algorithm": selected_algorithm,
            "artifact_name": "heat-residual-unet.pt",
            "width": 8,
            "epochs": 200,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 1e-6,
            "gradient_weight": 0.05,
            "uncertainty_ood_scale": 0.25,
            "seed": int(spec["seed"]),
            "device": "cpu",
            "deterministic": True,
        }
        if selected_algorithm == str(spec["model"]["algorithm"]):
            model_spec.update(spec["model"])
        model_spec["artifact_name"] = "heat-residual-unet.pt"
    elif selected_algorithm == str(spec["model"]["algorithm"]):
        model_spec = dict(spec["model"])
    elif selected_algorithm == "heat-local-residual":
        model_spec = {
            "algorithm": selected_algorithm,
            "artifact_name": "heat-local-residual.npz",
            "ridge": 0.03,
            "uncertainty_ood_scale": 0.25,
        }
    elif selected_algorithm == "mean-delta":
        model_spec = {
            "algorithm": selected_algorithm,
            "artifact_name": "mean-delta.npz",
        }
    model_spec.update(selected_overrides)
    if selected_algorithm == "heat-residual-unet":
        if int(model_spec["width"]) < 4:
            raise ValueError("CNN width must be at least 4.")
        if int(model_spec["epochs"]) < 1 or int(model_spec["batch_size"]) < 1:
            raise ValueError("CNN epochs and batch size must be positive.")
        if str(model_spec["device"]) not in ("cpu", "cuda"):
            raise ValueError("CNN device must be cpu or cuda.")
    effective = dict(spec)
    effective["model"] = model_spec
    return effective


def _model_lineage_id(spec: Mapping[str, Any]) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "campaign_name": spec["campaign_name"],
                "domain": spec["domain"],
                "equation": spec["equation"],
                "fidelity": spec["fidelity"],
                "algorithm": spec["model"]["algorithm"],
                "model_parameters": _model_parameters(spec),
            }
        )
    )


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _minimum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.min([float(row[key]) for row in rows]))


def _maximum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.max([float(row[key]) for row in rows]))


def _percentile(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    percentile: float,
) -> float:
    return float(np.percentile([float(row[key]) for row in rows], percentile))


def _metrics_by_family(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, float | int]]:
    families = sorted({str(row["family"]) for row in rows})
    return {
        family: {
            "case_count": len(family_rows),
            "mean_preview_relative_l2": _mean(
                family_rows, "preview_relative_l2"
            ),
            "worst_preview_relative_l2": _maximum(
                family_rows, "preview_relative_l2"
            ),
            "mean_relative_gain_vs_coarse": _mean(
                family_rows, "relative_gain_vs_coarse"
            ),
            "mean_gradient_relative_l2": _mean(
                family_rows, "gradient_relative_l2"
            ),
            "minimum_coverage_2sigma": _minimum(
                family_rows, "coverage_2sigma"
            ),
        }
        for family in families
        if (
            family_rows := [row for row in rows if str(row["family"]) == family]
        )
    }


def _derived_seed(seed: int, family: str) -> int:
    digest = hashlib.sha256(f"{seed}:{family}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("A campaign family name must contain letters or digits.")
    return slug


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = np.asarray(archive[name])
            digest.update(name.encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(_canonical_json(list(array.shape)).encode("utf-8"))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_artifact_content_sha256(path: Path) -> str:
    if path.suffix == ".npz":
        return _npz_content_sha256(path)
    if path.suffix != ".pt":
        raise ValueError(f"Unsupported model artifact extension: {path.suffix}")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to verify CNN artifacts.") from error
    payload = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    _update_content_digest(digest, payload)
    return digest.hexdigest()


def _update_content_digest(digest: Any, value: Any) -> None:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_content_digest(digest, key)
            _update_content_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _update_content_digest(digest, item)
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(_canonical_json(value).encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _environment_snapshot(spec: Mapping[str, Any]) -> dict[str, Any]:
    try:
        package_version = importlib.metadata.version("shardsim")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-tree"
    git_commit = _git_output(("rev-parse", "HEAD"))
    git_status = _git_output(("status", "--porcelain"))
    environment = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "shardsim_version": package_version,
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status_sha256": _sha256_text(git_status) if git_status is not None else None,
        "nominal_solver": spec["solvers"]["nominal"],
        "openfoam_image": spec["solvers"].get("openfoam_image"),
    }
    if spec.get("model", {}).get("algorithm") == "heat-residual-unet":
        try:
            import torch

            environment.update(
                {
                    "torch_version": torch.__version__,
                    "torch_cuda_version": torch.version.cuda,
                    "cnn_device": spec["model"]["device"],
                    "deterministic_algorithms": spec["model"]["deterministic"],
                }
            )
        except ImportError:
            environment["torch_version"] = None
    return environment


def _git_output(arguments: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
