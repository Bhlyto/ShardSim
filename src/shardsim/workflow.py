from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from shardsim.active_learning import (
    ActiveLearningIteration,
    ActiveLearningLoop,
    ActiveLearningPolicy,
)
from shardsim.canonical import FieldLocation
from shardsim.contracts import Fidelity, ModelDescriptor, SimulationCase, SimulationResult
from shardsim.dataset import ReferenceDatasetStore
from shardsim.pipeline import BootstrapPipeline, FidelityPlan, ReferenceSample
from shardsim.preview import PreviewPipeline, PreviewPolicy, PreviewResult, PreviewValidation
from shardsim.solvers.base import Solver
from shardsim.surrogates.base import PersistentDeltaSurrogate
from shardsim.surrogates.mean_delta import MeanDeltaSurrogate


class VariableRole(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class VariableDefinition:
    name: str
    role: VariableRole
    data_type: str
    unit: str | None
    shape: tuple[int, ...]
    location: FieldLocation | None = None
    summary: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ProblemAnalysis:
    case_id: str
    domain: str
    equation: str
    schema_version: str
    inputs: tuple[VariableDefinition, ...]
    outputs: tuple[VariableDefinition, ...]
    coarse_shape: tuple[int, int]
    nominal_shape: tuple[int, int]
    coarse_solver_id: str
    nominal_solver_id: str
    case_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class WorkflowBootstrapResult:
    analyses: tuple[ProblemAnalysis, ...]
    new_references: tuple[ReferenceSample, ...]
    model: ModelDescriptor
    total_reference_count: int
    model_artifact: Path | None


@dataclass(frozen=True, slots=True)
class WorkflowPreviewResult:
    analysis: ProblemAnalysis
    preview: PreviewResult
    accepted_by_policy: bool
    model_promoted: bool
    validation: PreviewValidation | None

    @property
    def field(self) -> np.ndarray:
        if self.accepted_by_policy:
            return self.preview.prediction.mean
        return self.preview.coarse_on_nominal


@dataclass(frozen=True, slots=True)
class WorkflowIterationResult:
    analyses: tuple[ProblemAnalysis, ...]
    active_learning: ActiveLearningIteration
    model_artifact: Path | None


class CampaignStopReason(str, Enum):
    QUALITY_PASSED = "quality-passed"
    CANDIDATES_EXHAUSTED = "candidates-exhausted"
    MAX_ITERATIONS = "max-iterations"


@dataclass(frozen=True, slots=True)
class LearningCampaignPolicy:
    selection_count: int = 1
    max_iterations: int = 5
    minimum_iterations: int = 1

    def __post_init__(self) -> None:
        if self.selection_count < 1 or self.max_iterations < 1:
            raise ValueError("Campaign selection count and maximum iterations must be positive.")
        if self.minimum_iterations < 0 or self.minimum_iterations > self.max_iterations:
            raise ValueError("minimum_iterations must lie between zero and max_iterations.")


@dataclass(frozen=True, slots=True)
class CampaignRound:
    index: int
    iteration: WorkflowIterationResult
    evaluation: ModelEvaluation


@dataclass(frozen=True, slots=True)
class LearningCampaignResult:
    initial_evaluation: ModelEvaluation
    rounds: tuple[CampaignRound, ...]
    final_evaluation: ModelEvaluation
    stop_reason: CampaignStopReason
    total_reference_count: int


@dataclass(frozen=True, slots=True)
class ModelQualityPolicy:
    max_relative_l2: float = 0.20
    max_error_ratio_vs_coarse: float = 1.0
    min_coverage_2sigma: float = 0.50

    def __post_init__(self) -> None:
        values = (
            self.max_relative_l2,
            self.max_error_ratio_vs_coarse,
            self.min_coverage_2sigma,
        )
        if any(not isfinite(float(value)) or value < 0 for value in values):
            raise ValueError("Model quality thresholds must be finite and non-negative.")
        if self.min_coverage_2sigma > 1:
            raise ValueError("Coverage threshold cannot exceed one.")

    def accepts(self, metrics: Mapping[str, float]) -> bool:
        return (
            metrics["worst_preview_relative_l2"] <= self.max_relative_l2
            and metrics["worst_error_ratio_vs_coarse"] <= self.max_error_ratio_vs_coarse
            and metrics["minimum_coverage_2sigma"] >= self.min_coverage_2sigma
        )


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    analyses: tuple[ProblemAnalysis, ...]
    validations: tuple[PreviewValidation, ...]
    metrics: Mapping[str, float]
    passed: bool
    report_path: Path | None


class SimulationLearningWorkflow:
    """End-to-end nominal-simulation, preview, validation, and retraining loop."""

    def __init__(
        self,
        coarse_solver: Solver,
        nominal_solver: Solver,
        plan: FidelityPlan,
        store: ReferenceDatasetStore | None = None,
        model_artifact: str | Path | None = None,
        surrogate: PersistentDeltaSurrogate | None = None,
        active_learning_policy: ActiveLearningPolicy | None = None,
        preview_policy: PreviewPolicy | None = None,
        model_quality_policy: ModelQualityPolicy | None = None,
    ) -> None:
        self.coarse_solver = coarse_solver
        self.nominal_solver = nominal_solver
        self.plan = plan
        self.store = store
        self.model_artifact = Path(model_artifact) if model_artifact is not None else None
        self.surrogate = surrogate or MeanDeltaSurrogate()
        self.active_learning_policy = active_learning_policy or ActiveLearningPolicy()
        self.preview_policy = preview_policy or PreviewPolicy()
        self.model_quality_policy = model_quality_policy or ModelQualityPolicy()
        self.bootstrap_pipeline = BootstrapPipeline(
            coarse_solver,
            plan,
            nominal_solver=nominal_solver,
        )
        self.preview_pipeline = PreviewPipeline(
            coarse_solver,
            self.surrogate,
            plan,
            nominal_solver=nominal_solver,
        )
        self._references = list(store.load_all()) if store is not None else []
        self._active_loop: ActiveLearningLoop | None = None
        self._model_promoted = False
        self._evaluation_cache: dict[
            str,
            tuple[SimulationCase, SimulationResult, SimulationResult],
        ] = {}
        if len(self._references) >= 2:
            self._initialize_active_loop()

    @property
    def references(self) -> tuple[ReferenceSample, ...]:
        if self._active_loop is not None:
            return self._active_loop.references
        return tuple(self._references)

    def analyze(self, case: SimulationCase) -> ProblemAnalysis:
        supports = getattr(self.nominal_solver, "supports", None)
        if callable(supports) and not supports(case):
            raise ValueError(f"The nominal solver does not support case {case.case_id!r}.")

        output_location = _solver_output_location(self.nominal_solver)
        output_unit = next(iter(case.problem.output_units.values()), None)
        inputs: list[VariableDefinition] = []
        for name in sorted(case.problem.parameters):
            inputs.append(
                VariableDefinition(
                    name=name,
                    role=VariableRole.INPUT,
                    data_type="scalar",
                    unit=case.problem.input_units.get(name),
                    shape=(),
                    summary={"value": case.problem.parameter(name)},
                )
            )
        inputs.extend(
            [
                VariableDefinition(
                    name="t_end",
                    role=VariableRole.INPUT,
                    data_type="scalar",
                    unit=case.problem.input_units.get("time"),
                    shape=(),
                    summary={"value": case.problem.t_end},
                ),
                VariableDefinition(
                    name="extent",
                    role=VariableRole.INPUT,
                    data_type="vector",
                    unit=case.problem.input_units.get("x"),
                    shape=(2,),
                    summary={"values": case.problem.extent},
                ),
                VariableDefinition(
                    name="initial_field",
                    role=VariableRole.INPUT,
                    data_type="scalar_field",
                    unit=output_unit,
                    shape=case.initial_field.shape,
                    location=FieldLocation.POINT,
                    summary={
                        "minimum": float(np.min(case.initial_field)),
                        "maximum": float(np.max(case.initial_field)),
                        "mean": float(np.mean(case.initial_field)),
                    },
                ),
                VariableDefinition(
                    name="dirichlet_boundaries",
                    role=VariableRole.INPUT,
                    data_type="boundary_values",
                    unit=output_unit,
                    shape=(4,),
                    summary={
                        "top": case.boundaries.top,
                        "bottom": case.boundaries.bottom,
                        "left": case.boundaries.left,
                        "right": case.boundaries.right,
                    },
                ),
            ]
        )
        outputs = tuple(
            VariableDefinition(
                name=name,
                role=VariableRole.OUTPUT,
                data_type="scalar_field",
                unit=unit,
                shape=self.plan.nominal_shape,
                location=output_location,
                summary={"physical_time": case.problem.t_end},
            )
            for name, unit in sorted(case.problem.output_units.items())
        )
        if not outputs:
            outputs = (
                VariableDefinition(
                    name="field",
                    role=VariableRole.OUTPUT,
                    data_type="scalar_field",
                    unit=None,
                    shape=self.plan.nominal_shape,
                    location=output_location,
                    summary={"physical_time": case.problem.t_end},
                ),
            )
        return ProblemAnalysis(
            case_id=case.case_id,
            domain=case.problem.domain,
            equation=case.problem.equation,
            schema_version=case.problem.schema_version,
            inputs=tuple(inputs),
            outputs=outputs,
            coarse_shape=self.plan.coarse_shape,
            nominal_shape=self.plan.nominal_shape,
            coarse_solver_id=_solver_id(self.coarse_solver),
            nominal_solver_id=_solver_id(self.nominal_solver),
            case_metadata=case.metadata,
        )

    def bootstrap(self, cases: Sequence[SimulationCase]) -> WorkflowBootstrapResult:
        bootstrap_cases = _unique_cases(cases)
        analyses = tuple(self.analyze(case) for case in bootstrap_cases)
        known_case_ids = {sample.case_id for sample in self._references}
        new_references: list[ReferenceSample] = []
        for case in bootstrap_cases:
            if case.case_id in known_case_ids:
                continue
            sample = self.bootstrap_pipeline.run_case(case)
            self._references.append(sample)
            new_references.append(sample)
            known_case_ids.add(case.case_id)
        if len(self._references) < 2:
            raise ValueError("Workflow bootstrap requires at least two reference cases.")
        self._initialize_active_loop()
        self._model_promoted = False
        artifact = self._save_model()
        return WorkflowBootstrapResult(
            analyses=analyses,
            new_references=tuple(new_references),
            model=self.surrogate.descriptor,
            total_reference_count=len(self.references),
            model_artifact=artifact,
        )

    def preview(
        self,
        case: SimulationCase,
        validate_nominal: bool = False,
    ) -> WorkflowPreviewResult:
        self._require_active_loop()
        analysis = self.analyze(case)
        preview = self.preview_pipeline.preview(case)
        validation = self.preview_pipeline.validate(case, preview) if validate_nominal else None
        accepted = self._model_promoted and self.preview_policy.accepts(preview.prediction)
        return WorkflowPreviewResult(
            analysis=analysis,
            preview=preview,
            accepted_by_policy=accepted,
            model_promoted=self._model_promoted,
            validation=validation,
        )

    def evaluate(self, cases: Sequence[SimulationCase]) -> ModelEvaluation:
        self._require_active_loop()
        evaluation_cases = _unique_cases(cases)
        if not evaluation_cases:
            raise ValueError("Model evaluation requires at least one disjoint case.")
        known_case_ids = {sample.case_id for sample in self.references}
        overlap = sorted(case.case_id for case in evaluation_cases if case.case_id in known_case_ids)
        if overlap:
            raise ValueError(f"Evaluation cases overlap training references: {', '.join(overlap)}")

        analyses = tuple(self.analyze(case) for case in evaluation_cases)
        validations = tuple(self._evaluate_case(case) for case in evaluation_cases)
        preview_errors = np.array(
            [validation.metrics["preview_relative_l2"] for validation in validations]
        )
        coarse_errors = np.array(
            [validation.metrics["coarse_relative_l2"] for validation in validations]
        )
        ratios = preview_errors / np.maximum(coarse_errors, 1e-15)
        coverages = np.array(
            [validation.metrics["coverage_2sigma"] for validation in validations]
        )
        metrics = {
            "mean_preview_relative_l2": float(np.mean(preview_errors)),
            "mean_coarse_relative_l2": float(np.mean(coarse_errors)),
            "worst_preview_relative_l2": float(np.max(preview_errors)),
            "worst_error_ratio_vs_coarse": float(np.max(ratios)),
            "minimum_coverage_2sigma": float(np.min(coverages)),
        }
        passed = self.model_quality_policy.accepts(metrics)
        self._model_promoted = passed
        report_path = self._write_evaluation_report(evaluation_cases, metrics, passed)
        return ModelEvaluation(
            analyses=analyses,
            validations=validations,
            metrics=metrics,
            passed=passed,
            report_path=report_path,
        )

    def run_campaign(
        self,
        candidates: Sequence[SimulationCase],
        evaluation_cases: Sequence[SimulationCase],
        policy: LearningCampaignPolicy | None = None,
    ) -> LearningCampaignResult:
        campaign_policy = policy or LearningCampaignPolicy()
        candidate_cases = _unique_cases(candidates)
        holdout_cases = _unique_cases(evaluation_cases)
        initial_evaluation = self.evaluate(holdout_cases)
        if campaign_policy.minimum_iterations == 0 and initial_evaluation.passed:
            return LearningCampaignResult(
                initial_evaluation=initial_evaluation,
                rounds=(),
                final_evaluation=initial_evaluation,
                stop_reason=CampaignStopReason.QUALITY_PASSED,
                total_reference_count=len(self.references),
            )

        rounds: list[CampaignRound] = []
        final_evaluation = initial_evaluation
        stop_reason = CampaignStopReason.MAX_ITERATIONS
        for index in range(1, campaign_policy.max_iterations + 1):
            known_case_ids = {sample.case_id for sample in self.references}
            unseen = tuple(case for case in candidate_cases if case.case_id not in known_case_ids)
            if not unseen:
                stop_reason = CampaignStopReason.CANDIDATES_EXHAUSTED
                break
            iteration = self.run_iteration(unseen, campaign_policy.selection_count)
            final_evaluation = self.evaluate(holdout_cases)
            rounds.append(
                CampaignRound(
                    index=index,
                    iteration=iteration,
                    evaluation=final_evaluation,
                )
            )
            if index >= campaign_policy.minimum_iterations and final_evaluation.passed:
                stop_reason = CampaignStopReason.QUALITY_PASSED
                break
        return LearningCampaignResult(
            initial_evaluation=initial_evaluation,
            rounds=tuple(rounds),
            final_evaluation=final_evaluation,
            stop_reason=stop_reason,
            total_reference_count=len(self.references),
        )

    def run_iteration(
        self,
        candidates: Sequence[SimulationCase],
        selection_count: int,
    ) -> WorkflowIterationResult:
        active_loop = self._require_active_loop()
        candidate_cases = _unique_cases(candidates)
        analyses = tuple(self.analyze(case) for case in candidate_cases)
        iteration = active_loop.run_iteration(candidate_cases, selection_count)
        self._references = list(active_loop.references)
        self._model_promoted = False
        artifact = self._save_model()
        return WorkflowIterationResult(
            analyses=analyses,
            active_learning=iteration,
            model_artifact=artifact,
        )

    def _initialize_active_loop(self) -> None:
        self._active_loop = ActiveLearningLoop(
            bootstrap_pipeline=self.bootstrap_pipeline,
            preview_pipeline=self.preview_pipeline,
            surrogate=self.surrogate,
            references=self._references,
            policy=self.active_learning_policy,
            store=self.store,
        )
        self._references = list(self._active_loop.references)

    def _require_active_loop(self) -> ActiveLearningLoop:
        if self._active_loop is None:
            raise RuntimeError("Bootstrap or load at least two references before previewing.")
        return self._active_loop

    def _save_model(self) -> Path | None:
        if self.model_artifact is None:
            return None
        return self.surrogate.save(self.model_artifact)

    def _evaluate_case(self, case: SimulationCase) -> PreviewValidation:
        cached = self._evaluation_cache.get(case.case_id)
        if cached is None:
            preview = self.preview_pipeline.preview(case)
            nominal = self.nominal_solver.solve(
                case,
                Fidelity.NOMINAL,
                self.plan.nominal_shape,
            )
            self._evaluation_cache[case.case_id] = (case, preview.coarse, nominal)
            return self.preview_pipeline.validate_against_nominal(case, preview, nominal)
        cached_case, coarse, nominal = cached
        if not _same_case(cached_case, case):
            raise ValueError(f"Evaluation case_id {case.case_id!r} was reused with new inputs.")
        preview = self.preview_pipeline.preview_from_coarse(case, coarse)
        return self.preview_pipeline.validate_against_nominal(case, preview, nominal)

    def _write_evaluation_report(
        self,
        cases: Sequence[SimulationCase],
        metrics: Mapping[str, float],
        passed: bool,
    ) -> Path | None:
        if self.model_artifact is None:
            return None
        report_path = self.model_artifact.with_suffix(".evaluation.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.surrogate.descriptor.model_id,
            "training_case_ids": self.surrogate.descriptor.training_case_ids,
            "evaluation_case_ids": [case.case_id for case in cases],
            "policy": {
                "max_relative_l2": self.model_quality_policy.max_relative_l2,
                "max_error_ratio_vs_coarse": (
                    self.model_quality_policy.max_error_ratio_vs_coarse
                ),
                "min_coverage_2sigma": self.model_quality_policy.min_coverage_2sigma,
            },
            "metrics": metrics,
            "passed": passed,
        }
        temporary = report_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(report_path)
        return report_path


def _solver_id(solver: Solver) -> str:
    adapter_id = getattr(solver, "adapter_id", None)
    return str(adapter_id) if adapter_id else type(solver).__name__


def _solver_output_location(solver: Solver) -> FieldLocation:
    location = getattr(solver, "output_location", FieldLocation.POINT)
    if not isinstance(location, FieldLocation):
        raise ValueError("Solver output_location must be a FieldLocation.")
    return location


def _unique_cases(cases: Sequence[SimulationCase]) -> tuple[SimulationCase, ...]:
    case_tuple = tuple(cases)
    case_ids = [case.case_id for case in case_tuple]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Workflow case identifiers must be unique.")
    return case_tuple


def _same_case(first: SimulationCase, second: SimulationCase) -> bool:
    return (
        first.case_id == second.case_id
        and first.problem == second.problem
        and first.boundaries == second.boundaries
        and first.metadata == second.metadata
        and np.array_equal(first.initial_field, second.initial_field)
    )
