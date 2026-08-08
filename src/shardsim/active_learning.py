from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

import numpy as np

from shardsim.contracts import ModelDescriptor, SimulationCase
from shardsim.dataset import ReferenceDatasetStore
from shardsim.pipeline import BootstrapPipeline, ReferenceSample
from shardsim.preview import PreviewPipeline, PreviewResult, PreviewValidation
from shardsim.surrogates.base import PersistentDeltaSurrogate


def _candidate_features(case: SimulationCase, field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    parameter_names = tuple(sorted(case.problem.parameters))
    y = np.linspace(0.0, 1.0, values.shape[0])
    x = np.linspace(0.0, 1.0, values.shape[1])
    x_grid, y_grid = np.meshgrid(x, y)
    weights = np.abs(values)
    weight_sum = float(np.sum(weights))
    if weight_sum > 1e-12:
        center_x = float(np.sum(weights * x_grid) / weight_sum)
        center_y = float(np.sum(weights * y_grid) / weight_sum)
        spread_x = float(np.sqrt(np.sum(weights * np.square(x_grid - center_x)) / weight_sum))
        spread_y = float(np.sqrt(np.sum(weights * np.square(y_grid - center_y)) / weight_sum))
    else:
        center_x = center_y = 0.5
        spread_x = spread_y = 0.0
    return np.array(
        [
            *(case.problem.parameter(name) for name in parameter_names),
            case.problem.t_end,
            *case.problem.extent,
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.sqrt(np.mean(np.square(values))),
            center_x,
            center_y,
            spread_x,
            spread_y,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    case: SimulationCase
    preview: PreviewResult
    ood_score: float
    relative_uncertainty: float
    diversity_score: float
    priority_score: float


@dataclass(frozen=True, slots=True)
class ActiveLearningPolicy:
    ood_weight: float = 1.0
    uncertainty_weight: float = 1.0
    diversity_weight: float = 0.25

    def __post_init__(self) -> None:
        weights = (self.ood_weight, self.uncertainty_weight, self.diversity_weight)
        if any(not isfinite(float(weight)) or weight < 0 for weight in weights):
            raise ValueError("Active-learning weights must be finite and non-negative.")
        if sum(weights) == 0:
            raise ValueError("At least one active-learning weight must be positive.")

    def select(
        self,
        preview_pipeline: PreviewPipeline,
        candidates: Sequence[SimulationCase],
        count: int,
    ) -> tuple[CandidateAssessment, ...]:
        candidate_cases = tuple(candidates)
        if count < 0:
            raise ValueError("Selection count cannot be negative.")
        if len({case.case_id for case in candidate_cases}) != len(candidate_cases):
            raise ValueError("Candidate case identifiers must be unique.")
        if count == 0 or not candidate_cases:
            return ()

        previews = [preview_pipeline.preview(case) for case in candidate_cases]
        features = np.stack(
            [
                _candidate_features(case, preview.coarse_on_nominal)
                for case, preview in zip(candidate_cases, previews)
            ]
        )
        feature_scale = np.std(features, axis=0)
        standardized = (features - np.mean(features, axis=0)) / np.where(
            feature_scale < 1e-12,
            1.0,
            feature_scale,
        )

        base_scores: list[tuple[float, float, float]] = []
        for preview in previews:
            uncertainty = float(np.mean(preview.prediction.uncertainty))
            field_scale = float(np.sqrt(np.mean(np.square(preview.prediction.mean))))
            relative_uncertainty = uncertainty / max(field_scale, 1e-12)
            ood_score = preview.prediction.ood_score
            base_score = self.ood_weight * ood_score + self.uncertainty_weight * relative_uncertainty
            base_scores.append((base_score, ood_score, relative_uncertainty))

        remaining = set(range(len(candidate_cases)))
        selected_indices: list[int] = []
        assessments: list[CandidateAssessment] = []
        while remaining and len(assessments) < min(count, len(candidate_cases)):
            ranked: list[tuple[float, str, int, float]] = []
            for index in remaining:
                if selected_indices:
                    diversity = min(
                        float(
                            np.linalg.norm(standardized[index] - standardized[selected_index])
                            / sqrt(standardized.shape[1])
                        )
                        for selected_index in selected_indices
                    )
                else:
                    diversity = 0.0
                priority = base_scores[index][0] + self.diversity_weight * diversity
                ranked.append((priority, candidate_cases[index].case_id, index, diversity))
            priority, _, selected_index, diversity = max(
                ranked,
                key=lambda item: (item[0], item[1]),
            )
            _, ood_score, relative_uncertainty = base_scores[selected_index]
            assessments.append(
                CandidateAssessment(
                    case=candidate_cases[selected_index],
                    preview=previews[selected_index],
                    ood_score=ood_score,
                    relative_uncertainty=relative_uncertainty,
                    diversity_score=diversity,
                    priority_score=priority,
                )
            )
            selected_indices.append(selected_index)
            remaining.remove(selected_index)
        return tuple(assessments)


@dataclass(frozen=True, slots=True)
class ActiveLearningIteration:
    selected: tuple[CandidateAssessment, ...]
    new_references: tuple[ReferenceSample, ...]
    validations: tuple[PreviewValidation, ...]
    model: ModelDescriptor
    total_reference_count: int


class ActiveLearningLoop:
    def __init__(
        self,
        bootstrap_pipeline: BootstrapPipeline,
        preview_pipeline: PreviewPipeline,
        surrogate: PersistentDeltaSurrogate,
        references: Sequence[ReferenceSample],
        policy: ActiveLearningPolicy | None = None,
        store: ReferenceDatasetStore | None = None,
    ) -> None:
        self.bootstrap_pipeline = bootstrap_pipeline
        self.preview_pipeline = preview_pipeline
        self.surrogate = surrogate
        self.policy = policy or ActiveLearningPolicy()
        self.store = store
        self._references = list(references)
        if len(self._references) < 2:
            raise ValueError("Active learning requires at least two bootstrap references.")
        self.surrogate.fit(self._references)
        if self.store is not None:
            persisted = set(self.store.case_ids())
            for sample in self._references:
                if sample.case_id not in persisted:
                    self.store.add(sample)

    @property
    def references(self) -> tuple[ReferenceSample, ...]:
        return tuple(self._references)

    def select(
        self,
        candidates: Sequence[SimulationCase],
        count: int,
    ) -> tuple[CandidateAssessment, ...]:
        known_case_ids = {sample.case_id for sample in self._references}
        unseen = tuple(case for case in candidates if case.case_id not in known_case_ids)
        return self.policy.select(self.preview_pipeline, unseen, count)

    def enrich(
        self,
        selected: Sequence[CandidateAssessment],
    ) -> ActiveLearningIteration:
        selected_candidates = tuple(selected)
        known_case_ids = {sample.case_id for sample in self._references}
        new_references: list[ReferenceSample] = []
        validations: list[PreviewValidation] = []
        for assessment in selected_candidates:
            if assessment.case.case_id in known_case_ids:
                continue
            sample = self.bootstrap_pipeline.complete_reference(
                assessment.case,
                assessment.preview.coarse,
            )
            self._references.append(sample)
            new_references.append(sample)
            validations.append(
                self.preview_pipeline.validate_against_nominal(
                    assessment.case,
                    assessment.preview,
                    sample.nominal,
                )
            )
            known_case_ids.add(sample.case_id)
            if self.store is not None:
                self.store.add(sample)
        descriptor = self.surrogate.fit(self._references)
        return ActiveLearningIteration(
            selected=selected_candidates,
            new_references=tuple(new_references),
            validations=tuple(validations),
            model=descriptor,
            total_reference_count=len(self._references),
        )

    def run_iteration(
        self,
        candidates: Sequence[SimulationCase],
        count: int,
    ) -> ActiveLearningIteration:
        return self.enrich(self.select(candidates, count))
