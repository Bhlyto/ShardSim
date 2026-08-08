import tempfile
from pathlib import Path

from shardsim.active_learning import ActiveLearningLoop, ActiveLearningPolicy
from shardsim.dataset import ReferenceDatasetStore
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.preview import PreviewPipeline
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.surrogates.mean_delta import MeanDeltaSurrogate


def make_case(case_id: str, center: tuple[float, float], alpha: float):
    return make_heat_case(
        case_id=case_id,
        alpha=alpha,
        t_end=0.04,
        initial_field=gaussian_initial_field(
            (65, 65),
            center=center,
            sigma=(0.08, 0.08),
        ),
    )


def main() -> None:
    solver = HeatEquationSolver()
    plan = FidelityPlan(coarse_shape=(17, 17), nominal_shape=(65, 65))
    bootstrap = BootstrapPipeline(solver, plan)
    training_cases = [
        make_case("train-0", (0.45, 0.45), 0.018),
        make_case("train-1", (0.55, 0.45), 0.020),
        make_case("train-2", (0.45, 0.55), 0.022),
        make_case("train-3", (0.55, 0.55), 0.024),
    ]
    references = [bootstrap.run_case(case) for case in training_cases]
    surrogate = MeanDeltaSurrogate()
    preview_pipeline = PreviewPipeline(solver, surrogate, plan)
    candidates = [
        make_case("candidate-near", (0.50, 0.50), 0.021),
        make_case("candidate-shifted", (0.30, 0.70), 0.026),
        make_case("candidate-alpha", (0.50, 0.50), 0.080),
        make_case("candidate-corner", (0.15, 0.15), 0.015),
    ]

    with tempfile.TemporaryDirectory() as directory:
        store = ReferenceDatasetStore(Path(directory) / "heat-dataset", "heat-active-demo")
        loop = ActiveLearningLoop(
            bootstrap_pipeline=bootstrap,
            preview_pipeline=preview_pipeline,
            surrogate=surrogate,
            references=references,
            policy=ActiveLearningPolicy(
                ood_weight=1.0,
                uncertainty_weight=1.0,
                diversity_weight=0.25,
            ),
            store=store,
        )
        selected = loop.select(candidates, count=2)
        for rank, assessment in enumerate(selected, start=1):
            print(
                f"rank={rank} case={assessment.case.case_id} "
                f"ood={assessment.ood_score:.3f} "
                f"uncertainty={assessment.relative_uncertainty:.3e} "
                f"diversity={assessment.diversity_score:.3f}"
            )
        iteration = loop.enrich(selected)
        print(f"new_references={len(iteration.new_references)}")
        print(f"total_references={iteration.total_reference_count}")
        print(f"persisted_cases={len(store.case_ids())}")
        print(f"model_training_cases={len(iteration.model.training_case_ids)}")


if __name__ == "__main__":
    main()
