from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.preview import PreviewPipeline, PreviewPolicy
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.surrogates.mean_delta import MeanDeltaSurrogate


def make_case(
    index: int,
    center: tuple[float, float],
    alpha: float,
    sigma_x: float = 0.08,
):
    initial_field = gaussian_initial_field(
        (65, 65),
        center=center,
        sigma=(sigma_x, 0.08),
    )
    return make_heat_case(
        case_id=f"train-{index:02d}",
        alpha=alpha,
        t_end=0.04,
        initial_field=initial_field,
        metadata={"center": center},
    )


def main() -> None:
    solver = HeatEquationSolver()
    plan = FidelityPlan(coarse_shape=(17, 17), nominal_shape=(65, 65))
    bootstrap = BootstrapPipeline(solver=solver, plan=plan)
    training_cases = [
        make_case(0, (0.45, 0.45), 0.018),
        make_case(1, (0.55, 0.45), 0.020, 0.082),
        make_case(2, (0.45, 0.55), 0.022, 0.084),
        make_case(3, (0.55, 0.55), 0.024, 0.086),
        make_case(4, (0.50, 0.48), 0.021, 0.083),
        make_case(5, (0.48, 0.52), 0.019, 0.081),
    ]
    references = [bootstrap.run_case(case) for case in training_cases]

    surrogate = MeanDeltaSurrogate()
    descriptor = surrogate.fit(references)
    preview_pipeline = PreviewPipeline(solver=solver, surrogate=surrogate, plan=plan)
    evaluation_case = make_case(99, (0.51, 0.49), 0.021, 0.082)
    preview = preview_pipeline.preview(evaluation_case)
    validation = preview_pipeline.validate(evaluation_case, preview)
    policy = PreviewPolicy(max_ood_score=3.0)

    print(f"model={descriptor.model_id} training_cases={len(descriptor.training_case_ids)}")
    print(f"ood_score={preview.prediction.ood_score:.4f}")
    print(f"preview_accepted={policy.accepts(preview.prediction)}")
    print(f"coarse_mae={validation.metrics['coarse_mae']:.6e}")
    print(f"preview_mae={validation.metrics['preview_mae']:.6e}")
    print(f"coverage_2sigma={validation.metrics['coverage_2sigma']:.3f}")


if __name__ == "__main__":
    main()
