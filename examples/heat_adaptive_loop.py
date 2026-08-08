from shardsim.adaptive import AdaptivePreviewPipeline
from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
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
    surrogate = MeanDeltaSurrogate()
    surrogate.fit([bootstrap.run_case(case) for case in training_cases])

    pipeline = AdaptivePreviewPipeline(
        solver=solver,
        surrogate=surrogate,
        plan=plan,
        tile_shape=(16, 16),
        max_regions=1,
        halo=6,
    )
    case = make_case("adaptive-evaluation", (0.51, 0.49), 0.021)
    adaptive_preview = pipeline.run(case)
    validation = pipeline.validate(case, adaptive_preview)

    print(f"regions={len(adaptive_preview.refinement.regions)}")
    print(f"refined_domain={validation.metrics['refined_domain_fraction']:.3%}")
    print(f"estimated_compute={validation.metrics['estimated_total_compute_fraction']:.3%}")
    print(f"coarse_mae={validation.metrics['coarse_mae']:.6e}")
    print(f"preview_mae={validation.metrics['preview_mae']:.6e}")
    print(f"adaptive_mae={validation.metrics['adaptive_mae']:.6e}")


if __name__ == "__main__":
    main()
