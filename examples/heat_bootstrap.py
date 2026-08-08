from shardsim.domains.heat2d import gaussian_initial_field, make_heat_case
from shardsim.pipeline import BootstrapPipeline, FidelityPlan
from shardsim.solvers.heat import HeatEquationSolver


def main() -> None:
    initial_field = gaussian_initial_field((129, 129), sigma=(0.08, 0.08))
    case = make_heat_case(
        case_id="heat-gaussian-001",
        alpha=0.02,
        t_end=0.05,
        initial_field=initial_field,
        metadata={"scenario": "centered-gaussian"},
    )
    pipeline = BootstrapPipeline(
        solver=HeatEquationSolver(),
        plan=FidelityPlan(coarse_shape=(25, 25), nominal_shape=(101, 101)),
    )
    sample = pipeline.run_case(case)
    print(f"case={sample.case_id}")
    print(f"coarse_steps={sample.coarse.n_steps} nominal_steps={sample.nominal.n_steps}")
    print(f"t_end={sample.coarse.t_end:.6f}s")
    print(f"mae={sample.metrics['mae']:.6e}")
    print(f"relative_l2={sample.metrics['relative_l2']:.6e}")


if __name__ == "__main__":
    main()
