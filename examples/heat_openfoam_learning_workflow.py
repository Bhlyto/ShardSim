from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.dataset import ReferenceDatasetStore
from shardsim.design import HeatDesignSpace
from shardsim.pipeline import FidelityPlan
from shardsim.solvers.heat import HeatEquationSolver
from shardsim.solvers.openfoam import OpenFOAMAdapter
from shardsim.surrogates.heat_local import HeatLocalResidualSurrogate
from shardsim.workflow import LearningCampaignPolicy, SimulationLearningWorkflow


def main() -> None:
    nominal_solver = OpenFOAMAdapter()
    if not nominal_solver.is_available():
        raise SystemExit("Docker or the pinned OpenFOAM image is unavailable.")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        training_space = HeatDesignSpace(
            alpha=(0.018, 0.028),
            t_end=(0.018, 0.024),
            center_x=(0.35, 0.65),
            center_y=(0.35, 0.65),
            sigma_x=(0.075, 0.115),
            sigma_y=(0.075, 0.115),
            amplitude=(0.85, 1.15),
            initial_shape=(33, 33),
        )
        candidate_space = HeatDesignSpace(
            alpha=(0.015, 0.050),
            t_end=(0.015, 0.030),
            center_x=(0.20, 0.80),
            center_y=(0.20, 0.80),
            sigma_x=(0.06, 0.14),
            sigma_y=(0.06, 0.14),
            amplitude=(0.75, 1.25),
            initial_shape=(33, 33),
        )
        workflow = SimulationLearningWorkflow(
            coarse_solver=HeatEquationSolver(),
            nominal_solver=nominal_solver,
            plan=FidelityPlan(coarse_shape=(9, 9), nominal_shape=(16, 16)),
            store=ReferenceDatasetStore(root / "dataset", "openfoam-learning-demo"),
            model_artifact=root / "models" / "heat-openfoam.npz",
            surrogate=HeatLocalResidualSurrogate(),
        )
        bootstrap = workflow.bootstrap(training_space.sample(5, seed=11, prefix="bootstrap"))
        campaign = workflow.run_campaign(
            candidates=candidate_space.sample(3, seed=23, prefix="candidate"),
            evaluation_cases=training_space.sample(2, seed=37, prefix="evaluation"),
            policy=LearningCampaignPolicy(
                selection_count=1,
                max_iterations=3,
                minimum_iterations=1,
            ),
        )
        runtime_preview = workflow.preview(
            training_space.sample(1, seed=41, prefix="runtime")[0],
            validate_nominal=False,
        )
        last_round = campaign.rounds[-1]
        selected = last_round.iteration.active_learning.selected[0]
        validation = last_round.iteration.active_learning.validations[0]
        evaluation = campaign.final_evaluation
        summary = {
            "problem": {
                "domain": bootstrap.analyses[0].domain,
                "equation": bootstrap.analyses[0].equation,
                "inputs": [variable.name for variable in bootstrap.analyses[0].inputs],
                "outputs": [variable.name for variable in bootstrap.analyses[0].outputs],
            },
            "solvers": {
                "coarse": bootstrap.analyses[0].coarse_solver_id,
                "nominal": bootstrap.analyses[0].nominal_solver_id,
            },
            "bootstrap_references": bootstrap.total_reference_count,
            "model_algorithm": bootstrap.model.metadata["algorithm"],
            "selected_case": selected.case.case_id,
            "selected_ood_score": selected.ood_score,
            "comparison": validation.metrics,
            "campaign": {
                "rounds": len(campaign.rounds),
                "stop_reason": campaign.stop_reason.value,
                "references": campaign.total_reference_count,
            },
            "model_evaluation": {
                "passed": evaluation.passed,
                **evaluation.metrics,
            },
            "runtime_preview": {
                "case_id": runtime_preview.preview.case_id,
                "accepted": runtime_preview.accepted_by_policy,
                "model_promoted": runtime_preview.model_promoted,
                "nominal_solver_called": runtime_preview.validation is not None,
            },
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
