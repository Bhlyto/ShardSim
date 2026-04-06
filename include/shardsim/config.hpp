#pragma once

#include <cstddef>
#include <string>

namespace shardsim {

struct SimulationConfig {
    std::size_t grid_x {128};
    std::size_t grid_y {128};
    std::size_t grid_z {1};
    std::size_t steps {200};

    double dt {1e-3};
    double alpha {1.0};

    double coarse_tolerance {5e-2};
    double fine_tolerance {5e-3};

    double refine_local_error_tau {0.05};
    double refine_uncertainty_tau {0.20};
    double min_critical_fraction {0.0};

    std::size_t memory_ceiling_gb {16};
    std::size_t memory_ceiling_mb {0};
    std::size_t wallclock_limit_minutes {0};
    std::size_t wallclock_limit_ms {0};
    double halo_overhead_ratio_max {0.95};

    bool deterministic_mode {true};
    std::string partitioning_policy {"strict_geometric"};
    std::string decision_policy {"heuristic"};
    std::string solver_family {"generic_thermal"};
    // decomposition_mode: "hybrid" (default) | "coarse" | "fine"
    std::string decomposition_mode {"hybrid"};

    bool export_training_data {false};
    std::string training_data_export_dir {};

    std::string surrogate_model_path {};
    std::string surrogate_python_executable {"python3"};
    std::string surrogate_script_path {"scripts/predict_critical_mask.py"};
    std::string surrogate_temp_dir {};
    double surrogate_score_threshold {0.0};
    double surrogate_top_fraction {0.0};

    double source_x_fraction {0.5};
    double source_y_fraction {0.5};
    double source_z_fraction {0.5};
    double source_temperature {100.0};

    bool source2_enabled {false};
    double source2_x_fraction {0.75};
    double source2_y_fraction {0.25};
    double source2_z_fraction {0.5};
    double source2_temperature {50.0};

    // Pre-simulation module (Phase 3).
    // presim_steps > 0 enables a fast coarsened pre-simulation pass that generates
    // an uncertainty map used to seed the decision core before the main solve.
    std::size_t presim_steps {0};
    std::size_t presim_coarsening_factor {4};  // spatial coarsening ratio relative to main grid

    // ML correction loop (Phase 4).
    // After the coarse solve, apply a trained correction model to estimate the
    // fine field without running the full fine solve.
    // correction_policy: "none" (default) | "linear" | "python"
    // When enabled, the corrected coarse field is returned as `SolveResult::fine`.
    std::string correction_policy {"none"};
    std::string correction_model_path {};
    std::string correction_script_path {"scripts/apply_correction.py"};
    std::string correction_python_executable {"python3"};

    // Automatic orchestration knobs used by external solver-aware planners.
    bool auto_enable_presim {false};
    std::size_t auto_presim_steps {0};
    std::size_t auto_presim_coarsening_factor {4};
    bool auto_enable_correction {false};
    std::string auto_correction_policy {"none"};
    std::string auto_correction_model_path {};
};

}  // namespace shardsim
