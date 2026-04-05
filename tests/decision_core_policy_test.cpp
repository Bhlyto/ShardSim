#include <cmath>
#include <iostream>

#include "shardsim/config.hpp"
#include "shardsim/decision_core/policy.hpp"
#include "shardsim/mesh/mesh.hpp"
#include "shardsim/types.hpp"

int main() {
    shardsim::SimulationConfig cfg;
    cfg.grid_x = 64;
    cfg.grid_y = 64;

    const auto grid = shardsim::mesh::make_non_uniform_grid(cfg);

    shardsim::Field2D coarse;
    coarse.size = {cfg.grid_x, cfg.grid_y};
    coarse.values.assign(cfg.grid_x * cfg.grid_y, 0.0);

    const double cx = static_cast<double>(cfg.grid_x) * 0.5;
    const double cy = static_cast<double>(cfg.grid_y) * 0.5;
    for (std::size_t j = 0; j < cfg.grid_y; ++j) {
        for (std::size_t i = 0; i < cfg.grid_x; ++i) {
            const double dx = static_cast<double>(i) - cx;
            const double dy = static_cast<double>(j) - cy;
            coarse.at(i, j) = std::exp(-(dx * dx + dy * dy) / 180.0);
        }
    }

    // Monotonicity check: looser thresholds should not select fewer cells.
    auto cfg_loose = cfg;
    cfg_loose.refine_local_error_tau = 0.15;
    cfg_loose.refine_uncertainty_tau = 0.15;
    cfg_loose.min_critical_fraction = 0.0;

    auto cfg_strict = cfg;
    cfg_strict.refine_local_error_tau = 0.60;
    cfg_strict.refine_uncertainty_tau = 0.60;
    cfg_strict.min_critical_fraction = 0.0;

    const auto loose = shardsim::decision_core::select_critical_regions(grid, coarse, cfg_loose);
    const auto strict = shardsim::decision_core::select_critical_regions(grid, coarse, cfg_strict);

    if (loose.critical_cells < strict.critical_cells) {
        std::cerr << "Monotonicity violated: loose=" << loose.critical_cells
                  << " strict=" << strict.critical_cells << "\n";
        return 1;
    }

    // Minimum critical-fraction check under very strict thresholds.
    auto cfg_min = cfg;
    cfg_min.refine_local_error_tau = 1.0;
    cfg_min.refine_uncertainty_tau = 1.0;
    cfg_min.min_critical_fraction = 0.10;

    const auto with_min = shardsim::decision_core::select_critical_regions(grid, coarse, cfg_min);
    const std::size_t min_required = static_cast<std::size_t>(
        std::ceil(cfg_min.min_critical_fraction * static_cast<double>(cfg.grid_x * cfg.grid_y)));

    if (with_min.critical_cells < min_required) {
        std::cerr << "Min critical fraction violated: selected=" << with_min.critical_cells
                  << " required=" << min_required << "\n";
        return 1;
    }

    if (with_min.mask.size() != cfg.grid_x * cfg.grid_y) {
        std::cerr << "Mask size mismatch\n";
        return 1;
    }

    return 0;
}
