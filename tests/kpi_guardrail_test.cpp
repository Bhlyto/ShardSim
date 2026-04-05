#include <exception>
#include <iostream>

#include "shardsim/config.hpp"
#include "shardsim/orchestrator/orchestrator.hpp"

int main() {
    try {
        shardsim::SimulationConfig config;
        config.grid_x = 96;
        config.grid_y = 96;
        config.steps = 120;
        config.dt = 0.001;
        config.alpha = 1.0;
        config.coarse_tolerance = 0.08;
        config.fine_tolerance = 0.04;
        config.memory_ceiling_gb = 16;
        config.wallclock_limit_minutes = 0;
        config.halo_overhead_ratio_max = 0.98;

        shardsim::orchestrator::Orchestrator orchestrator(config);
        const auto summary = orchestrator.run();

        const double runtime = (summary.runtime_ms > 1.0e-12) ? summary.runtime_ms : 1.0e-12;
        const double ratio = summary.halo_ms_avg / runtime;
        if (ratio > config.halo_overhead_ratio_max) {
            std::cerr << "KPI guardrail failed: ratio=" << ratio
                      << " max=" << config.halo_overhead_ratio_max << "\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "KPI guardrail test exception: " << ex.what() << "\n";
        return 1;
    }
}
