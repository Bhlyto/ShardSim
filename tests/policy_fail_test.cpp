#include <exception>
#include <iostream>
#include <string>

#include "shardsim/config.hpp"
#include "shardsim/orchestrator/orchestrator.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: policy_fail_test <wallclock|memory>\n";
        return 2;
    }

    const std::string mode = argv[1];

    try {
        shardsim::SimulationConfig config;
        config.grid_x = 128;
        config.grid_y = 128;
        config.steps = 200;
        config.dt = 0.001;
        config.alpha = 1.0;
        config.coarse_tolerance = 0.05;
        config.fine_tolerance = 0.03;
        config.halo_overhead_ratio_max = 0.99;

        if (mode == "wallclock") {
            config.wallclock_limit_minutes = 0;
            config.wallclock_limit_ms = 1;  // Intentionally tiny to force failure.
            config.memory_ceiling_gb = 16;
            config.memory_ceiling_mb = 0;
        } else if (mode == "memory") {
            config.wallclock_limit_minutes = 0;
            config.wallclock_limit_ms = 0;
            config.memory_ceiling_gb = 0;
            config.memory_ceiling_mb = 1;  // Intentionally tiny to force failure.
        } else {
            std::cerr << "unknown mode: " << mode << "\n";
            return 2;
        }

        shardsim::orchestrator::Orchestrator orchestrator(config);
        const auto summary = orchestrator.run();
        (void)summary;

        // If we reached here, guardrail did not fail as expected.
        return 0;
    } catch (const std::exception&) {
        // Expected path for this test executable. CTest marks these as WILL_FAIL.
        return 1;
    }
}
