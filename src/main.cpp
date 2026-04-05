#include <exception>
#include <iostream>
#include <string>

#include "shardsim/io/config_loader.hpp"
#include "shardsim/mpi/mpi_runtime.hpp"
#include "shardsim/orchestrator/orchestrator.hpp"

int main(int argc, char** argv) {
    const std::string config_path = (argc > 1) ? argv[1] : "config/default.yaml";
    const auto mpi_ctx = shardsim::mpi_runtime::initialize();

    try {
        const auto config = shardsim::io::load_config(config_path);
        shardsim::orchestrator::Orchestrator orchestrator(config);
        const auto summary = orchestrator.run();

        if (mpi_ctx.world_rank == 0) {
            std::cout << "ShardSim run complete\n";
            std::cout << "  steps: " << summary.steps << "\n";
            std::cout << "  coarse_steps: " << summary.coarse_steps << "\n";
            std::cout << "  fine_steps: " << summary.fine_steps << "\n";
            std::cout << "  critical_cells: " << summary.critical_cells << "\n";
            std::cout << "  critical_fraction: " << summary.critical_fraction << "\n";
            std::cout << "  decision_ms: " << summary.decision_ms << "\n";
            std::cout << "  halo_calls: " << summary.halo_calls << "\n";
            std::cout << "  runtime_ms: " << summary.runtime_ms << "\n";
            std::cout << "  halo_ms_local: " << summary.halo_ms_local << "\n";
            std::cout << "  halo_ms_min: " << summary.halo_ms_min << "\n";
            std::cout << "  halo_ms_avg: " << summary.halo_ms_avg << "\n";
            std::cout << "  halo_ms_max: " << summary.halo_ms_max << "\n";
            const double runtime = (summary.runtime_ms > 1.0e-12) ? summary.runtime_ms : 1.0e-12;
            std::cout << "  halo_overhead_ratio: " << (summary.halo_ms_avg / runtime) << "\n";
            std::cout << "  mae: " << summary.mae << "\n";
            std::cout << "  global_error_norm: " << summary.global_error_norm << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ShardSim fatal error: " << ex.what() << "\n";
        return 1;
    }
}
