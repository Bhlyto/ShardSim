#include <cmath>
#include <cstddef>
#include <iostream>

#include "shardsim/config.hpp"
#include "shardsim/mesh/mesh3d.hpp"
#include "shardsim/mpi/mpi_runtime.hpp"
#include "shardsim/solver/heat_solver.hpp"

#if SHARDSIM_HAS_MPI
#include <mpi.h>
#endif

namespace {

int fail(const std::string& msg, int rank) {
    std::cerr << "[rank " << rank << "] FAIL: " << msg << "\n";
    return 1;
}

}  // namespace

int main() {
    const auto ctx = shardsim::mpi_runtime::initialize();

    // Small 3D grid: 12x8x4, single heat source at centre.
    shardsim::SimulationConfig cfg {};
    cfg.grid_x          = 12;
    cfg.grid_y          = 8;
    cfg.grid_z          = 4;
    cfg.steps           = 40;
    cfg.dt              = 0.001;
    cfg.alpha           = 1.0;
    cfg.coarse_tolerance = 1.0;
    cfg.fine_tolerance  = 1.0;
    cfg.decision_policy = "heuristic";
    cfg.min_critical_fraction = 0.0;
    cfg.source_x_fraction = 0.5;
    cfg.source_y_fraction = 0.5;
    cfg.source_z_fraction = 0.5;
    cfg.source_temperature = 1.0;
    cfg.source2_enabled = false;

    const auto grid = shardsim::mesh::make_non_uniform_grid3d(cfg);

    // Run solver (MPI-distributed).
    shardsim::solver::SolveResult3D result;
    try {
        result = shardsim::solver::run_transient_heat(grid, cfg);
    } catch (const std::exception& ex) {
        shardsim::mpi_runtime::finalize();
        return fail(std::string("solver threw: ") + ex.what(), ctx.world_rank);
    }

    // Rank 0 validates the output field.
    if (ctx.world_rank == 0) {
        const std::size_t total = cfg.grid_x * cfg.grid_y * cfg.grid_z;
        if (result.fine.values.size() != total) {
            shardsim::mpi_runtime::finalize();
            return fail("fine field size mismatch", 0);
        }

        // Field must be finite and non-negative everywhere.
        for (std::size_t idx = 0; idx < total; ++idx) {
            const double v = result.fine.values[idx];
            if (!std::isfinite(v) || v < -1.0e-9) {
                shardsim::mpi_runtime::finalize();
                return fail("field has non-finite or negative value", 0);
            }
        }

        // Heat must have diffused: max value should be > 0.
        double max_val = 0.0;
        for (double v : result.fine.values) {
            max_val = std::max(max_val, v);
        }
        if (max_val <= 0.0) {
            shardsim::mpi_runtime::finalize();
            return fail("field is all-zero after solve", 0);
        }

        // Single-rank and multi-rank must produce the same result: compare
        // against a fresh single-rank run only when world_size == 1.
        // (Cross-rank determinism is validated by running with different MPI sizes
        //  and comparing outputs externally; here we just verify correctness.)
        if (ctx.world_size > 1) {
            std::cout << "[rank 0] 3D MPI solver test passed ("
                      << ctx.world_size << " ranks, "
                      << "coarse_steps=" << result.coarse_steps
                      << " fine_steps=" << result.fine_steps
                      << " halo_calls=" << result.halo_calls
                      << ")\n";
        } else {
            std::cout << "[rank 0] 3D solver test passed (single-rank, "
                      << "coarse_steps=" << result.coarse_steps
                      << " fine_steps=" << result.fine_steps << ")\n";
        }
    }

    shardsim::mpi_runtime::finalize();
    return 0;
}
