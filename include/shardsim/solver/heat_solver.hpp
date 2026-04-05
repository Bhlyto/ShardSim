#pragma once

#include "shardsim/config.hpp"
#include "shardsim/mesh/mesh.hpp"
#include "shardsim/mesh/mesh3d.hpp"
#include "shardsim/types.hpp"

namespace shardsim::solver {

struct SolveResult {
    Field2D coarse;
    Field2D fine;
    std::size_t coarse_steps {0};
    std::size_t fine_steps {0};
    std::size_t critical_cells {0};
    double critical_fraction {0.0};
    double decision_ms {0.0};
    std::size_t halo_calls {0};
    double halo_ms_local {0.0};
    double halo_ms_min {0.0};
    double halo_ms_avg {0.0};
    double halo_ms_max {0.0};
};

struct SolveResult3D {
    Field3D coarse;
    Field3D fine;
    std::size_t coarse_steps {0};
    std::size_t fine_steps {0};
    std::size_t critical_cells {0};
    double critical_fraction {0.0};
    double decision_ms {0.0};
    std::size_t halo_calls {0};
    double halo_ms_local {0.0};
    double halo_ms_min {0.0};
    double halo_ms_avg {0.0};
    double halo_ms_max {0.0};
};

SolveResult run_transient_heat(const mesh::Grid2D& grid, const SimulationConfig& config);
SolveResult3D run_transient_heat(const mesh::Grid3D& grid, const SimulationConfig& config);

}  // namespace shardsim::solver
