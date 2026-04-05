#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "shardsim/config.hpp"
#include "shardsim/mesh/mesh.hpp"
#include "shardsim/mesh/mesh3d.hpp"
#include "shardsim/types.hpp"

namespace shardsim::decision_core {

struct RegionSelection {
    std::vector<std::uint8_t> mask;
    std::size_t critical_cells {0};
    double critical_fraction {0.0};
};

struct RegionSelection3D {
    std::vector<std::uint8_t> mask;
    std::size_t critical_cells {0};
    double critical_fraction {0.0};
};

RegionSelection select_critical_regions(const mesh::Grid2D& grid,
                                        const Field2D& coarse,
                                        const SimulationConfig& config);

RegionSelection3D select_critical_regions(const mesh::Grid3D& grid,
                                          const Field3D& coarse,
                                          const SimulationConfig& config);

}  // namespace shardsim::decision_core
