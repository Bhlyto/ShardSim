#pragma once

#include "shardsim/config.hpp"
#include "shardsim/types.hpp"

namespace shardsim::mesh {

struct Grid3D {
    Vec3u size;
    Field3D dx;
    Field3D dy;
    Field3D dz;
};

Grid3D make_non_uniform_grid3d(const SimulationConfig& config);

}  // namespace shardsim::mesh
