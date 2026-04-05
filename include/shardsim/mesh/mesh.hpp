#pragma once

#include "shardsim/config.hpp"
#include "shardsim/types.hpp"

namespace shardsim::mesh {

struct Grid2D {
    Vec2u size;
    Field2D dx;
    Field2D dy;
};

Grid2D make_non_uniform_grid(const SimulationConfig& config);

}  // namespace shardsim::mesh
