#include "shardsim/mesh/mesh.hpp"

#include <cmath>

namespace shardsim::mesh {

Grid2D make_non_uniform_grid(const SimulationConfig& config) {
    Grid2D grid {};
    grid.size = {config.grid_x, config.grid_y};

    grid.dx.size = grid.size;
    grid.dy.size = grid.size;
    grid.dx.values.resize(grid.size.x * grid.size.y, 1.0);
    grid.dy.values.resize(grid.size.x * grid.size.y, 1.0);

    for (std::size_t j = 0; j < grid.size.y; ++j) {
        for (std::size_t i = 0; i < grid.size.x; ++i) {
            const double x = static_cast<double>(i) / static_cast<double>(grid.size.x);
            const double y = static_cast<double>(j) / static_cast<double>(grid.size.y);
            const double modulation = std::sin(6.28318530718 * x * y);
            const double stretch = 0.75 + 0.25 * modulation;  // [0.5, 1.0]
            grid.dx.at(i, j) = stretch;
            grid.dy.at(i, j) = 1.25 - stretch;  // [0.25, 0.75]
        }
    }

    return grid;
}

}  // namespace shardsim::mesh
