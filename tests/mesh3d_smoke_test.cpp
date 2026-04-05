#include <iostream>

#include "shardsim/config.hpp"
#include "shardsim/mesh/mesh3d.hpp"

int main() {
    shardsim::SimulationConfig config;
    config.grid_x = 24;
    config.grid_y = 18;
    config.grid_z = 12;

    const auto grid = shardsim::mesh::make_non_uniform_grid3d(config);

    if (grid.size.x != config.grid_x || grid.size.y != config.grid_y || grid.size.z != config.grid_z) {
        std::cerr << "3D grid size mismatch\n";
        return 1;
    }

    const std::size_t expected = config.grid_x * config.grid_y * config.grid_z;
    if (grid.dx.values.size() != expected || grid.dy.values.size() != expected || grid.dz.values.size() != expected) {
        std::cerr << "3D spacing field size mismatch\n";
        return 1;
    }

    for (std::size_t k = 0; k < config.grid_z; ++k) {
        for (std::size_t j = 0; j < config.grid_y; ++j) {
            for (std::size_t i = 0; i < config.grid_x; ++i) {
                const double dx = grid.dx.at(i, j, k);
                const double dy = grid.dy.at(i, j, k);
                const double dz = grid.dz.at(i, j, k);
                if (!(dx > 0.0 && dy > 0.0 && dz > 0.0)) {
                    std::cerr << "Non-positive 3D spacing encountered\n";
                    return 1;
                }
            }
        }
    }

    return 0;
}
