#include "shardsim/mesh/mesh3d.hpp"

#include <cmath>

namespace shardsim::mesh {

Grid3D make_non_uniform_grid3d(const SimulationConfig& config) {
    Grid3D grid {};
    grid.size = {config.grid_x, config.grid_y, config.grid_z};

    const std::size_t n = grid.size.x * grid.size.y * grid.size.z;

    grid.dx.size = grid.size;
    grid.dy.size = grid.size;
    grid.dz.size = grid.size;
    grid.dx.values.resize(n, 1.0);
    grid.dy.values.resize(n, 1.0);
    grid.dz.values.resize(n, 1.0);

    for (std::size_t k = 0; k < grid.size.z; ++k) {
        for (std::size_t j = 0; j < grid.size.y; ++j) {
            for (std::size_t i = 0; i < grid.size.x; ++i) {
                const double x = static_cast<double>(i) / static_cast<double>(grid.size.x);
                const double y = static_cast<double>(j) / static_cast<double>(grid.size.y);
                const double z = static_cast<double>(k) /
                    static_cast<double>((grid.size.z > 1) ? grid.size.z : 2);
                const double mxy = std::sin(6.28318530718 * x * y);
                const double mxz = std::cos(6.28318530718 * x * z);
                const double myz = std::sin(6.28318530718 * y * z);

                grid.dx.at(i, j, k) = 0.7 + 0.2 * mxy;
                grid.dy.at(i, j, k) = 0.7 + 0.2 * mxz;
                grid.dz.at(i, j, k) = 0.7 + 0.2 * myz;
            }
        }
    }

    return grid;
}

}  // namespace shardsim::mesh
