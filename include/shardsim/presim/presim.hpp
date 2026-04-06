#pragma once

#include <cstddef>
#include <vector>

#include "shardsim/config.hpp"
#include "shardsim/mesh/mesh.hpp"
#include "shardsim/types.hpp"

namespace shardsim::presim {

/// Per-cell uncertainty estimate produced by the pre-simulation pass.
/// Values are normalised to [0, 1] where higher means more likely to be critical.
struct UncertaintyMap {
    std::vector<double> scores;  // length nx * ny (row-major, same as Field2D)
    std::size_t nx {0};
    std::size_t ny {0};

    double at(std::size_t x, std::size_t y) const {
        return scores[x * ny + y];
    }
};

/// Run a fast coarsened pre-simulation on a 2D grid and return an uncertainty map
/// scaled back to the full-resolution grid shape.
///
/// The pre-sim runs `config.presim_steps` explicit heat-equation steps on a grid
/// coarsened by `config.presim_coarsening_factor` in each dimension.  The resulting
/// Laplacian and gradient magnitudes are used to produce a normalised uncertainty
/// score which is then bilinearly up-sampled to (nx, ny) matching the full grid.
///
/// If presim_steps == 0 this function returns an empty UncertaintyMap (scores empty).
UncertaintyMap run_presim(const mesh::Grid2D& full_grid,
                          const SimulationConfig& config);

/// Merge a pre-sim uncertainty map with the heuristic scores already computed by the
/// decision core.  The merged scores are a weighted average:
///   merged = (1 - weight) * heuristic + weight * presim
/// where weight = 0.5 by default (equal blend).
void blend_presim_scores(std::vector<double>& heuristic_scores,
                         const UncertaintyMap& presim_map,
                         std::size_t nx,
                         std::size_t ny,
                         double weight = 0.5);

}  // namespace shardsim::presim
