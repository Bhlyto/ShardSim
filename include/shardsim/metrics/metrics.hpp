#pragma once

#include <cstddef>

#include "shardsim/types.hpp"

namespace shardsim::metrics {

struct RunSummary {
    std::size_t steps {0};
    std::size_t coarse_steps {0};
    std::size_t fine_steps {0};
    std::size_t critical_cells {0};
    double critical_fraction {0.0};
    double decision_ms {0.0};
    std::size_t halo_calls {0};
    double runtime_ms {0.0};
    double halo_ms_local {0.0};
    double halo_ms_min {0.0};
    double halo_ms_avg {0.0};
    double halo_ms_max {0.0};
    double mae {0.0};
    double global_error_norm {0.0};
};

RunSummary summarize(const Field2D& coarse,
                     const Field2D& fine,
                     std::size_t coarse_steps,
                     std::size_t fine_steps,
                     std::size_t critical_cells,
                     double critical_fraction,
                     double decision_ms,
                     std::size_t halo_calls,
                     double runtime_ms);

RunSummary summarize(const Field3D& coarse,
                     const Field3D& fine,
                     std::size_t coarse_steps,
                     std::size_t fine_steps,
                     std::size_t critical_cells,
                     double critical_fraction,
                     double decision_ms,
                     std::size_t halo_calls,
                     double runtime_ms);

}  // namespace shardsim::metrics
