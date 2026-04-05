#include "shardsim/metrics/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace shardsim::metrics {

RunSummary summarize(const Field2D& coarse,
                     const Field2D& fine,
                     std::size_t coarse_steps,
                     std::size_t fine_steps,
                     std::size_t critical_cells,
                     double critical_fraction,
                     double decision_ms,
                     std::size_t halo_calls,
                     double runtime_ms) {
    if (coarse.size.x != fine.size.x || coarse.size.y != fine.size.y) {
        throw std::runtime_error("Metric computation requires same field dimensions");
    }

    const std::size_t n = coarse.size.x * coarse.size.y;
    if (n == 0) {
        return RunSummary {
            std::max(coarse_steps, fine_steps),
            coarse_steps,
            fine_steps,
            critical_cells,
            critical_fraction,
            decision_ms,
            halo_calls,
            runtime_ms,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        };
    }

    double abs_sum = 0.0;
    double sq_sum = 0.0;
    double ref_sq_sum = 0.0;

    for (std::size_t idx = 0; idx < n; ++idx) {
        const double diff = fine.values[idx] - coarse.values[idx];
        abs_sum += std::abs(diff);
        sq_sum += diff * diff;
        ref_sq_sum += fine.values[idx] * fine.values[idx];
    }

    const double mae = abs_sum / static_cast<double>(n);
    const double global_error_norm = (ref_sq_sum > 0.0)
        ? std::sqrt(sq_sum) / std::sqrt(ref_sq_sum)
        : 0.0;

    return RunSummary {
        std::max(coarse_steps, fine_steps),
        coarse_steps,
        fine_steps,
        critical_cells,
        critical_fraction,
        decision_ms,
        halo_calls,
        runtime_ms,
        0.0,
        0.0,
        0.0,
        0.0,
        mae,
        global_error_norm,
    };
}

RunSummary summarize(const Field3D& coarse,
                     const Field3D& fine,
                     std::size_t coarse_steps,
                     std::size_t fine_steps,
                     std::size_t critical_cells,
                     double critical_fraction,
                     double decision_ms,
                     std::size_t halo_calls,
                     double runtime_ms) {
    if (coarse.size.x != fine.size.x || coarse.size.y != fine.size.y || coarse.size.z != fine.size.z) {
        throw std::runtime_error("Metric computation requires same field dimensions");
    }

    const std::size_t n = coarse.size.x * coarse.size.y * coarse.size.z;
    if (n == 0) {
        return RunSummary {
            std::max(coarse_steps, fine_steps),
            coarse_steps,
            fine_steps,
            critical_cells,
            critical_fraction,
            decision_ms,
            halo_calls,
            runtime_ms,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        };
    }

    double abs_sum = 0.0;
    double sq_sum = 0.0;
    double ref_sq_sum = 0.0;

    for (std::size_t idx = 0; idx < n; ++idx) {
        const double diff = fine.values[idx] - coarse.values[idx];
        abs_sum += std::abs(diff);
        sq_sum += diff * diff;
        ref_sq_sum += fine.values[idx] * fine.values[idx];
    }

    const double mae = abs_sum / static_cast<double>(n);
    const double global_error_norm = (ref_sq_sum > 0.0)
        ? std::sqrt(sq_sum) / std::sqrt(ref_sq_sum)
        : 0.0;

    return RunSummary {
        std::max(coarse_steps, fine_steps),
        coarse_steps,
        fine_steps,
        critical_cells,
        critical_fraction,
        decision_ms,
        halo_calls,
        runtime_ms,
        0.0,
        0.0,
        0.0,
        0.0,
        mae,
        global_error_norm,
    };
}

}  // namespace shardsim::metrics
