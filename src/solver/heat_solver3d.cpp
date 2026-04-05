#include "shardsim/solver/heat_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "shardsim/decision_core/policy.hpp"

namespace {

std::size_t idx3(std::size_t i, std::size_t j, std::size_t k, std::size_t nx, std::size_t ny) {
    return (k * ny + j) * nx + i;
}

void apply_mixed_boundaries_3d(std::vector<double>& field,
                               std::size_t nx,
                               std::size_t ny,
                               std::size_t nz) {
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            field[idx3(0, j, k, nx, ny)] = 0.0;
            if (nx >= 2) {
                field[idx3(nx - 1, j, k, nx, ny)] = field[idx3(nx - 2, j, k, nx, ny)];
            }
        }
    }

    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t i = 0; i < nx; ++i) {
            field[idx3(i, 0, k, nx, ny)] = 0.0;
            if (ny >= 2) {
                field[idx3(i, ny - 1, k, nx, ny)] = field[idx3(i, ny - 2, k, nx, ny)];
            }
        }
    }

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            field[idx3(i, j, 0, nx, ny)] = 0.0;
            if (nz >= 2) {
                field[idx3(i, j, nz - 1, nx, ny)] = field[idx3(i, j, nz - 2, nx, ny)];
            }
        }
    }
}

std::pair<shardsim::Field3D, std::size_t> solve_explicit_until_tolerance_3d(
    const shardsim::mesh::Grid3D& grid,
    const shardsim::SimulationConfig& config,
    std::size_t max_steps,
    double dt,
    double alpha,
    double tolerance,
    const shardsim::Field3D* initial_field,
    const std::vector<std::uint8_t>* active_mask) {
    const std::size_t nx = grid.size.x;
    const std::size_t ny = grid.size.y;
    const std::size_t nz = grid.size.z;
    if (nx < 2 || ny < 2 || nz < 2) {
        throw std::runtime_error("3D solver requires all dimensions >= 2");
    }

    std::vector<double> current(nx * ny * nz, 0.0);
    std::vector<double> next(nx * ny * nz, 0.0);

    if (initial_field != nullptr) {
        if (initial_field->size.x != nx || initial_field->size.y != ny || initial_field->size.z != nz) {
            throw std::runtime_error("3D initial field size does not match grid");
        }
        current = initial_field->values;
    } else {
        const std::size_t cx = static_cast<std::size_t>(
            std::clamp(config.source_x_fraction, 0.0, 1.0) * static_cast<double>(nx - 1));
        const std::size_t cy = static_cast<std::size_t>(
            std::clamp(config.source_y_fraction, 0.0, 1.0) * static_cast<double>(ny - 1));
        const std::size_t cz = static_cast<std::size_t>(
            std::clamp(config.source_z_fraction, 0.0, 1.0) * static_cast<double>(nz - 1));
        current[idx3(cx, cy, cz, nx, ny)] = config.source_temperature;

        if (config.source2_enabled) {
            const std::size_t cx2 = static_cast<std::size_t>(
                std::clamp(config.source2_x_fraction, 0.0, 1.0) * static_cast<double>(nx - 1));
            const std::size_t cy2 = static_cast<std::size_t>(
                std::clamp(config.source2_y_fraction, 0.0, 1.0) * static_cast<double>(ny - 1));
            const std::size_t cz2 = static_cast<std::size_t>(
                std::clamp(config.source2_z_fraction, 0.0, 1.0) * static_cast<double>(nz - 1));
            current[idx3(cx2, cy2, cz2, nx, ny)] = config.source2_temperature;
        }
    }

    double min_h2 = 1.0e12;
    for (std::size_t k = 1; k + 1 < nz; ++k) {
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            for (std::size_t i = 1; i + 1 < nx; ++i) {
                const double dx2 = grid.dx.at(i, j, k) * grid.dx.at(i, j, k);
                const double dy2 = grid.dy.at(i, j, k) * grid.dy.at(i, j, k);
                const double dz2 = grid.dz.at(i, j, k) * grid.dz.at(i, j, k);
                min_h2 = std::min(min_h2, std::min(dx2, std::min(dy2, dz2)));
            }
        }
    }

    const double alpha_abs = std::max(std::abs(alpha), 1.0e-12);
    const double stable_dt = 0.12 * min_h2 / alpha_abs;
    const double dt_used = std::min(dt, stable_dt);

    std::size_t used_steps = 0;
    double initial_rms_rate = -1.0;

    for (std::size_t n = 0; n < max_steps; ++n) {
        for (std::size_t k = 1; k + 1 < nz; ++k) {
            for (std::size_t j = 1; j + 1 < ny; ++j) {
                for (std::size_t i = 1; i + 1 < nx; ++i) {
                    const auto flat = idx3(i, j, k, nx, ny);
                    if (active_mask != nullptr && (*active_mask)[flat] == 0) {
                        next[flat] = current[flat];
                        continue;
                    }

                    const double t = current[flat];
                    const double d2x = current[idx3(i + 1, j, k, nx, ny)] - 2.0 * t +
                                       current[idx3(i - 1, j, k, nx, ny)];
                    const double d2y = current[idx3(i, j + 1, k, nx, ny)] - 2.0 * t +
                                       current[idx3(i, j - 1, k, nx, ny)];
                    const double d2z = current[idx3(i, j, k + 1, nx, ny)] - 2.0 * t +
                                       current[idx3(i, j, k - 1, nx, ny)];
                    const double lap = d2x / (grid.dx.at(i, j, k) * grid.dx.at(i, j, k)) +
                                       d2y / (grid.dy.at(i, j, k) * grid.dy.at(i, j, k)) +
                                       d2z / (grid.dz.at(i, j, k) * grid.dz.at(i, j, k));

                    double updated = t + dt_used * alpha * lap;
                    if (!std::isfinite(updated)) {
                        updated = t;
                    }
                    next[flat] = std::clamp(updated, -1.0e9, 1.0e9);
                }
            }
        }

        apply_mixed_boundaries_3d(next, nx, ny, nz);

        double rate_sq_sum = 0.0;
        std::size_t rate_cells = 0;
        for (std::size_t k = 1; k + 1 < nz; ++k) {
            for (std::size_t j = 1; j + 1 < ny; ++j) {
                for (std::size_t i = 1; i + 1 < nx; ++i) {
                    const auto flat = idx3(i, j, k, nx, ny);
                    if (active_mask != nullptr && (*active_mask)[flat] == 0) {
                        continue;
                    }
                    const double transient_term = (next[flat] - current[flat]) / dt_used;
                    const double bounded_rate = std::clamp(transient_term, -1.0e9, 1.0e9);
                    rate_sq_sum += bounded_rate * bounded_rate;
                    ++rate_cells;
                }
            }
        }

        std::swap(current, next);
        ++used_steps;

        if (rate_cells == 0) {
            throw std::runtime_error("3D convergence metric received zero interior cells");
        }

        double rms_rate = std::sqrt(rate_sq_sum / static_cast<double>(rate_cells));
        if (!std::isfinite(rms_rate)) {
            rms_rate = 1.0e12;
        }
        if (initial_rms_rate < 0.0) {
            initial_rms_rate = std::max(rms_rate, 1.0e-12);
        }

        const double transient_ratio = rms_rate / initial_rms_rate;
        const bool reached = (tolerance <= 1.0)
            ? (transient_ratio <= tolerance)
            : (rms_rate <= tolerance);
        if (reached) {
            break;
        }
    }

    shardsim::Field3D out;
    out.size = {nx, ny, nz};
    out.values = std::move(current);
    return {std::move(out), used_steps};
}

}  // namespace

namespace shardsim::solver {

SolveResult3D run_transient_heat(const mesh::Grid3D& grid, const SimulationConfig& config) {
    if (config.decision_policy != "heuristic") {
        throw std::runtime_error("3D solver currently supports only decision_policy=heuristic");
    }

    SolveResult3D out;

    auto coarse = solve_explicit_until_tolerance_3d(
        grid, config, config.steps, config.dt, config.alpha, config.coarse_tolerance, nullptr, nullptr);

    const auto select_begin = std::chrono::steady_clock::now();
    const auto selection = shardsim::decision_core::select_critical_regions(grid, coarse.first, config);
    const auto select_end = std::chrono::steady_clock::now();
    auto fine = solve_explicit_until_tolerance_3d(
        grid,
        config,
        config.steps,
        config.dt,
        config.alpha,
        config.fine_tolerance,
        &coarse.first,
        &selection.mask);

    out.coarse = std::move(coarse.first);
    out.fine = std::move(fine.first);
    out.coarse_steps = coarse.second;
    out.fine_steps = fine.second;
    out.critical_cells = selection.critical_cells;
    out.critical_fraction = selection.critical_fraction;
    out.decision_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        select_end - select_begin).count();
    return out;
}

}  // namespace shardsim::solver