#include "shardsim/solver/heat_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "shardsim/correction/correction.hpp"
#include "shardsim/decision_core/policy.hpp"
#include "shardsim/mpi/mpi_runtime.hpp"

#if SHARDSIM_HAS_MPI
#include <mpi.h>
#endif

namespace {

std::size_t local_idx(std::size_t i, std::size_t j, std::size_t ny) {
    return i * ny + j;
}

void apply_mixed_boundaries_local(std::vector<double>& local,
                                  const shardsim::mpi_runtime::Partition1D& part,
                                  std::size_t global_nx,
                                  std::size_t ny) {
    for (std::size_t i = 1; i <= part.local_nx; ++i) {
        local[local_idx(i, 0, ny)] = 0.0;
        if (ny >= 2) {
            local[local_idx(i, ny - 1, ny)] = local[local_idx(i, ny - 2, ny)];
        }
    }

    for (std::size_t i = 1; i <= part.local_nx; ++i) {
        const std::size_t gi = part.global_x_begin + (i - 1);
        if (gi == 0) {
            for (std::size_t j = 0; j < ny; ++j) {
                local[local_idx(i, j, ny)] = 0.0;
            }
        }
        if (gi + 1 == global_nx) {
            for (std::size_t j = 0; j < ny; ++j) {
                local[local_idx(i, j, ny)] = local[local_idx(i - 1, j, ny)];
            }
        }
    }
}

shardsim::Field2D gather_global_field(const std::vector<double>& local,
                                      const shardsim::mpi_runtime::Partition1D& part,
                                      const shardsim::mpi_runtime::Context& ctx,
                                      std::size_t global_nx,
                                      std::size_t ny) {
    std::vector<double> local_no_ghost(part.local_nx * ny, 0.0);
    for (std::size_t i = 0; i < part.local_nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            local_no_ghost[i * ny + j] = local[local_idx(i + 1, j, ny)];
        }
    }

    shardsim::Field2D out;
    out.size = {global_nx, ny};
    out.values.assign(global_nx * ny, 0.0);

    if (ctx.world_size <= 1) {
        for (std::size_t i = 0; i < global_nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                out.at(i, j) = local_no_ghost[i * ny + j];
            }
        }
        return out;
    }

#if SHARDSIM_HAS_MPI
    std::vector<int> counts(static_cast<std::size_t>(ctx.world_size), 0);
    std::vector<int> displs(static_cast<std::size_t>(ctx.world_size), 0);
    for (int r = 0; r < ctx.world_size; ++r) {
        const auto rp = shardsim::mpi_runtime::make_strict_geometric_x_partition(global_nx, {r, ctx.world_size});
        counts[static_cast<std::size_t>(r)] = static_cast<int>(rp.local_nx * ny);
        displs[static_cast<std::size_t>(r)] = static_cast<int>(rp.global_x_begin * ny);
    }

    std::vector<double> global_xmajor;
    if (ctx.world_rank == 0) {
        global_xmajor.assign(global_nx * ny, 0.0);
    }

    MPI_Gatherv(local_no_ghost.data(),
                static_cast<int>(local_no_ghost.size()),
                MPI_DOUBLE,
                (ctx.world_rank == 0) ? global_xmajor.data() : nullptr,
                counts.data(),
                displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (ctx.world_rank == 0) {
        for (std::size_t i = 0; i < global_nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                out.at(i, j) = global_xmajor[i * ny + j];
            }
        }
    }

    MPI_Bcast(out.values.data(), static_cast<int>(out.values.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    return out;
}

std::pair<shardsim::Field2D, std::size_t> solve_explicit_until_tolerance(
    const shardsim::mesh::Grid2D& grid,
    const shardsim::SimulationConfig& config,
    const std::size_t max_steps,
    const double dt,
    const double alpha,
    const double tolerance,
    const shardsim::mpi_runtime::Context& ctx,
    const shardsim::Field2D* initial_field,
    const std::vector<std::uint8_t>* active_mask) {
    const auto part = shardsim::mpi_runtime::make_strict_geometric_x_partition(grid.size.x, ctx);
    const std::size_t ny = grid.size.y;
    if (part.local_nx == 0 || ny < 2) {
        throw std::runtime_error("Invalid partition or grid size for solver");
    }

    std::vector<double> current((part.local_nx + 2) * ny, 0.0);
    std::vector<double> next((part.local_nx + 2) * ny, 0.0);

    if (initial_field != nullptr) {
        if (initial_field->size.x != grid.size.x || initial_field->size.y != grid.size.y) {
            throw std::runtime_error("Initial field size does not match grid");
        }
        for (std::size_t li = 1; li <= part.local_nx; ++li) {
            const std::size_t gi = part.global_x_begin + (li - 1);
            for (std::size_t j = 0; j < ny; ++j) {
                current[local_idx(li, j, ny)] = initial_field->at(gi, j);
            }
        }
    } else {
        const double x_frac = std::clamp(config.source_x_fraction, 0.0, 1.0);
        const double y_frac = std::clamp(config.source_y_fraction, 0.0, 1.0);
        const std::size_t cx = static_cast<std::size_t>(x_frac * static_cast<double>(grid.size.x - 1));
        const std::size_t cy = static_cast<std::size_t>(y_frac * static_cast<double>(ny - 1));
        if (cx >= part.global_x_begin && cx < part.global_x_end && cy < ny) {
            const std::size_t li = (cx - part.global_x_begin) + 1;
            current[local_idx(li, cy, ny)] = config.source_temperature;
        }

        if (config.source2_enabled) {
            const double x2_frac = std::clamp(config.source2_x_fraction, 0.0, 1.0);
            const double y2_frac = std::clamp(config.source2_y_fraction, 0.0, 1.0);
            const std::size_t cx2 =
                static_cast<std::size_t>(x2_frac * static_cast<double>(grid.size.x - 1));
            const std::size_t cy2 = static_cast<std::size_t>(y2_frac * static_cast<double>(ny - 1));
            if (cx2 >= part.global_x_begin && cx2 < part.global_x_end && cy2 < ny) {
                const std::size_t li2 = (cx2 - part.global_x_begin) + 1;
                current[local_idx(li2, cy2, ny)] = config.source2_temperature;
            }
        }
    }

    double min_h2_local = 1.0e12;
    for (std::size_t li = 1; li <= part.local_nx; ++li) {
        const std::size_t gi = part.global_x_begin + (li - 1);
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            const double dx2 = grid.dx.at(gi, j) * grid.dx.at(gi, j);
            const double dy2 = grid.dy.at(gi, j) * grid.dy.at(gi, j);
            min_h2_local = std::min(min_h2_local, std::min(dx2, dy2));
        }
    }

    double min_h2 = min_h2_local;
#if SHARDSIM_HAS_MPI
    if (ctx.world_size > 1) {
        MPI_Allreduce(&min_h2_local, &min_h2, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    }
#endif

    const double alpha_abs = std::max(std::abs(alpha), 1.0e-12);
    const double stable_dt = 0.24 * min_h2 / alpha_abs;
    const double dt_used = std::min(dt, stable_dt);

    std::size_t used_steps = 0;
    double initial_rms_rate = -1.0;

    for (std::size_t n = 0; n < max_steps; ++n) {
        shardsim::mpi_runtime::exchange_halo_x(current, part.local_nx, ny, ctx);

        for (std::size_t li = 1; li <= part.local_nx; ++li) {
            const std::size_t gi = part.global_x_begin + (li - 1);
            for (std::size_t j = 1; j + 1 < ny; ++j) {
                if (active_mask != nullptr && (*active_mask)[j * grid.size.x + gi] == 0) {
                    next[local_idx(li, j, ny)] = current[local_idx(li, j, ny)];
                    continue;
                }
                const double t = current[local_idx(li, j, ny)];
                const double d2x = current[local_idx(li + 1, j, ny)] - 2.0 * t +
                                   current[local_idx(li - 1, j, ny)];
                const double d2y = current[local_idx(li, j + 1, ny)] - 2.0 * t +
                                   current[local_idx(li, j - 1, ny)];
                const double lap = d2x / (grid.dx.at(gi, j) * grid.dx.at(gi, j)) +
                                   d2y / (grid.dy.at(gi, j) * grid.dy.at(gi, j));
                double updated = t + dt_used * alpha * lap;
                if (!std::isfinite(updated)) {
                    updated = t;
                }
                updated = std::clamp(updated, -1.0e9, 1.0e9);
                next[local_idx(li, j, ny)] = updated;
            }
        }

        apply_mixed_boundaries_local(next, part, grid.size.x, ny);

        double rate_sq_sum_local = 0.0;
        std::size_t rate_cells_local = 0;
        for (std::size_t li = 1; li <= part.local_nx; ++li) {
            const std::size_t gi = part.global_x_begin + (li - 1);
            if (gi == 0 || gi + 1 == grid.size.x) {
                continue;
            }
            for (std::size_t j = 1; j + 1 < ny; ++j) {
                if (active_mask != nullptr && (*active_mask)[j * grid.size.x + gi] == 0) {
                    continue;
                }
                const double t_prev = current[local_idx(li, j, ny)];
                const double t_next = next[local_idx(li, j, ny)];
                double transient_term = (t_next - t_prev) / dt_used;
                if (!std::isfinite(transient_term)) {
                    transient_term = 0.0;
                }
                const double bounded_rate = std::clamp(transient_term, -1.0e9, 1.0e9);
                rate_sq_sum_local += bounded_rate * bounded_rate;
                ++rate_cells_local;
            }
        }

        std::swap(current, next);
        ++used_steps;

        double rate_sq_sum = rate_sq_sum_local;
        std::size_t rate_cells = rate_cells_local;
#if SHARDSIM_HAS_MPI
        if (ctx.world_size > 1) {
            MPI_Allreduce(&rate_sq_sum_local, &rate_sq_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            std::size_t cells_global = rate_cells_local;
            MPI_Allreduce(&rate_cells_local,
                          &cells_global,
                          1,
                          MPI_UNSIGNED_LONG_LONG,
                          MPI_SUM,
                          MPI_COMM_WORLD);
            rate_cells = cells_global;
        }
#endif

        if (rate_cells == 0) {
            throw std::runtime_error("Convergence metric received zero interior cells");
        }

        const double denom = static_cast<double>(rate_cells);
        double rms_rate = std::sqrt(rate_sq_sum / denom);
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

    return {gather_global_field(current, part, ctx, grid.size.x, ny), used_steps};
}

}  // namespace

namespace shardsim::solver {

SolveResult run_transient_heat(const mesh::Grid2D& grid, const SimulationConfig& config) {
    SolveResult out;

    const auto ctx = shardsim::mpi_runtime::initialize();
    shardsim::mpi_runtime::reset_exchange_stats();

    if ((config.decision_policy == "surrogate_python" ||
         config.decision_policy == "surrogate_python_cached") &&
        ctx.world_size > 1) {
        throw std::runtime_error(
            "decision_policy=surrogate_python(_cached) currently supports only single-rank runs");
    }

    auto coarse = solve_explicit_until_tolerance(
        grid, config, config.steps, config.dt, config.alpha, config.coarse_tolerance, ctx, nullptr, nullptr);

    // Phase 4: ML correction loop.
    // When correction_policy != "none", apply the trained correction model to
    // the coarse field instead of running the full fine solve.
    if (config.correction_policy != "none" && !config.correction_policy.empty()) {
        const auto select_begin = std::chrono::steady_clock::now();
        const auto selection = shardsim::decision_core::select_critical_regions(grid, coarse.first, config);
        const auto select_end = std::chrono::steady_clock::now();

        out.coarse = coarse.first;
        out.fine = shardsim::correction::apply_correction(coarse.first, config);
        out.coarse_steps = coarse.second;
        out.fine_steps = 0;
        out.critical_cells = selection.critical_cells;
        out.critical_fraction = selection.critical_fraction;
        out.decision_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            select_end - select_begin).count();
        out.correction_applied = true;

        const auto halo_stats = shardsim::mpi_runtime::collect_exchange_stats(ctx);
        out.halo_calls = halo_stats.calls;
        out.halo_ms_local = halo_stats.local_ms;
        out.halo_ms_min = halo_stats.min_ms;
        out.halo_ms_avg = halo_stats.avg_ms;
        out.halo_ms_max = halo_stats.max_ms;
        return out;
    }

    const auto select_begin = std::chrono::steady_clock::now();
    const auto selection = shardsim::decision_core::select_critical_regions(grid, coarse.first, config);
    const auto select_end = std::chrono::steady_clock::now();
    auto fine = solve_explicit_until_tolerance(
        grid,
        config,
        config.steps,
        config.dt,
        config.alpha,
        config.fine_tolerance,
        ctx,
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

    const auto halo_stats = shardsim::mpi_runtime::collect_exchange_stats(ctx);
    out.halo_calls = halo_stats.calls;
    out.halo_ms_local = halo_stats.local_ms;
    out.halo_ms_min = halo_stats.min_ms;
    out.halo_ms_avg = halo_stats.avg_ms;
    out.halo_ms_max = halo_stats.max_ms;
    return out;
}

}  // namespace shardsim::solver
