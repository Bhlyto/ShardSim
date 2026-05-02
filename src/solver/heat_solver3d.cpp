#include "shardsim/solver/heat_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "shardsim/decision_core/policy.hpp"
#include "shardsim/mpi/mpi_runtime.hpp"

#if SHARDSIM_HAS_MPI
#include <mpi.h>
#endif

namespace {

// Local x-major storage with ghost planes on both sides of x.
// Layout: index = i*(ny*nz) + j*nz + k
//   i = 0            : left ghost plane
//   i = 1..local_nx  : owned interior
//   i = local_nx+1   : right ghost plane
inline std::size_t local_idx3(std::size_t i, std::size_t j, std::size_t k,
                               std::size_t ny, std::size_t nz) {
    return i * (ny * nz) + j * nz + k;
}

// Global Field3D uses k-major: index = (k*ny + j)*nx + i
inline std::size_t global_idx3(std::size_t i, std::size_t j, std::size_t k,
                                std::size_t global_nx, std::size_t ny) {
    return (k * ny + j) * global_nx + i;
}

void apply_mixed_boundaries_local_3d(std::vector<double>& local,
                                      const shardsim::mpi_runtime::Partition1D& part,
                                      std::size_t global_nx,
                                      std::size_t ny,
                                      std::size_t nz) {
    // Y boundaries — every rank.
    for (std::size_t li = 1; li <= part.local_nx; ++li) {
        for (std::size_t k = 0; k < nz; ++k) {
            local[local_idx3(li, 0, k, ny, nz)] = 0.0;
            if (ny >= 2) {
                local[local_idx3(li, ny - 1, k, ny, nz)] =
                    local[local_idx3(li, ny - 2, k, ny, nz)];
            }
        }
    }
    // Z boundaries — every rank.
    for (std::size_t li = 1; li <= part.local_nx; ++li) {
        for (std::size_t j = 0; j < ny; ++j) {
            local[local_idx3(li, j, 0, ny, nz)] = 0.0;
            if (nz >= 2) {
                local[local_idx3(li, j, nz - 1, ny, nz)] =
                    local[local_idx3(li, j, nz - 2, ny, nz)];
            }
        }
    }
    // X left boundary — only the rank that owns global x == 0.
    if (part.global_x_begin == 0) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                local[local_idx3(1, j, k, ny, nz)] = 0.0;
            }
        }
    }
    // X right boundary — only the rank that owns global x == global_nx - 1.
    if (part.global_x_end == global_nx && part.local_nx >= 2) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                local[local_idx3(part.local_nx, j, k, ny, nz)] =
                    local[local_idx3(part.local_nx - 1, j, k, ny, nz)];
            }
        }
    }
}

shardsim::Field3D gather_global_field_3d(const std::vector<double>& local_with_ghosts,
                                          const shardsim::mpi_runtime::Partition1D& part,
                                          const shardsim::mpi_runtime::Context& ctx,
                                          std::size_t global_nx,
                                          std::size_t ny,
                                          std::size_t nz) {
    const std::size_t plane = ny * nz;

    // Strip ghosts: copy i=1..local_nx into x-major contiguous buffer.
    std::vector<double> local_no_ghost(part.local_nx * plane, 0.0);
    for (std::size_t li = 0; li < part.local_nx; ++li) {
        const std::size_t src_base = local_idx3(li + 1, 0, 0, ny, nz);
        std::copy(local_with_ghosts.begin() + static_cast<std::ptrdiff_t>(src_base),
                  local_with_ghosts.begin() + static_cast<std::ptrdiff_t>(src_base + plane),
                  local_no_ghost.begin() + static_cast<std::ptrdiff_t>(li * plane));
    }

    shardsim::Field3D out;
    out.size = {global_nx, ny, nz};
    out.values.assign(global_nx * ny * nz, 0.0);

    // Reorder from x-major gather buffer into Field3D's k-major layout.
    auto xmajor_to_field3d = [&](const std::vector<double>& src, std::size_t nx_src,
                                  std::size_t gi_offset) {
        for (std::size_t li = 0; li < nx_src; ++li) {
            const std::size_t gi = gi_offset + li;
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    out.values[global_idx3(gi, j, k, global_nx, ny)] =
                        src[li * plane + j * nz + k];
                }
            }
        }
    };

    if (ctx.world_size <= 1) {
        xmajor_to_field3d(local_no_ghost, part.local_nx, part.global_x_begin);
        return out;
    }

#if SHARDSIM_HAS_MPI
    std::vector<int> counts(static_cast<std::size_t>(ctx.world_size), 0);
    std::vector<int> displs(static_cast<std::size_t>(ctx.world_size), 0);
    for (int r = 0; r < ctx.world_size; ++r) {
        const auto rp = shardsim::mpi_runtime::make_strict_geometric_x_partition(
            global_nx, {r, ctx.world_size});
        counts[static_cast<std::size_t>(r)] = static_cast<int>(rp.local_nx * plane);
        displs[static_cast<std::size_t>(r)] = static_cast<int>(rp.global_x_begin * plane);
    }

    std::vector<double> global_xmajor;
    if (ctx.world_rank == 0) {
        global_xmajor.assign(global_nx * plane, 0.0);
    }

    MPI_Gatherv(local_no_ghost.data(),
                static_cast<int>(local_no_ghost.size()),
                MPI_DOUBLE,
                ctx.world_rank == 0 ? global_xmajor.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (ctx.world_rank == 0) {
        xmajor_to_field3d(global_xmajor, global_nx, 0);
    }

    MPI_Bcast(out.values.data(), static_cast<int>(out.values.size()), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
#endif
    return out;
}

std::pair<shardsim::Field3D, std::size_t> solve_explicit_until_tolerance_3d(
    const shardsim::mesh::Grid3D& grid,
    const shardsim::SimulationConfig& config,
    std::size_t max_steps,
    double dt,
    double alpha,
    double tolerance,
    const shardsim::mpi_runtime::Context& ctx,
    const shardsim::Field3D* initial_field,
    const std::vector<std::uint8_t>* active_mask) {
    const std::size_t global_nx = grid.size.x;
    const std::size_t ny = grid.size.y;
    const std::size_t nz = grid.size.z;

    const auto part = shardsim::mpi_runtime::make_strict_geometric_x_partition(global_nx, ctx);
    if (part.local_nx == 0 || ny < 2 || nz < 2) {
        throw std::runtime_error("3D solver requires all dimensions >= 2 and a valid partition");
    }

    const std::size_t plane = ny * nz;
    std::vector<double> current((part.local_nx + 2) * plane, 0.0);
    std::vector<double> next((part.local_nx + 2) * plane, 0.0);

    if (initial_field != nullptr) {
        if (initial_field->size.x != global_nx || initial_field->size.y != ny ||
            initial_field->size.z != nz) {
            throw std::runtime_error("3D initial field size does not match grid");
        }
        for (std::size_t li = 1; li <= part.local_nx; ++li) {
            const std::size_t gi = part.global_x_begin + (li - 1);
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    current[local_idx3(li, j, k, ny, nz)] =
                        initial_field->values[global_idx3(gi, j, k, global_nx, ny)];
                }
            }
        }
    } else {
        const std::size_t cx = static_cast<std::size_t>(
            std::clamp(config.source_x_fraction, 0.0, 1.0) * static_cast<double>(global_nx - 1));
        const std::size_t cy = static_cast<std::size_t>(
            std::clamp(config.source_y_fraction, 0.0, 1.0) * static_cast<double>(ny - 1));
        const std::size_t cz = static_cast<std::size_t>(
            std::clamp(config.source_z_fraction, 0.0, 1.0) * static_cast<double>(nz - 1));
        if (cx >= part.global_x_begin && cx < part.global_x_end && cy < ny && cz < nz) {
            const std::size_t li = (cx - part.global_x_begin) + 1;
            current[local_idx3(li, cy, cz, ny, nz)] = config.source_temperature;
        }
        if (config.source2_enabled) {
            const std::size_t cx2 = static_cast<std::size_t>(
                std::clamp(config.source2_x_fraction, 0.0, 1.0) *
                static_cast<double>(global_nx - 1));
            const std::size_t cy2 = static_cast<std::size_t>(
                std::clamp(config.source2_y_fraction, 0.0, 1.0) * static_cast<double>(ny - 1));
            const std::size_t cz2 = static_cast<std::size_t>(
                std::clamp(config.source2_z_fraction, 0.0, 1.0) * static_cast<double>(nz - 1));
            if (cx2 >= part.global_x_begin && cx2 < part.global_x_end && cy2 < ny && cz2 < nz) {
                const std::size_t li2 = (cx2 - part.global_x_begin) + 1;
                current[local_idx3(li2, cy2, cz2, ny, nz)] = config.source2_temperature;
            }
        }
    }

    // CFL stability: local min, then global AllReduce.
    double min_h2_local = 1.0e12;
    for (std::size_t li = 1; li <= part.local_nx; ++li) {
        const std::size_t gi = part.global_x_begin + (li - 1);
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            for (std::size_t k = 1; k + 1 < nz; ++k) {
                const double dx2 = grid.dx.at(gi, j, k) * grid.dx.at(gi, j, k);
                const double dy2 = grid.dy.at(gi, j, k) * grid.dy.at(gi, j, k);
                const double dz2 = grid.dz.at(gi, j, k) * grid.dz.at(gi, j, k);
                min_h2_local = std::min(min_h2_local, std::min(dx2, std::min(dy2, dz2)));
            }
        }
    }
    double min_h2 = min_h2_local;
#if SHARDSIM_HAS_MPI
    if (ctx.world_size > 1) {
        MPI_Allreduce(&min_h2_local, &min_h2, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    }
#endif

    const double alpha_abs = std::max(std::abs(alpha), 1.0e-12);
    const double stable_dt = 0.12 * min_h2 / alpha_abs;
    const double dt_used = std::min(dt, stable_dt);

    std::size_t used_steps = 0;
    double initial_rms_rate = -1.0;

    for (std::size_t n = 0; n < max_steps; ++n) {
        shardsim::mpi_runtime::exchange_halo_x_3d(current, part.local_nx, ny, nz, ctx);

        for (std::size_t li = 1; li <= part.local_nx; ++li) {
            const std::size_t gi = part.global_x_begin + (li - 1);
            for (std::size_t j = 1; j + 1 < ny; ++j) {
                for (std::size_t k = 1; k + 1 < nz; ++k) {
                    const auto flat_local = local_idx3(li, j, k, ny, nz);
                    if (active_mask != nullptr &&
                        (*active_mask)[global_idx3(gi, j, k, global_nx, ny)] == 0) {
                        next[flat_local] = current[flat_local];
                        continue;
                    }
                    const double t = current[flat_local];
                    const double d2x = current[local_idx3(li + 1, j, k, ny, nz)] - 2.0 * t +
                                       current[local_idx3(li - 1, j, k, ny, nz)];
                    const double d2y = current[local_idx3(li, j + 1, k, ny, nz)] - 2.0 * t +
                                       current[local_idx3(li, j - 1, k, ny, nz)];
                    const double d2z = current[local_idx3(li, j, k + 1, ny, nz)] - 2.0 * t +
                                       current[local_idx3(li, j, k - 1, ny, nz)];
                    const double lap =
                        d2x / (grid.dx.at(gi, j, k) * grid.dx.at(gi, j, k)) +
                        d2y / (grid.dy.at(gi, j, k) * grid.dy.at(gi, j, k)) +
                        d2z / (grid.dz.at(gi, j, k) * grid.dz.at(gi, j, k));
                    double updated = t + dt_used * alpha * lap;
                    if (!std::isfinite(updated)) {
                        updated = t;
                    }
                    next[flat_local] = std::clamp(updated, -1.0e9, 1.0e9);
                }
            }
        }
        // Keep ghost planes consistent (not updated by stencil).
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t k = 0; k < nz; ++k) {
                next[local_idx3(0, j, k, ny, nz)] = current[local_idx3(0, j, k, ny, nz)];
                next[local_idx3(part.local_nx + 1, j, k, ny, nz)] =
                    current[local_idx3(part.local_nx + 1, j, k, ny, nz)];
            }
        }

        apply_mixed_boundaries_local_3d(next, part, global_nx, ny, nz);

        double rate_sq_sum_local = 0.0;
        std::size_t rate_cells_local = 0;
        for (std::size_t li = 1; li <= part.local_nx; ++li) {
            const std::size_t gi = part.global_x_begin + (li - 1);
            if (gi == 0 || gi + 1 == global_nx) {
                continue;
            }
            for (std::size_t j = 1; j + 1 < ny; ++j) {
                for (std::size_t k = 1; k + 1 < nz; ++k) {
                    if (active_mask != nullptr &&
                        (*active_mask)[global_idx3(gi, j, k, global_nx, ny)] == 0) {
                        continue;
                    }
                    const auto flat_local = local_idx3(li, j, k, ny, nz);
                    const double transient_term =
                        (next[flat_local] - current[flat_local]) / dt_used;
                    const double bounded_rate = std::clamp(transient_term, -1.0e9, 1.0e9);
                    rate_sq_sum_local += bounded_rate * bounded_rate;
                    ++rate_cells_local;
                }
            }
        }

        std::swap(current, next);
        ++used_steps;

        double rate_sq_sum = rate_sq_sum_local;
        std::size_t rate_cells = rate_cells_local;
#if SHARDSIM_HAS_MPI
        if (ctx.world_size > 1) {
            MPI_Allreduce(&rate_sq_sum_local, &rate_sq_sum, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            std::size_t cells_global = rate_cells_local;
            MPI_Allreduce(&rate_cells_local, &cells_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD);
            rate_cells = cells_global;
        }
#endif
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
        const bool reached = (tolerance <= 1.0) ? (transient_ratio <= tolerance)
                                                 : (rms_rate <= tolerance);
        if (reached) {
            break;
        }
    }

    return {gather_global_field_3d(current, part, ctx, global_nx, ny, nz), used_steps};
}

}  // namespace

namespace shardsim::solver {

SolveResult3D run_transient_heat(const mesh::Grid3D& grid, const SimulationConfig& config) {
    SolveResult3D out;

    const auto ctx = shardsim::mpi_runtime::initialize();
    shardsim::mpi_runtime::reset_exchange_stats();

    if ((config.decision_policy == "surrogate_python" ||
         config.decision_policy == "surrogate_python_cached") &&
        ctx.world_size > 1) {
        throw std::runtime_error(
            "decision_policy=surrogate_python(_cached) currently supports only single-rank 3D runs");
    }

    auto coarse = solve_explicit_until_tolerance_3d(
        grid, config, config.steps, config.dt, config.alpha, config.coarse_tolerance, ctx,
        nullptr, nullptr);

    const auto select_begin = std::chrono::steady_clock::now();
    const auto selection =
        shardsim::decision_core::select_critical_regions(grid, coarse.first, config);
    const auto select_end = std::chrono::steady_clock::now();

    auto fine = solve_explicit_until_tolerance_3d(
        grid, config, config.steps, config.dt, config.alpha, config.fine_tolerance, ctx,
        &coarse.first, &selection.mask);

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
