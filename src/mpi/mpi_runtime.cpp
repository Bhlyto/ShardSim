#include "shardsim/mpi/mpi_runtime.hpp"

#include <algorithm>
#include <chrono>

#if SHARDSIM_HAS_MPI
#include <mpi.h>
#endif

namespace shardsim::mpi_runtime {

namespace {

std::size_t g_exchange_calls = 0;
double g_exchange_time_ms = 0.0;

}  // namespace

Context initialize() {
    Context ctx {};

#if SHARDSIM_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized == 0) {
        MPI_Init(nullptr, nullptr);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.world_size);
#endif

    return ctx;
}

void finalize() {
#if SHARDSIM_HAS_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) {
        MPI_Finalize();
    }
#endif
}

Partition1D make_strict_geometric_x_partition(std::size_t global_nx, const Context& ctx) {
    Partition1D part {};

    const std::size_t size = (ctx.world_size > 0) ? static_cast<std::size_t>(ctx.world_size) : 1;
    const std::size_t rank = (ctx.world_rank >= 0) ? static_cast<std::size_t>(ctx.world_rank) : 0;
    const std::size_t base = global_nx / size;
    const std::size_t rem = global_nx % size;

    part.local_nx = base + ((rank < rem) ? 1 : 0);
    part.global_x_begin = rank * base + std::min(rank, rem);
    part.global_x_end = part.global_x_begin + part.local_nx;
    return part;
}

void exchange_halo_x(std::vector<double>& local_with_ghosts,
                     std::size_t local_nx,
                     std::size_t ny,
                     const Context& ctx) {
    if (local_nx == 0 || ny == 0 || ctx.world_size <= 1) {
        return;
    }

#if SHARDSIM_HAS_MPI
    const auto t0 = std::chrono::steady_clock::now();

    auto idx = [ny](std::size_t i, std::size_t j) {
        return i * ny + j;
    };

    std::vector<double> send_left(ny, 0.0);
    std::vector<double> send_right(ny, 0.0);
    std::vector<double> recv_left(ny, 0.0);
    std::vector<double> recv_right(ny, 0.0);

    for (std::size_t j = 0; j < ny; ++j) {
        send_left[j] = local_with_ghosts[idx(1, j)];
        send_right[j] = local_with_ghosts[idx(local_nx, j)];
    }

    const int left_rank = (ctx.world_rank > 0) ? (ctx.world_rank - 1) : MPI_PROC_NULL;
    const int right_rank = (ctx.world_rank + 1 < ctx.world_size) ? (ctx.world_rank + 1) : MPI_PROC_NULL;

    MPI_Sendrecv(send_left.data(),
                 static_cast<int>(ny),
                 MPI_DOUBLE,
                 left_rank,
                 100,
                 recv_right.data(),
                 static_cast<int>(ny),
                 MPI_DOUBLE,
                 right_rank,
                 100,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_right.data(),
                 static_cast<int>(ny),
                 MPI_DOUBLE,
                 right_rank,
                 101,
                 recv_left.data(),
                 static_cast<int>(ny),
                 MPI_DOUBLE,
                 left_rank,
                 101,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    for (std::size_t j = 0; j < ny; ++j) {
        local_with_ghosts[idx(0, j)] = recv_left[j];
        local_with_ghosts[idx(local_nx + 1, j)] = recv_right[j];
    }

    const auto t1 = std::chrono::steady_clock::now();
    g_exchange_time_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    ++g_exchange_calls;
#else
    (void)local_with_ghosts;
    (void)local_nx;
    (void)ny;
    (void)ctx;
#endif
}

void exchange_halo_x_3d(std::vector<double>& local_with_ghosts,
                        std::size_t local_nx,
                        std::size_t ny,
                        std::size_t nz,
                        const Context& ctx) {
    if (local_nx == 0 || ny == 0 || nz == 0 || ctx.world_size <= 1) {
        return;
    }

#if SHARDSIM_HAS_MPI
    const auto t0 = std::chrono::steady_clock::now();

    // Local storage is x-major: plane size = ny*nz.
    // Left ghost i=0  → offset 0          (size ny*nz, contiguous)
    // Left inner i=1  → offset ny*nz      (contiguous)
    // Right inner i=local_nx → offset local_nx*ny*nz (contiguous)
    // Right ghost i=local_nx+1 → offset (local_nx+1)*ny*nz (contiguous)
    const std::size_t plane = ny * nz;

    const int left_rank  = (ctx.world_rank > 0)                          ? (ctx.world_rank - 1) : MPI_PROC_NULL;
    const int right_rank = (ctx.world_rank + 1 < ctx.world_size)         ? (ctx.world_rank + 1) : MPI_PROC_NULL;

    // Send left inner plane to left neighbour; receive our right ghost from right neighbour.
    MPI_Sendrecv(local_with_ghosts.data() + plane,
                 static_cast<int>(plane), MPI_DOUBLE, left_rank,  100,
                 local_with_ghosts.data() + (local_nx + 1) * plane,
                 static_cast<int>(plane), MPI_DOUBLE, right_rank, 100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send right inner plane to right neighbour; receive our left ghost from left neighbour.
    MPI_Sendrecv(local_with_ghosts.data() + local_nx * plane,
                 static_cast<int>(plane), MPI_DOUBLE, right_rank, 101,
                 local_with_ghosts.data(),
                 static_cast<int>(plane), MPI_DOUBLE, left_rank,  101,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    const auto t1 = std::chrono::steady_clock::now();
    g_exchange_time_ms +=
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    ++g_exchange_calls;
#else
    (void)local_with_ghosts;
    (void)local_nx;
    (void)ny;
    (void)nz;
    (void)ctx;
#endif
}

void reset_exchange_stats() {
    g_exchange_calls = 0;
    g_exchange_time_ms = 0.0;
}

ExchangeStats collect_exchange_stats(const Context& ctx) {
    ExchangeStats out {};
    out.calls = g_exchange_calls;
    out.local_ms = g_exchange_time_ms;

    if (ctx.world_size <= 1) {
        out.min_ms = out.local_ms;
        out.avg_ms = out.local_ms;
        out.max_ms = out.local_ms;
        return out;
    }

#if SHARDSIM_HAS_MPI
    double min_ms = out.local_ms;
    double sum_ms = out.local_ms;
    double max_ms = out.local_ms;
    MPI_Allreduce(&out.local_ms, &min_ms, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&out.local_ms, &sum_ms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&out.local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    out.min_ms = min_ms;
    out.avg_ms = sum_ms / static_cast<double>(ctx.world_size);
    out.max_ms = max_ms;
#else
    out.min_ms = out.local_ms;
    out.avg_ms = out.local_ms;
    out.max_ms = out.local_ms;
#endif
    return out;
}

}  // namespace shardsim::mpi_runtime
