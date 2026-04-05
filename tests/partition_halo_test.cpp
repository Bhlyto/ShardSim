#include <cstddef>
#include <iostream>
#include <vector>

#include "shardsim/mpi/mpi_runtime.hpp"

#if SHARDSIM_HAS_MPI
#include <mpi.h>
#endif

namespace {

std::size_t idx(std::size_t i, std::size_t j, std::size_t ny) {
    return i * ny + j;
}

int fail(const std::string& msg, int rank) {
    std::cerr << "[rank " << rank << "] test failure: " << msg << "\n";
    return 1;
}

}  // namespace

int main() {
    const auto ctx = shardsim::mpi_runtime::initialize();

    const std::size_t global_nx = 23;
    const auto part = shardsim::mpi_runtime::make_strict_geometric_x_partition(global_nx, ctx);

    std::size_t covered = part.local_nx;
#if SHARDSIM_HAS_MPI
    if (ctx.world_size > 1) {
        std::size_t covered_global = 0;
        MPI_Allreduce(&covered, &covered_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        covered = covered_global;
    }
#endif
    if (covered != global_nx) {
        shardsim::mpi_runtime::finalize();
        return fail("partition coverage mismatch", ctx.world_rank);
    }

    const std::size_t ny = 6;
    std::vector<double> local((part.local_nx + 2) * ny, 0.0);
    for (std::size_t i = 1; i <= part.local_nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            local[idx(i, j, ny)] = static_cast<double>(ctx.world_rank * 1000 + static_cast<int>(j));
        }
    }

    shardsim::mpi_runtime::exchange_halo_x(local, part.local_nx, ny, ctx);

    if (ctx.world_size > 1 && part.local_nx > 0) {
        for (std::size_t j = 0; j < ny; ++j) {
            const double left_expected = (ctx.world_rank > 0)
                ? static_cast<double>((ctx.world_rank - 1) * 1000 + static_cast<int>(j))
                : 0.0;
            const double right_expected = (ctx.world_rank + 1 < ctx.world_size)
                ? static_cast<double>((ctx.world_rank + 1) * 1000 + static_cast<int>(j))
                : 0.0;

            if (local[idx(0, j, ny)] != left_expected) {
                shardsim::mpi_runtime::finalize();
                return fail("left halo mismatch", ctx.world_rank);
            }
            if (local[idx(part.local_nx + 1, j, ny)] != right_expected) {
                shardsim::mpi_runtime::finalize();
                return fail("right halo mismatch", ctx.world_rank);
            }
        }
    }

    shardsim::mpi_runtime::finalize();
    return 0;
}
