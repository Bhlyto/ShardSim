#pragma once

#include <cstddef>
#include <vector>

namespace shardsim::mpi_runtime {

struct Context {
    int world_rank {0};
    int world_size {1};
};

struct Partition1D {
    std::size_t global_x_begin {0};
    std::size_t global_x_end {0};
    std::size_t local_nx {0};
};

struct ExchangeStats {
    std::size_t calls {0};
    double local_ms {0.0};
    double min_ms {0.0};
    double avg_ms {0.0};
    double max_ms {0.0};
};

Context initialize();
void finalize();

Partition1D make_strict_geometric_x_partition(std::size_t global_nx, const Context& ctx);
void exchange_halo_x(std::vector<double>& local_with_ghosts,
                     std::size_t local_nx,
                     std::size_t ny,
                     const Context& ctx);
void reset_exchange_stats();
ExchangeStats collect_exchange_stats(const Context& ctx);

}  // namespace shardsim::mpi_runtime
