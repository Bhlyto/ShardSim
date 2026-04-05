#include "shardsim/orchestrator/orchestrator.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "shardsim/mesh/mesh.hpp"
#include "shardsim/mesh/mesh3d.hpp"
#include "shardsim/metrics/metrics.hpp"
#include "shardsim/mpi/mpi_runtime.hpp"
#include "shardsim/solver/heat_solver.hpp"

namespace shardsim::orchestrator {

namespace {

using Clock = std::chrono::system_clock;

constexpr std::uint32_t kTrainingMagic = 0x31534453;   // "SDS1"
constexpr std::uint32_t kTrainingVersion = 1;
constexpr std::uint32_t kTrainingMagic3D = 0x33534453; // "SDS3"
constexpr std::uint32_t kTrainingVersion3D = 1;

void write_u32(std::ostream& out, std::uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("Failed to write u32 to training data file");
    }
}

void write_u64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("Failed to write u64 to training data file");
    }
}

void write_f64(std::ostream& out, double value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("Failed to write f64 to training data file");
    }
}

void write_field(std::ostream& out, const shardsim::Field2D& field) {
    if (field.values.size() != field.size.x * field.size.y) {
        throw std::runtime_error("Invalid field shape while exporting training data");
    }

    const std::uint64_t count = static_cast<std::uint64_t>(field.values.size());
    write_u64(out, count);
    out.write(reinterpret_cast<const char*>(field.values.data()),
              static_cast<std::streamsize>(count * sizeof(double)));
    if (!out) {
        throw std::runtime_error("Failed to write field payload to training data file");
    }
}

void write_field(std::ostream& out, const shardsim::Field3D& field) {
    if (field.values.size() != field.size.x * field.size.y * field.size.z) {
        throw std::runtime_error("Invalid 3D field shape while exporting training data");
    }

    const std::uint64_t count = static_cast<std::uint64_t>(field.values.size());
    write_u64(out, count);
    out.write(reinterpret_cast<const char*>(field.values.data()),
              static_cast<std::streamsize>(count * sizeof(double)));
    if (!out) {
        throw std::runtime_error("Failed to write 3D field payload to training data file");
    }
}

void maybe_export_training_sample(const shardsim::SimulationConfig& config,
                                  const shardsim::solver::SolveResult& solve_result,
                                  const shardsim::mpi_runtime::Context& ctx) {
    if (!config.export_training_data || ctx.world_rank != 0) {
        return;
    }

    std::filesystem::path out_dir = config.training_data_export_dir.empty()
        ? std::filesystem::path("runs/training_data")
        : std::filesystem::path(config.training_data_export_dir);
    std::filesystem::create_directories(out_dir);

    const auto now = Clock::now();
    const auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    const auto filename = std::string("sample_") + std::to_string(ts_us) + "_r" +
        std::to_string(ctx.world_size) + "_g" +
        std::to_string(solve_result.coarse.size.x) + "x" +
        std::to_string(solve_result.coarse.size.y) + ".bin";
    const auto out_path = out_dir / filename;

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open training export path: " + out_path.string());
    }

    write_u32(out, kTrainingMagic);
    write_u32(out, kTrainingVersion);
    write_u64(out, static_cast<std::uint64_t>(solve_result.coarse.size.x));
    write_u64(out, static_cast<std::uint64_t>(solve_result.coarse.size.y));
    write_u64(out, static_cast<std::uint64_t>(solve_result.coarse_steps));
    write_u64(out, static_cast<std::uint64_t>(solve_result.fine_steps));
    write_f64(out, solve_result.critical_fraction);
    write_field(out, solve_result.coarse);
    write_field(out, solve_result.fine);
}

void maybe_export_training_sample(const shardsim::SimulationConfig& config,
                                  const shardsim::solver::SolveResult3D& solve_result,
                                  const shardsim::mpi_runtime::Context& ctx) {
    if (!config.export_training_data || ctx.world_rank != 0) {
        return;
    }

    std::filesystem::path out_dir = config.training_data_export_dir.empty()
        ? std::filesystem::path("runs/training_data_3d")
        : std::filesystem::path(config.training_data_export_dir);
    std::filesystem::create_directories(out_dir);

    const auto now = Clock::now();
    const auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    const auto filename = std::string("sample_") + std::to_string(ts_us) + "_r" +
        std::to_string(ctx.world_size) + "_g" +
        std::to_string(solve_result.coarse.size.x) + "x" +
        std::to_string(solve_result.coarse.size.y) + "x" +
        std::to_string(solve_result.coarse.size.z) + ".bin";
    const auto out_path = out_dir / filename;

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open 3D training export path: " + out_path.string());
    }

    write_u32(out, kTrainingMagic3D);
    write_u32(out, kTrainingVersion3D);
    write_u64(out, static_cast<std::uint64_t>(solve_result.coarse.size.x));
    write_u64(out, static_cast<std::uint64_t>(solve_result.coarse.size.y));
    write_u64(out, static_cast<std::uint64_t>(solve_result.coarse.size.z));
    write_u64(out, static_cast<std::uint64_t>(solve_result.coarse_steps));
    write_u64(out, static_cast<std::uint64_t>(solve_result.fine_steps));
    write_f64(out, solve_result.critical_fraction);
    write_field(out, solve_result.coarse);
    write_field(out, solve_result.fine);
}

std::optional<double> read_rss_mb_linux() {
    std::ifstream in("/proc/self/status");
    if (!in) {
        return std::nullopt;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line);
            std::string key;
            double rss_kb = 0.0;
            std::string unit;
            iss >> key >> rss_kb >> unit;
            if (!iss.fail() && rss_kb >= 0.0) {
                return rss_kb / 1024.0;
            }
            return std::nullopt;
        }
    }

    return std::nullopt;
}

}  // namespace

Orchestrator::Orchestrator(SimulationConfig config) : config_(std::move(config)) {}

metrics::RunSummary Orchestrator::run() const {
    if (config_.partitioning_policy != "strict_geometric") {
        throw std::runtime_error("Only strict_geometric partitioning policy is currently supported");
    }

    const auto mpi_ctx = mpi_runtime::initialize();
    (void)mpi_ctx;

    try {
        const auto t0 = std::chrono::steady_clock::now();
        metrics::RunSummary summary;
        if (config_.grid_z > 1) {
            if (mpi_ctx.world_size > 1) {
                throw std::runtime_error("3D solver path currently supports only single-rank runs");
            }
            const auto grid = mesh::make_non_uniform_grid3d(config_);
            const auto solve_result = solver::run_transient_heat(grid, config_);
            maybe_export_training_sample(config_, solve_result, mpi_ctx);
            const auto t1 = std::chrono::steady_clock::now();
            const auto runtime_ms =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
            summary = metrics::summarize(
                solve_result.coarse,
                solve_result.fine,
                solve_result.coarse_steps,
                solve_result.fine_steps,
                solve_result.critical_cells,
                solve_result.critical_fraction,
                solve_result.decision_ms,
                solve_result.halo_calls,
                runtime_ms);
            summary.halo_ms_local = solve_result.halo_ms_local;
            summary.halo_ms_min = solve_result.halo_ms_min;
            summary.halo_ms_avg = solve_result.halo_ms_avg;
            summary.halo_ms_max = solve_result.halo_ms_max;
        } else {
            const auto grid = mesh::make_non_uniform_grid(config_);
            const auto solve_result = solver::run_transient_heat(grid, config_);
            maybe_export_training_sample(config_, solve_result, mpi_ctx);
            const auto t1 = std::chrono::steady_clock::now();
            const auto runtime_ms =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
            summary = metrics::summarize(
                solve_result.coarse,
                solve_result.fine,
                solve_result.coarse_steps,
                solve_result.fine_steps,
                solve_result.critical_cells,
                solve_result.critical_fraction,
                solve_result.decision_ms,
                solve_result.halo_calls,
                runtime_ms);
            summary.halo_ms_local = solve_result.halo_ms_local;
            summary.halo_ms_min = solve_result.halo_ms_min;
            summary.halo_ms_avg = solve_result.halo_ms_avg;
            summary.halo_ms_max = solve_result.halo_ms_max;
        }

        const double wallclock_limit_ms = (config_.wallclock_limit_ms > 0)
            ? static_cast<double>(config_.wallclock_limit_ms)
            : static_cast<double>(config_.wallclock_limit_minutes) * 60.0 * 1000.0;
        if (wallclock_limit_ms > 0.0 && summary.runtime_ms > wallclock_limit_ms) {
            throw std::runtime_error("Wall-clock limit exceeded");
        }

        const double memory_limit_mb = (config_.memory_ceiling_mb > 0)
            ? static_cast<double>(config_.memory_ceiling_mb)
            : static_cast<double>(config_.memory_ceiling_gb) * 1024.0;
        if (memory_limit_mb > 0.0) {
            const auto rss_mb = read_rss_mb_linux();
            if (rss_mb.has_value() && rss_mb.value() > memory_limit_mb) {
                throw std::runtime_error("Memory ceiling exceeded");
            }
        }

        if (config_.halo_overhead_ratio_max > 0.0) {
            const double runtime = (summary.runtime_ms > 1.0e-12) ? summary.runtime_ms : 1.0e-12;
            const double ratio = summary.halo_ms_avg / runtime;
            if (ratio > config_.halo_overhead_ratio_max) {
                throw std::runtime_error("Halo overhead ratio exceeded configured maximum");
            }
        }

        mpi_runtime::finalize();
        return summary;
    } catch (...) {
        mpi_runtime::finalize();
        throw;
    }
}

}  // namespace shardsim::orchestrator
