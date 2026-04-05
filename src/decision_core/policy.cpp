#include "shardsim/decision_core/policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace shardsim::decision_core {

namespace {

constexpr std::uint32_t kCoarseFieldMagic2D = 0x31434453;  // "SDC1"
constexpr std::uint32_t kCoarseFieldVersion2D = 1;
constexpr std::uint32_t kMaskMagic2D = 0x314d4453;         // "SDM1"
constexpr std::uint32_t kMaskVersion2D = 1;

struct LinearPatchModel2D {
    std::size_t patch_radius {0};
    double bias {0.0};
    std::vector<double> feature_mean;
    std::vector<double> feature_scale;
    std::vector<double> weights;
};

void write_u32(std::ostream& out, std::uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("Failed to write u32 payload");
    }
}

void write_u64(std::ostream& out, std::uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("Failed to write u64 payload");
    }
}

std::uint32_t read_u32(std::istream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("Failed to read u32 payload");
    }
    return value;
}

std::uint64_t read_u64(std::istream& in) {
    std::uint64_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("Failed to read u64 payload");
    }
    return value;
}

std::string shell_quote(const std::string& value) {
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('\'');
    for (const char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::vector<double> parse_csv_doubles(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        values.push_back(std::stod(token));
    }
    return values;
}

LinearPatchModel2D load_linear_patch_model(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open linear surrogate model: " + path);
    }

    LinearPatchModel2D model;
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const auto key = line.substr(0, colon);
        const auto value = line.substr(colon + 1);
        const auto trimmed = [&value]() {
            std::size_t start = 0;
            while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
                ++start;
            }
            std::size_t end = value.size();
            while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
                --end;
            }
            return value.substr(start, end - start);
        }();

        if (key == "patch_radius") {
            model.patch_radius = static_cast<std::size_t>(std::stoull(trimmed));
        } else if (key == "bias") {
            model.bias = std::stod(trimmed);
        } else if (key == "feature_mean") {
            model.feature_mean = parse_csv_doubles(trimmed);
        } else if (key == "feature_scale") {
            model.feature_scale = parse_csv_doubles(trimmed);
        } else if (key == "weights") {
            model.weights = parse_csv_doubles(trimmed);
        }
    }

    if (model.weights.empty() || model.feature_mean.size() != model.weights.size() ||
        model.feature_scale.size() != model.weights.size()) {
        throw std::runtime_error("Invalid linear surrogate model payload: " + path);
    }
    return model;
}

RegionSelection finalize_scored_selection(const std::vector<double>& scores,
                                          std::size_t nx,
                                          std::size_t ny,
                                          double min_fraction,
                                          double score_threshold) {
    RegionSelection out;
    out.mask.assign(nx * ny, 0);

    if (score_threshold > 0.0) {
        for (std::size_t idx = 0; idx < scores.size(); ++idx) {
            if (scores[idx] > score_threshold) {
                out.mask[idx] = 1;
                ++out.critical_cells;
            }
        }
    }

    const std::size_t min_required = static_cast<std::size_t>(
        std::ceil(std::clamp(min_fraction, 0.0, 1.0) * static_cast<double>(scores.size())));
    if (out.critical_cells < min_required) {
        std::vector<std::size_t> order(scores.size());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
            order.begin(),
            order.begin() + static_cast<std::ptrdiff_t>(min_required),
            order.end(),
            [&scores](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
        for (std::size_t rank = 0; rank < min_required; ++rank) {
            const auto flat = order[rank];
            if (out.mask[flat] == 0) {
                out.mask[flat] = 1;
                ++out.critical_cells;
            }
        }
    }

    out.critical_fraction = static_cast<double>(out.critical_cells) /
        static_cast<double>(nx * ny);
    return out;
}

void validate_2d_inputs(const mesh::Grid2D& grid, const Field2D& coarse) {
    if (coarse.size.x != grid.size.x || coarse.size.y != grid.size.y) {
        throw std::runtime_error("Decision core requires coarse field and grid with matching size");
    }
}

RegionSelection select_critical_regions_heuristic(const mesh::Grid2D& grid,
                                                  const Field2D& coarse,
                                                  const SimulationConfig& config) {
    validate_2d_inputs(grid, coarse);

    const std::size_t nx = coarse.size.x;
    const std::size_t ny = coarse.size.y;
    RegionSelection out;
    out.mask.assign(nx * ny, 0);

    if (nx < 3 || ny < 3) {
        return out;
    }

    std::vector<double> error_proxy(nx * ny, 0.0);
    std::vector<double> uncertainty_proxy(nx * ny, 0.0);

    double max_error_proxy = 0.0;
    double max_uncertainty_proxy = 0.0;

    auto idx = [nx](std::size_t i, std::size_t j) {
        return j * nx + i;
    };

    for (std::size_t j = 1; j + 1 < ny; ++j) {
        for (std::size_t i = 1; i + 1 < nx; ++i) {
            const double t = coarse.at(i, j);
            const double d2x = coarse.at(i + 1, j) - 2.0 * t + coarse.at(i - 1, j);
            const double d2y = coarse.at(i, j + 1) - 2.0 * t + coarse.at(i, j - 1);

            const double lap = d2x / (grid.dx.at(i, j) * grid.dx.at(i, j)) +
                               d2y / (grid.dy.at(i, j) * grid.dy.at(i, j));
            const double gx = (coarse.at(i + 1, j) - coarse.at(i - 1, j)) / (2.0 * grid.dx.at(i, j));
            const double gy = (coarse.at(i, j + 1) - coarse.at(i, j - 1)) / (2.0 * grid.dy.at(i, j));
            const double grad_mag = std::sqrt(gx * gx + gy * gy);

            const double e = std::abs(lap);
            const double u = grad_mag;
            error_proxy[idx(i, j)] = e;
            uncertainty_proxy[idx(i, j)] = u;
            max_error_proxy = std::max(max_error_proxy, e);
            max_uncertainty_proxy = std::max(max_uncertainty_proxy, u);
        }
    }

    const double error_den = (max_error_proxy > 1.0e-12) ? max_error_proxy : 1.0;
    const double uncertainty_den = (max_uncertainty_proxy > 1.0e-12) ? max_uncertainty_proxy : 1.0;

    std::vector<std::pair<double, std::size_t>> ranked_candidates;
    ranked_candidates.reserve((nx - 2) * (ny - 2));

    for (std::size_t j = 1; j + 1 < ny; ++j) {
        for (std::size_t i = 1; i + 1 < nx; ++i) {
            const double e_norm = error_proxy[idx(i, j)] / error_den;
            const double u_norm = uncertainty_proxy[idx(i, j)] / uncertainty_den;
            const double score = e_norm + u_norm;
            const bool critical =
                (e_norm > config.refine_local_error_tau) || (u_norm > config.refine_uncertainty_tau);
            if (critical) {
                out.mask[idx(i, j)] = 1;
                ++out.critical_cells;
            }
            ranked_candidates.emplace_back(score, idx(i, j));
        }
    }

    const double min_frac = std::clamp(config.min_critical_fraction, 0.0, 1.0);
    std::size_t min_required = static_cast<std::size_t>(
        std::ceil(min_frac * static_cast<double>(nx * ny)));
    min_required = std::max<std::size_t>(1, min_required);

    if (out.critical_cells < min_required) {
        std::sort(
            ranked_candidates.begin(),
            ranked_candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        for (const auto& candidate : ranked_candidates) {
            const std::size_t k = candidate.second;
            if (out.mask[k] == 0) {
                out.mask[k] = static_cast<std::uint8_t>(1);
                ++out.critical_cells;
                if (out.critical_cells >= min_required) {
                    break;
                }
            }
        }
    }

    const double total_cells = static_cast<double>(nx * ny);
    out.critical_fraction = static_cast<double>(out.critical_cells) / total_cells;
    return out;
}

RegionSelection select_critical_regions_surrogate_python(const mesh::Grid2D& grid,
                                                         const Field2D& coarse,
                                                         const SimulationConfig& config) {
    validate_2d_inputs(grid, coarse);
    if (config.surrogate_model_path.empty()) {
        throw std::runtime_error("decision_policy=surrogate_python requires surrogate_model_path");
    }

    const auto temp_root = config.surrogate_temp_dir.empty()
        ? (std::filesystem::temp_directory_path() / "shardsim_surrogate")
        : std::filesystem::path(config.surrogate_temp_dir);
    std::filesystem::create_directories(temp_root);

    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto coarse_path = temp_root / ("coarse_" + std::to_string(stamp) + ".bin");
    const auto mask_path = temp_root / ("mask_" + std::to_string(stamp) + ".bin");

    {
        std::ofstream out(coarse_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not create surrogate coarse input: " + coarse_path.string());
        }
        write_u32(out, kCoarseFieldMagic2D);
        write_u32(out, kCoarseFieldVersion2D);
        write_u64(out, static_cast<std::uint64_t>(coarse.size.x));
        write_u64(out, static_cast<std::uint64_t>(coarse.size.y));
        write_u64(out, static_cast<std::uint64_t>(coarse.values.size()));
        out.write(reinterpret_cast<const char*>(coarse.values.data()),
                  static_cast<std::streamsize>(coarse.values.size() * sizeof(double)));
        if (!out) {
            throw std::runtime_error("Failed to write surrogate coarse input");
        }
    }

    const double min_fraction = std::clamp(
        std::max(config.min_critical_fraction, config.surrogate_top_fraction), 0.0, 1.0);

    std::ostringstream cmd;
    cmd << shell_quote(config.surrogate_python_executable)
        << " " << shell_quote(config.surrogate_script_path)
        << " --model " << shell_quote(config.surrogate_model_path)
        << " --input " << shell_quote(coarse_path.string())
        << " --output " << shell_quote(mask_path.string())
        << " --min-fraction " << min_fraction
        << " --score-threshold " << config.surrogate_score_threshold;

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::filesystem::remove(coarse_path);
        std::filesystem::remove(mask_path);
        throw std::runtime_error("Python surrogate mask generation failed with exit code " +
                                 std::to_string(rc));
    }

    RegionSelection out;
    out.mask.assign(coarse.size.x * coarse.size.y, 0);
    {
        std::ifstream in(mask_path, std::ios::binary);
        if (!in) {
            std::filesystem::remove(coarse_path);
            throw std::runtime_error("Could not open surrogate mask output: " + mask_path.string());
        }

        const auto magic = read_u32(in);
        const auto version = read_u32(in);
        const auto nx = read_u64(in);
        const auto ny = read_u64(in);
        const auto count = read_u64(in);
        if (magic != kMaskMagic2D || version != kMaskVersion2D ||
            nx != coarse.size.x || ny != coarse.size.y || count != out.mask.size()) {
            std::filesystem::remove(coarse_path);
            std::filesystem::remove(mask_path);
            throw std::runtime_error("Invalid surrogate mask payload");
        }

        in.read(reinterpret_cast<char*>(out.mask.data()), static_cast<std::streamsize>(out.mask.size()));
        if (!in) {
            std::filesystem::remove(coarse_path);
            std::filesystem::remove(mask_path);
            throw std::runtime_error("Failed to read surrogate mask payload");
        }
    }

    std::filesystem::remove(coarse_path);
    std::filesystem::remove(mask_path);

    for (const auto value : out.mask) {
        out.critical_cells += (value != 0U) ? 1U : 0U;
    }
    const double total_cells = static_cast<double>(coarse.size.x * coarse.size.y);
    out.critical_fraction = static_cast<double>(out.critical_cells) / total_cells;
    return out;
}

RegionSelection select_critical_regions_surrogate_linear(const mesh::Grid2D& grid,
                                                         const Field2D& coarse,
                                                         const SimulationConfig& config) {
    validate_2d_inputs(grid, coarse);
    if (config.surrogate_model_path.empty()) {
        throw std::runtime_error("decision_policy=surrogate_linear requires surrogate_model_path");
    }

    const auto model = load_linear_patch_model(config.surrogate_model_path);
    const std::size_t nx = coarse.size.x;
    const std::size_t ny = coarse.size.y;
    const int radius = static_cast<int>(model.patch_radius);
    const std::size_t patch_width = 2 * model.patch_radius + 1;
    if (model.weights.size() != patch_width * patch_width) {
        throw std::runtime_error("Linear surrogate patch size does not match weight count");
    }

    std::vector<double> scores(nx * ny, 0.0);
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            double value = model.bias;
            std::size_t feature_idx = 0;
            for (int dj = -radius; dj <= radius; ++dj) {
                const std::size_t jj = static_cast<std::size_t>(std::clamp<int>(static_cast<int>(j) + dj, 0, static_cast<int>(ny) - 1));
                for (int di = -radius; di <= radius; ++di) {
                    const std::size_t ii = static_cast<std::size_t>(std::clamp<int>(static_cast<int>(i) + di, 0, static_cast<int>(nx) - 1));
                    const double centered = coarse.at(ii, jj) - model.feature_mean[feature_idx];
                    const double scaled = centered / std::max(model.feature_scale[feature_idx], 1.0e-12);
                    value += scaled * model.weights[feature_idx];
                    ++feature_idx;
                }
            }
            scores[j * nx + i] = std::abs(value);
        }
    }

    const double min_fraction = std::clamp(
        std::max(config.min_critical_fraction, config.surrogate_top_fraction), 0.0, 1.0);
    return finalize_scored_selection(scores, nx, ny, min_fraction, config.surrogate_score_threshold);
}

RegionSelection3D select_critical_regions_heuristic(const mesh::Grid3D& grid,
                                                    const Field3D& coarse,
                                                    const SimulationConfig& config) {
    if (coarse.size.x != grid.size.x || coarse.size.y != grid.size.y || coarse.size.z != grid.size.z) {
        throw std::runtime_error("Decision core requires coarse field and 3D grid with matching size");
    }

    const std::size_t nx = coarse.size.x;
    const std::size_t ny = coarse.size.y;
    const std::size_t nz = coarse.size.z;
    RegionSelection3D out;
    out.mask.assign(nx * ny * nz, 0);

    if (nx < 3 || ny < 3 || nz < 3) {
        return out;
    }

    auto idx = [nx, ny](std::size_t i, std::size_t j, std::size_t k) {
        return (k * ny + j) * nx + i;
    };

    std::vector<double> scores(nx * ny * nz, 0.0);
    double max_score = 0.0;
    std::vector<std::pair<double, std::size_t>> ranked_candidates;
    ranked_candidates.reserve((nx - 2) * (ny - 2) * (nz - 2));

    for (std::size_t k = 1; k + 1 < nz; ++k) {
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            for (std::size_t i = 1; i + 1 < nx; ++i) {
                const double t = coarse.at(i, j, k);
                const double d2x = coarse.at(i + 1, j, k) - 2.0 * t + coarse.at(i - 1, j, k);
                const double d2y = coarse.at(i, j + 1, k) - 2.0 * t + coarse.at(i, j - 1, k);
                const double d2z = coarse.at(i, j, k + 1) - 2.0 * t + coarse.at(i, j, k - 1);
                const double lap = d2x / (grid.dx.at(i, j, k) * grid.dx.at(i, j, k)) +
                                   d2y / (grid.dy.at(i, j, k) * grid.dy.at(i, j, k)) +
                                   d2z / (grid.dz.at(i, j, k) * grid.dz.at(i, j, k));
                const double gx = (coarse.at(i + 1, j, k) - coarse.at(i - 1, j, k)) /
                    (2.0 * grid.dx.at(i, j, k));
                const double gy = (coarse.at(i, j + 1, k) - coarse.at(i, j - 1, k)) /
                    (2.0 * grid.dy.at(i, j, k));
                const double gz = (coarse.at(i, j, k + 1) - coarse.at(i, j, k - 1)) /
                    (2.0 * grid.dz.at(i, j, k));
                const double score = std::abs(lap) + std::sqrt(gx * gx + gy * gy + gz * gz);
                scores[idx(i, j, k)] = score;
                max_score = std::max(max_score, score);
            }
        }
    }

    const double score_den = (max_score > 1.0e-12) ? max_score : 1.0;
    for (std::size_t k = 1; k + 1 < nz; ++k) {
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            for (std::size_t i = 1; i + 1 < nx; ++i) {
                const auto flat = idx(i, j, k);
                const double norm = scores[flat] / score_den;
                if (norm > config.refine_local_error_tau) {
                    out.mask[flat] = 1;
                    ++out.critical_cells;
                }
                ranked_candidates.emplace_back(norm, flat);
            }
        }
    }

    const double min_frac = std::clamp(config.min_critical_fraction, 0.0, 1.0);
    std::size_t min_required = static_cast<std::size_t>(
        std::ceil(min_frac * static_cast<double>(nx * ny * nz)));
    min_required = std::max<std::size_t>(1, min_required);

    if (out.critical_cells < min_required) {
        std::sort(
            ranked_candidates.begin(),
            ranked_candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        for (const auto& candidate : ranked_candidates) {
            const auto flat = candidate.second;
            if (out.mask[flat] == 0) {
                out.mask[flat] = 1;
                ++out.critical_cells;
                if (out.critical_cells >= min_required) {
                    break;
                }
            }
        }
    }

    out.critical_fraction = static_cast<double>(out.critical_cells) /
        static_cast<double>(nx * ny * nz);
    return out;
}

}  // namespace

RegionSelection select_critical_regions(const mesh::Grid2D& grid,
                                        const Field2D& coarse,
                                        const SimulationConfig& config) {
    if (config.decision_policy == "heuristic") {
        return select_critical_regions_heuristic(grid, coarse, config);
    }
    if (config.decision_policy == "surrogate_python") {
        return select_critical_regions_surrogate_python(grid, coarse, config);
    }
    if (config.decision_policy == "surrogate_linear") {
        return select_critical_regions_surrogate_linear(grid, coarse, config);
    }
    throw std::runtime_error("Unsupported decision_policy: " + config.decision_policy);
}

RegionSelection3D select_critical_regions(const mesh::Grid3D& grid,
                                          const Field3D& coarse,
                                          const SimulationConfig& config) {
    if (config.decision_policy != "heuristic") {
        throw std::runtime_error("3D decision core currently supports only decision_policy=heuristic");
    }
    return select_critical_regions_heuristic(grid, coarse, config);
}

}  // namespace shardsim::decision_core
