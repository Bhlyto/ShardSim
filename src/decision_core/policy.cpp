#include "shardsim/decision_core/policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "shardsim/presim/presim.hpp"

namespace shardsim::decision_core {

namespace {

constexpr std::uint32_t kCoarseFieldMagic2D = 0x31434453;  // "SDC1"
constexpr std::uint32_t kCoarseFieldVersion2D = 1;
constexpr std::uint32_t kMaskMagic2D = 0x314d4453;         // "SDM1"
constexpr std::uint32_t kMaskVersion2D = 1;
constexpr std::uint32_t kCoarseFieldMagic3D = 0x33434453;  // "SDC3"
constexpr std::uint32_t kCoarseFieldVersion3D = 1;
constexpr std::uint32_t kMaskMagic3D = 0x334d4453;         // "SDM3"
constexpr std::uint32_t kMaskVersion3D = 1;

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

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    return out;
}

std::string parse_worker_error(const std::string& response) {
    const std::string key = "\"error\":";
    const auto pos = response.find(key);
    if (pos == std::string::npos) {
        return "unknown worker error";
    }
    const auto start = response.find('"', pos + key.size());
    if (start == std::string::npos) {
        return "unknown worker error";
    }
    std::string error;
    bool escaped = false;
    for (std::size_t i = start + 1; i < response.size(); ++i) {
        const char ch = response[i];
        if (escaped) {
            switch (ch) {
            case 'n':
                error.push_back('\n');
                break;
            case 'r':
                error.push_back('\r');
                break;
            case 't':
                error.push_back('\t');
                break;
            case '"':
            case '\\':
                error.push_back(ch);
                break;
            default:
                error.push_back(ch);
                break;
            }
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            break;
        }
        error.push_back(ch);
    }
    return error.empty() ? "unknown worker error" : error;
}

class PythonSurrogateWorker {
public:
    PythonSurrogateWorker() = default;
    ~PythonSurrogateWorker() {
        stop();
    }

    PythonSurrogateWorker(const PythonSurrogateWorker&) = delete;
    PythonSurrogateWorker& operator=(const PythonSurrogateWorker&) = delete;

    void run_inference(const SimulationConfig& config,
                       const std::filesystem::path& coarse_path,
                       const std::filesystem::path& mask_path,
                       double min_fraction,
                       double score_threshold) {
        ensure_started(config);

        std::ostringstream request;
        request << "{\"input\":\"" << json_escape(coarse_path.string())
                << "\",\"output\":\"" << json_escape(mask_path.string())
                << "\",\"min_fraction\":" << min_fraction
                << ",\"score_threshold\":" << score_threshold
                << "}";

        if (std::fprintf(to_worker_, "%s\n", request.str().c_str()) < 0 || std::fflush(to_worker_) != 0) {
            stop();
            throw std::runtime_error("Failed to send request to surrogate worker");
        }

        char response[8192] = {0};
        if (std::fgets(response, static_cast<int>(sizeof(response)), from_worker_) == nullptr) {
            stop();
            throw std::runtime_error("Surrogate worker exited before sending response");
        }

        const std::string response_line(response);
        if (response_line.find("\"ok\": true") != std::string::npos ||
            response_line.find("\"ok\":true") != std::string::npos) {
            return;
        }

        const std::string worker_error = parse_worker_error(response_line);
        stop();
        throw std::runtime_error("Surrogate worker inference failed: " + worker_error);
    }

private:
    void ensure_started(const SimulationConfig& config) {
        const bool same_config = running_ &&
            cached_python_ == config.surrogate_python_executable &&
            cached_script_ == config.surrogate_script_path &&
            cached_model_ == config.surrogate_model_path;
        if (same_config) {
            return;
        }

        stop();

        int to_child[2] = {-1, -1};
        int from_child[2] = {-1, -1};
        if (pipe(to_child) != 0 || pipe(from_child) != 0) {
            if (to_child[0] >= 0) {
                close(to_child[0]);
            }
            if (to_child[1] >= 0) {
                close(to_child[1]);
            }
            if (from_child[0] >= 0) {
                close(from_child[0]);
            }
            if (from_child[1] >= 0) {
                close(from_child[1]);
            }
            throw std::runtime_error("Failed to create pipes for surrogate worker");
        }

        const pid_t child = fork();
        if (child < 0) {
            close(to_child[0]);
            close(to_child[1]);
            close(from_child[0]);
            close(from_child[1]);
            throw std::runtime_error("Failed to fork surrogate worker process");
        }

        if (child == 0) {
            dup2(to_child[0], STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            close(to_child[0]);
            close(to_child[1]);
            close(from_child[0]);
            close(from_child[1]);

            const std::string model_arg = config.surrogate_model_path;
            const std::string script_arg = config.surrogate_script_path;
            const std::string worker_flag = "--worker-stdio";
            const std::string model_flag = "--model";

            execlp(config.surrogate_python_executable.c_str(),
                   config.surrogate_python_executable.c_str(),
                   script_arg.c_str(),
                   worker_flag.c_str(),
                   model_flag.c_str(),
                   model_arg.c_str(),
                   static_cast<char*>(nullptr));
            _exit(127);
        }

        close(to_child[0]);
        close(from_child[1]);

        to_worker_ = fdopen(to_child[1], "w");
        from_worker_ = fdopen(from_child[0], "r");
        if (to_worker_ == nullptr || from_worker_ == nullptr) {
            if (to_worker_ != nullptr) {
                fclose(to_worker_);
                to_worker_ = nullptr;
            } else {
                close(to_child[1]);
            }
            if (from_worker_ != nullptr) {
                fclose(from_worker_);
                from_worker_ = nullptr;
            } else {
                close(from_child[0]);
            }
            kill(child, SIGTERM);
            waitpid(child, nullptr, 0);
            throw std::runtime_error("Failed to attach surrogate worker pipes");
        }

        setvbuf(to_worker_, nullptr, _IOLBF, 0);
        setvbuf(from_worker_, nullptr, _IOLBF, 0);
        pid_ = child;
        running_ = true;
        cached_python_ = config.surrogate_python_executable;
        cached_script_ = config.surrogate_script_path;
        cached_model_ = config.surrogate_model_path;
    }

    void stop() {
        if (!running_) {
            return;
        }

        if (to_worker_ != nullptr) {
            std::fprintf(to_worker_, "{\"command\":\"shutdown\"}\n");
            std::fflush(to_worker_);
            fclose(to_worker_);
            to_worker_ = nullptr;
        }
        if (from_worker_ != nullptr) {
            fclose(from_worker_);
            from_worker_ = nullptr;
        }
        if (pid_ > 0) {
            int status = 0;
            const pid_t waited = waitpid(pid_, &status, WNOHANG);
            if (waited == 0) {
                kill(pid_, SIGTERM);
                waitpid(pid_, &status, 0);
            }
        }

        running_ = false;
        pid_ = -1;
        cached_python_.clear();
        cached_script_.clear();
        cached_model_.clear();
    }

    pid_t pid_ {-1};
    FILE* to_worker_ {nullptr};
    FILE* from_worker_ {nullptr};
    bool running_ {false};
    std::string cached_python_;
    std::string cached_script_;
    std::string cached_model_;
};

PythonSurrogateWorker& python_worker_singleton() {
    static PythonSurrogateWorker worker;
    return worker;
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
                                                         const SimulationConfig& config,
                                                         bool cached_worker_mode) {
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

    if (cached_worker_mode) {
        try {
            python_worker_singleton().run_inference(
                config,
                coarse_path,
                mask_path,
                min_fraction,
                config.surrogate_score_threshold);
        } catch (...) {
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
        }
    } else {
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

RegionSelection3D select_critical_regions_surrogate_python(const mesh::Grid3D& grid,
                                                           const Field3D& coarse,
                                                           const SimulationConfig& config,
                                                           bool cached_worker_mode) {
    if (coarse.size.x != grid.size.x || coarse.size.y != grid.size.y || coarse.size.z != grid.size.z) {
        throw std::runtime_error("Decision core requires coarse field and 3D grid with matching size");
    }
    if (config.surrogate_model_path.empty()) {
        throw std::runtime_error("decision_policy=surrogate_python requires surrogate_model_path");
    }

    const auto temp_root = config.surrogate_temp_dir.empty()
        ? (std::filesystem::temp_directory_path() / "shardsim_surrogate_3d")
        : std::filesystem::path(config.surrogate_temp_dir);
    std::filesystem::create_directories(temp_root);

    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto coarse_path = temp_root / ("coarse3d_" + std::to_string(stamp) + ".bin");
    const auto mask_path = temp_root / ("mask3d_" + std::to_string(stamp) + ".bin");

    {
        std::ofstream out(coarse_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not create surrogate 3D coarse input: " + coarse_path.string());
        }
        write_u32(out, kCoarseFieldMagic3D);
        write_u32(out, kCoarseFieldVersion3D);
        write_u64(out, static_cast<std::uint64_t>(coarse.size.x));
        write_u64(out, static_cast<std::uint64_t>(coarse.size.y));
        write_u64(out, static_cast<std::uint64_t>(coarse.size.z));
        write_u64(out, static_cast<std::uint64_t>(coarse.values.size()));
        out.write(reinterpret_cast<const char*>(coarse.values.data()),
                  static_cast<std::streamsize>(coarse.values.size() * sizeof(double)));
        if (!out) {
            throw std::runtime_error("Failed to write surrogate 3D coarse input");
        }
    }

    const double min_fraction = std::clamp(
        std::max(config.min_critical_fraction, config.surrogate_top_fraction), 0.0, 1.0);

    if (cached_worker_mode) {
        try {
            python_worker_singleton().run_inference(
                config,
                coarse_path,
                mask_path,
                min_fraction,
                config.surrogate_score_threshold);
        } catch (...) {
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
                throw std::runtime_error("Python surrogate 3D mask generation failed with exit code " +
                                         std::to_string(rc));
            }
        }
    } else {
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
            throw std::runtime_error("Python surrogate 3D mask generation failed with exit code " +
                                     std::to_string(rc));
        }
    }

    RegionSelection3D out;
    out.mask.assign(coarse.size.x * coarse.size.y * coarse.size.z, 0);
    {
        std::ifstream in(mask_path, std::ios::binary);
        if (!in) {
            std::filesystem::remove(coarse_path);
            throw std::runtime_error("Could not open surrogate 3D mask output: " + mask_path.string());
        }

        const auto magic = read_u32(in);
        const auto version = read_u32(in);
        const auto nx = read_u64(in);
        const auto ny = read_u64(in);
        const auto nz = read_u64(in);
        const auto count = read_u64(in);
        if (magic != kMaskMagic3D || version != kMaskVersion3D ||
            nx != coarse.size.x || ny != coarse.size.y || nz != coarse.size.z || count != out.mask.size()) {
            std::filesystem::remove(coarse_path);
            std::filesystem::remove(mask_path);
            throw std::runtime_error("Invalid surrogate 3D mask payload");
        }

        in.read(reinterpret_cast<char*>(out.mask.data()), static_cast<std::streamsize>(out.mask.size()));
        if (!in) {
            std::filesystem::remove(coarse_path);
            std::filesystem::remove(mask_path);
            throw std::runtime_error("Failed to read surrogate 3D mask payload");
        }
    }

    std::filesystem::remove(coarse_path);
    std::filesystem::remove(mask_path);

    for (const auto value : out.mask) {
        out.critical_cells += (value != 0U) ? 1U : 0U;
    }
    out.critical_fraction = static_cast<double>(out.critical_cells) /
        static_cast<double>(coarse.size.x * coarse.size.y * coarse.size.z);
    return out;
}

}  // namespace

RegionSelection select_critical_regions(const mesh::Grid2D& grid,
                                        const Field2D& coarse,
                                        const SimulationConfig& config) {
    RegionSelection out;
    if (config.decision_policy == "heuristic") {
        out = select_critical_regions_heuristic(grid, coarse, config);
    } else if (config.decision_policy == "surrogate_python") {
        out = select_critical_regions_surrogate_python(grid, coarse, config, false);
    } else if (config.decision_policy == "surrogate_python_cached") {
        out = select_critical_regions_surrogate_python(grid, coarse, config, true);
    } else if (config.decision_policy == "surrogate_linear") {
        out = select_critical_regions_surrogate_linear(grid, coarse, config);
    } else {
        throw std::runtime_error("Unsupported decision_policy: " + config.decision_policy);
    }

    // Phase 3: Pre-simulation overlay.  When presim_steps > 0 we run a fast
    // coarsened heat solve and mark any cell whose uncertainty score exceeds
    // refine_local_error_tau as critical, even if the base policy missed it.
    if (config.presim_steps > 0) {
        const auto presim_map = presim::run_presim(grid, config);
        if (!presim_map.scores.empty() && presim_map.scores.size() == out.mask.size()) {
            const double tau = std::max(config.refine_local_error_tau, 1.0e-6);
            for (std::size_t k = 0; k < out.mask.size(); ++k) {
                if (out.mask[k] == 0 && presim_map.scores[k] > tau) {
                    out.mask[k] = 1;
                    ++out.critical_cells;
                }
            }
            out.critical_fraction = static_cast<double>(out.critical_cells) /
                                    static_cast<double>(out.mask.size());
        }
    }

    return out;
}

RegionSelection3D select_critical_regions(const mesh::Grid3D& grid,
                                          const Field3D& coarse,
                                          const SimulationConfig& config) {
    if (config.decision_policy == "heuristic") {
        return select_critical_regions_heuristic(grid, coarse, config);
    }
    if (config.decision_policy == "surrogate_python") {
        return select_critical_regions_surrogate_python(grid, coarse, config, false);
    }
    if (config.decision_policy == "surrogate_python_cached") {
        return select_critical_regions_surrogate_python(grid, coarse, config, true);
    }
    throw std::runtime_error("Unsupported 3D decision_policy: " + config.decision_policy);
}

}  // namespace shardsim::decision_core
