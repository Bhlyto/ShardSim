#include "shardsim/correction/correction.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace shardsim::correction {

namespace {

// ---------------------------------------------------------------------------
// Model loading — same text format as policy.cpp LinearPatchModel2D
// ---------------------------------------------------------------------------

struct LinearPatchModel {
    std::size_t patch_radius {0};
    double bias {0.0};
    std::vector<double> feature_mean;
    std::vector<double> feature_scale;
    std::vector<double> weights;
};

std::vector<double> parse_csv_doubles(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    return values;
}

std::string trim(const std::string& s) {
    std::size_t lo = 0;
    while (lo < s.size() && std::isspace(static_cast<unsigned char>(s[lo])) != 0) {
        ++lo;
    }
    std::size_t hi = s.size();
    while (hi > lo && std::isspace(static_cast<unsigned char>(s[hi - 1])) != 0) {
        --hi;
    }
    return s.substr(lo, hi - lo);
}

LinearPatchModel load_model(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("correction: cannot open model file: " + path);
    }
    LinearPatchModel m;
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const auto key = trim(line.substr(0, colon));
        const auto val = trim(line.substr(colon + 1));
        if (key == "patch_radius") {
            m.patch_radius = static_cast<std::size_t>(std::stoull(val));
        } else if (key == "bias") {
            m.bias = std::stod(val);
        } else if (key == "feature_mean") {
            m.feature_mean = parse_csv_doubles(val);
        } else if (key == "feature_scale") {
            m.feature_scale = parse_csv_doubles(val);
        } else if (key == "weights") {
            m.weights = parse_csv_doubles(val);
        }
    }
    if (m.weights.empty() || m.feature_mean.size() != m.weights.size() ||
        m.feature_scale.size() != m.weights.size()) {
        throw std::runtime_error("correction: invalid model file: " + path);
    }
    return m;
}

// ---------------------------------------------------------------------------
// Native linear patch correction
// ---------------------------------------------------------------------------

Field2D apply_linear(const Field2D& coarse, const std::string& model_path) {
    const LinearPatchModel m = load_model(model_path);
    const int R = static_cast<int>(m.patch_radius);
    const int n_features = static_cast<int>(m.weights.size());

    const std::size_t nx = coarse.size.x;
    const std::size_t ny = coarse.size.y;

    Field2D out;
    out.size = coarse.size;
    out.values = coarse.values;  // start from coarse, add corrections in-place

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            double value = m.bias;
            int feature_idx = 0;
            for (int dj = -R; dj <= R && feature_idx < n_features; ++dj) {
                const std::size_t jj = static_cast<std::size_t>(
                    std::clamp(static_cast<int>(j) + dj, 0, static_cast<int>(ny) - 1));
                for (int di = -R; di <= R && feature_idx < n_features; ++di) {
                    const std::size_t ii = static_cast<std::size_t>(
                        std::clamp(static_cast<int>(i) + di, 0, static_cast<int>(nx) - 1));
                    const double raw = coarse.at(ii, jj);
                    const double centered = raw - m.feature_mean[feature_idx];
                    const double scaled = centered / std::max(m.feature_scale[feature_idx], 1.0e-12);
                    value += scaled * m.weights[feature_idx];
                    ++feature_idx;
                }
            }
            // correction = predicted discrepancy (fine - coarse)
            out.at(i, j) += value;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Binary I/O helpers (same layout as policy.cpp coarse field write)
// ---------------------------------------------------------------------------

constexpr std::uint32_t kCoarseFieldMagic2D = 0x31434453;  // "SDC1"
constexpr std::uint32_t kCoarseFieldVersion2D = 1;

void write_u32(std::ostream& out, std::uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void write_u64(std::ostream& out, std::uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
std::uint32_t read_u32(std::istream& in) {
    std::uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
std::uint64_t read_u64(std::istream& in) {
    std::uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

void write_field_bin(const Field2D& field, const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("correction: cannot write coarse field: " + path.string());
    }
    write_u32(out, kCoarseFieldMagic2D);
    write_u32(out, kCoarseFieldVersion2D);
    write_u64(out, static_cast<std::uint64_t>(field.size.x));
    write_u64(out, static_cast<std::uint64_t>(field.size.y));
    write_u64(out, static_cast<std::uint64_t>(field.values.size()));
    out.write(reinterpret_cast<const char*>(field.values.data()),
              static_cast<std::streamsize>(field.values.size() * sizeof(double)));
    if (!out) {
        throw std::runtime_error("correction: failed to write field payload");
    }
}

Field2D read_field_bin(const std::filesystem::path& path, std::size_t expected_nx, std::size_t expected_ny) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("correction: cannot read corrected field: " + path.string());
    }
    const auto magic   = read_u32(in);
    const auto version = read_u32(in);
    const auto nx      = static_cast<std::size_t>(read_u64(in));
    const auto ny      = static_cast<std::size_t>(read_u64(in));
    const auto count   = static_cast<std::size_t>(read_u64(in));
    if (magic != kCoarseFieldMagic2D || version != kCoarseFieldVersion2D ||
        nx != expected_nx || ny != expected_ny || count != nx * ny) {
        throw std::runtime_error("correction: invalid corrected field header: " + path.string());
    }
    Field2D out;
    out.size = {nx, ny};
    out.values.resize(count);
    in.read(reinterpret_cast<char*>(out.values.data()),
            static_cast<std::streamsize>(count * sizeof(double)));
    if (!in) {
        throw std::runtime_error("correction: failed to read corrected field payload");
    }
    return out;
}

std::string shell_quote(const std::string& s) {
    std::string q;
    q.reserve(s.size() + 2);
    q.push_back('\'');
    for (char c : s) {
        if (c == '\'') { q += "'\\''"; } else { q.push_back(c); }
    }
    q.push_back('\'');
    return q;
}

// ---------------------------------------------------------------------------
// Python XGBoost correction (shells out to apply_correction.py)
// ---------------------------------------------------------------------------

Field2D apply_python(const Field2D& coarse, const SimulationConfig& cfg) {
    if (cfg.correction_script_path.empty()) {
        throw std::runtime_error("correction_policy=python requires correction_script_path");
    }
    if (cfg.correction_model_path.empty()) {
        throw std::runtime_error("correction_policy=python requires correction_model_path");
    }

    const auto temp_root = cfg.surrogate_temp_dir.empty()
        ? (std::filesystem::temp_directory_path() / "shardsim_correction")
        : std::filesystem::path(cfg.surrogate_temp_dir);
    std::filesystem::create_directories(temp_root);

    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto in_path  = temp_root / ("coarse_" + std::to_string(stamp) + ".bin");
    const auto out_path = temp_root / ("corrected_" + std::to_string(stamp) + ".bin");

    write_field_bin(coarse, in_path);

    std::ostringstream cmd;
    cmd << shell_quote(cfg.correction_python_executable)
        << " " << shell_quote(cfg.correction_script_path)
        << " --model "  << shell_quote(cfg.correction_model_path)
        << " --input "  << shell_quote(in_path.string())
        << " --output " << shell_quote(out_path.string());

    const int rc = std::system(cmd.str().c_str());
    std::filesystem::remove(in_path);
    if (rc != 0) {
        std::filesystem::remove(out_path);
        throw std::runtime_error("correction: Python script failed with exit code " +
                                 std::to_string(rc));
    }

    const auto result = read_field_bin(out_path, coarse.size.x, coarse.size.y);
    std::filesystem::remove(out_path);
    return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

Field2D apply_correction(const Field2D& coarse, const SimulationConfig& config) {
    if (config.correction_policy == "none" || config.correction_policy.empty()) {
        return coarse;
    }
    if (config.correction_policy == "linear") {
        if (config.correction_model_path.empty()) {
            throw std::runtime_error("correction_policy=linear requires correction_model_path");
        }
        return apply_linear(coarse, config.correction_model_path);
    }
    if (config.correction_policy == "python") {
        return apply_python(coarse, config);
    }
    throw std::runtime_error("Unsupported correction_policy: " + config.correction_policy);
}

}  // namespace shardsim::correction
