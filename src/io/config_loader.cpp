#include "shardsim/io/config_loader.hpp"

#include <cctype>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
        ++start;
    }

    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }

    return s.substr(start, end - start);
}

std::unordered_map<std::string, std::string> parse_simple_yaml(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open config file: " + path);
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        auto key = trim(line.substr(0, colon));
        auto value = trim(line.substr(colon + 1));

        if (!value.empty() && (value.front() == '"' || value.front() == '\'')) {
            value.erase(0, 1);
        }
        if (!value.empty() && (value.back() == '"' || value.back() == '\'')) {
            value.pop_back();
        }

        kv[key] = value;
    }

    return kv;
}

template <typename T>
void assign_if_present(const std::unordered_map<std::string, std::string>& kv,
                       const std::string& key,
                       T& target) {
    const auto it = kv.find(key);
    if (it == kv.end()) {
        return;
    }

    std::stringstream ss(it->second);
    ss.imbue(std::locale::classic());
    ss >> target;
    if (ss.fail()) {
        throw std::runtime_error("Invalid config value for key: " + key);
    }
}

}  // namespace

namespace shardsim::io {

SimulationConfig load_config(const std::string& path) {
    auto config = SimulationConfig {};
    const auto kv = parse_simple_yaml(path);

    assign_if_present(kv, "grid_x", config.grid_x);
    assign_if_present(kv, "grid_y", config.grid_y);
    assign_if_present(kv, "grid_z", config.grid_z);
    assign_if_present(kv, "steps", config.steps);
    assign_if_present(kv, "dt", config.dt);
    assign_if_present(kv, "alpha", config.alpha);
    assign_if_present(kv, "coarse_tolerance", config.coarse_tolerance);
    assign_if_present(kv, "fine_tolerance", config.fine_tolerance);
    assign_if_present(kv, "refine_local_error_tau", config.refine_local_error_tau);
    assign_if_present(kv, "refine_uncertainty_tau", config.refine_uncertainty_tau);
    assign_if_present(kv, "min_critical_fraction", config.min_critical_fraction);
    assign_if_present(kv, "memory_ceiling_gb", config.memory_ceiling_gb);
    assign_if_present(kv, "memory_ceiling_mb", config.memory_ceiling_mb);
    assign_if_present(kv, "wallclock_limit_minutes", config.wallclock_limit_minutes);
    assign_if_present(kv, "wallclock_limit_ms", config.wallclock_limit_ms);
    assign_if_present(kv, "halo_overhead_ratio_max", config.halo_overhead_ratio_max);
    assign_if_present(kv, "surrogate_score_threshold", config.surrogate_score_threshold);
    assign_if_present(kv, "surrogate_top_fraction", config.surrogate_top_fraction);
    assign_if_present(kv, "source_x_fraction", config.source_x_fraction);
    assign_if_present(kv, "source_y_fraction", config.source_y_fraction);
    assign_if_present(kv, "source_z_fraction", config.source_z_fraction);
    assign_if_present(kv, "source_temperature", config.source_temperature);
    assign_if_present(kv, "source2_x_fraction", config.source2_x_fraction);
    assign_if_present(kv, "source2_y_fraction", config.source2_y_fraction);
    assign_if_present(kv, "source2_z_fraction", config.source2_z_fraction);
    assign_if_present(kv, "source2_temperature", config.source2_temperature);
    assign_if_present(kv, "presim_steps", config.presim_steps);
    assign_if_present(kv, "presim_coarsening_factor", config.presim_coarsening_factor);
    assign_if_present(kv, "auto_presim_steps", config.auto_presim_steps);
    assign_if_present(kv, "auto_presim_coarsening_factor", config.auto_presim_coarsening_factor);

    const auto det = kv.find("deterministic_mode");
    if (det != kv.end()) {
        config.deterministic_mode = (det->second == "true" || det->second == "1");
    }

    const auto partition = kv.find("partitioning_policy");
    if (partition != kv.end()) {
        config.partitioning_policy = partition->second;
    }

    const auto decision = kv.find("decision_policy");
    if (decision != kv.end()) {
        config.decision_policy = decision->second;
    }

    const auto solver_family = kv.find("solver_family");
    if (solver_family != kv.end()) {
        config.solver_family = solver_family->second;
    }

    const auto decomposition_mode = kv.find("decomposition_mode");
    if (decomposition_mode != kv.end()) {
        config.decomposition_mode = decomposition_mode->second;
    }

    const auto export_training = kv.find("export_training_data");
    if (export_training != kv.end()) {
        config.export_training_data =
            (export_training->second == "true" || export_training->second == "1");
    }

    const auto export_dir = kv.find("training_data_export_dir");
    if (export_dir != kv.end()) {
        config.training_data_export_dir = export_dir->second;
    }

    const auto surrogate_model = kv.find("surrogate_model_path");
    if (surrogate_model != kv.end()) {
        config.surrogate_model_path = surrogate_model->second;
    }

    const auto surrogate_python = kv.find("surrogate_python_executable");
    if (surrogate_python != kv.end()) {
        config.surrogate_python_executable = surrogate_python->second;
    }

    const auto surrogate_script = kv.find("surrogate_script_path");
    if (surrogate_script != kv.end()) {
        config.surrogate_script_path = surrogate_script->second;
    }

    const auto surrogate_temp_dir = kv.find("surrogate_temp_dir");
    if (surrogate_temp_dir != kv.end()) {
        config.surrogate_temp_dir = surrogate_temp_dir->second;
    }

    const auto source2_enabled = kv.find("source2_enabled");
    if (source2_enabled != kv.end()) {
        config.source2_enabled =
            (source2_enabled->second == "true" || source2_enabled->second == "1");
    }

    const auto auto_presim_enabled = kv.find("auto_enable_presim");
    if (auto_presim_enabled != kv.end()) {
        config.auto_enable_presim =
            (auto_presim_enabled->second == "true" || auto_presim_enabled->second == "1");
    }

    const auto auto_correction_enabled = kv.find("auto_enable_correction");
    if (auto_correction_enabled != kv.end()) {
        config.auto_enable_correction =
            (auto_correction_enabled->second == "true" || auto_correction_enabled->second == "1");
    }

    // Phase 4: ML correction loop.
    const auto correction_policy = kv.find("correction_policy");
    if (correction_policy != kv.end()) {
        config.correction_policy = correction_policy->second;
    }
    const auto correction_model = kv.find("correction_model_path");
    if (correction_model != kv.end()) {
        config.correction_model_path = correction_model->second;
    }
    const auto correction_script = kv.find("correction_script_path");
    if (correction_script != kv.end()) {
        config.correction_script_path = correction_script->second;
    }
    const auto correction_python = kv.find("correction_python_executable");
    if (correction_python != kv.end()) {
        config.correction_python_executable = correction_python->second;
    }

    const auto auto_correction_policy = kv.find("auto_correction_policy");
    if (auto_correction_policy != kv.end()) {
        config.auto_correction_policy = auto_correction_policy->second;
    }

    const auto auto_correction_model = kv.find("auto_correction_model_path");
    if (auto_correction_model != kv.end()) {
        config.auto_correction_model_path = auto_correction_model->second;
    }

    return config;
}

}  // namespace shardsim::io
