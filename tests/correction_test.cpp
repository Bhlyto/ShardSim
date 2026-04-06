#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "shardsim/config.hpp"
#include "shardsim/correction/correction.hpp"
#include "shardsim/mesh/mesh.hpp"
#include "shardsim/types.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static shardsim::Field2D make_gaussian_field(std::size_t nx, std::size_t ny) {
    shardsim::Field2D f;
    f.size = {nx, ny};
    f.values.resize(nx * ny);
    const double cx = static_cast<double>(nx) * 0.5;
    const double cy = static_cast<double>(ny) * 0.5;
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            const double dx = static_cast<double>(i) - cx;
            const double dy = static_cast<double>(j) - cy;
            f.at(i, j) = std::exp(-(dx * dx + dy * dy) / 200.0);
        }
    }
    return f;
}

static std::string find_model_path() {
    // Prefer the cmake-injected source dir when available.
#ifdef SHARDSIM_SOURCE_DIR
    {
        std::string p = std::string(SHARDSIM_SOURCE_DIR) + "/models/surrogate_linear_policy.txt";
        std::ifstream f(p);
        if (f.good()) {
            return p;
        }
    }
#endif
    // Try relative path from cwd (build/ → ../).
    for (const char* p : {"models/surrogate_linear_policy.txt",
                           "../models/surrogate_linear_policy.txt",
                           "../../models/surrogate_linear_policy.txt"}) {
        std::ifstream f(p);
        if (f.good()) {
            return p;
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// Test 1: policy=none passes coarse field through unchanged
// ---------------------------------------------------------------------------
static int test_none_passthrough() {
    shardsim::SimulationConfig cfg;
    cfg.correction_policy = "none";

    const auto field = make_gaussian_field(32, 32);
    const auto result = shardsim::correction::apply_correction(field, cfg);

    if (result.values != field.values) {
        std::cerr << "[FAIL] test_none_passthrough: values changed\n";
        return 1;
    }
    if (result.size.x != field.size.x || result.size.y != field.size.y) {
        std::cerr << "[FAIL] test_none_passthrough: size changed\n";
        return 1;
    }
    std::cout << "[PASS] test_none_passthrough\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 2: empty correction_policy treated as none
// ---------------------------------------------------------------------------
static int test_empty_policy_passthrough() {
    shardsim::SimulationConfig cfg;
    cfg.correction_policy = "";

    const auto field = make_gaussian_field(16, 16);
    const auto result = shardsim::correction::apply_correction(field, cfg);

    if (result.values != field.values) {
        std::cerr << "[FAIL] test_empty_policy_passthrough: values changed\n";
        return 1;
    }
    std::cout << "[PASS] test_empty_policy_passthrough\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 3: linear correction with missing model path throws
// ---------------------------------------------------------------------------
static int test_linear_missing_model_throws() {
    shardsim::SimulationConfig cfg;
    cfg.correction_policy = "linear";
    cfg.correction_model_path = "";  // empty → should throw

    const auto field = make_gaussian_field(16, 16);
    try {
        shardsim::correction::apply_correction(field, cfg);
        std::cerr << "[FAIL] test_linear_missing_model_throws: no exception thrown\n";
        return 1;
    } catch (const std::runtime_error&) {
        // expected
    }
    std::cout << "[PASS] test_linear_missing_model_throws\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 4: linear correction with nonexistent model path throws
// ---------------------------------------------------------------------------
static int test_linear_bad_model_path_throws() {
    shardsim::SimulationConfig cfg;
    cfg.correction_policy = "linear";
    cfg.correction_model_path = "/nonexistent/path/model.txt";

    const auto field = make_gaussian_field(16, 16);
    try {
        shardsim::correction::apply_correction(field, cfg);
        std::cerr << "[FAIL] test_linear_bad_model_path_throws: no exception thrown\n";
        return 1;
    } catch (const std::runtime_error&) {
        // expected
    }
    std::cout << "[PASS] test_linear_bad_model_path_throws\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 5: linear correction with real model — output has same dimensions
//         and corrections are finite
// ---------------------------------------------------------------------------
static int test_linear_real_model_dimensions() {
    const std::string model_path = find_model_path();
    if (model_path.empty()) {
        std::cout << "[SKIP] test_linear_real_model_dimensions: model not found\n";
        return 0;
    }

    shardsim::SimulationConfig cfg;
    cfg.correction_policy = "linear";
    cfg.correction_model_path = model_path;

    const auto field = make_gaussian_field(64, 64);
    const auto result = shardsim::correction::apply_correction(field, cfg);

    if (result.size.x != field.size.x || result.size.y != field.size.y) {
        std::cerr << "[FAIL] test_linear_real_model_dimensions: size changed\n";
        return 1;
    }
    if (result.values.size() != field.values.size()) {
        std::cerr << "[FAIL] test_linear_real_model_dimensions: values size changed\n";
        return 1;
    }
    for (double v : result.values) {
        if (!std::isfinite(v)) {
            std::cerr << "[FAIL] test_linear_real_model_dimensions: non-finite value\n";
            return 1;
        }
    }
    std::cout << "[PASS] test_linear_real_model_dimensions\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 6: linear correction modifies the field (not a no-op)
// ---------------------------------------------------------------------------
static int test_linear_real_model_modifies_field() {
    const std::string model_path = find_model_path();
    if (model_path.empty()) {
        std::cout << "[SKIP] test_linear_real_model_modifies_field: model not found\n";
        return 0;
    }

    shardsim::SimulationConfig cfg;
    cfg.correction_policy = "linear";
    cfg.correction_model_path = model_path;

    const auto field = make_gaussian_field(64, 64);
    const auto result = shardsim::correction::apply_correction(field, cfg);

    bool any_different = false;
    for (std::size_t k = 0; k < field.values.size(); ++k) {
        if (std::abs(result.values[k] - field.values[k]) > 1.0e-15) {
            any_different = true;
            break;
        }
    }
    if (!any_different) {
        std::cerr << "[FAIL] test_linear_real_model_modifies_field: correction is all-zero\n";
        return 1;
    }
    std::cout << "[PASS] test_linear_real_model_modifies_field\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 7: unsupported policy throws
// ---------------------------------------------------------------------------
static int test_unsupported_policy_throws() {
    shardsim::SimulationConfig cfg;
    cfg.correction_policy = "xgboost_magic";

    const auto field = make_gaussian_field(8, 8);
    try {
        shardsim::correction::apply_correction(field, cfg);
        std::cerr << "[FAIL] test_unsupported_policy_throws: no exception thrown\n";
        return 1;
    } catch (const std::runtime_error&) {
        // expected
    }
    std::cout << "[PASS] test_unsupported_policy_throws\n";
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    int failures = 0;
    failures += test_none_passthrough();
    failures += test_empty_policy_passthrough();
    failures += test_linear_missing_model_throws();
    failures += test_linear_bad_model_path_throws();
    failures += test_linear_real_model_dimensions();
    failures += test_linear_real_model_modifies_field();
    failures += test_unsupported_policy_throws();

    if (failures == 0) {
        std::cout << "All correction tests passed.\n";
    } else {
        std::cerr << failures << " correction test(s) failed.\n";
    }
    return failures;
}
