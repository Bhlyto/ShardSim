#include <algorithm>
#include <cstddef>
#include <iostream>

#include "shardsim/config.hpp"
#include "shardsim/mesh/mesh.hpp"
#include "shardsim/presim/presim.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static shardsim::SimulationConfig make_base_config(std::size_t nx, std::size_t ny) {
    shardsim::SimulationConfig cfg;
    cfg.grid_x = nx;
    cfg.grid_y = ny;
    cfg.alpha = 1.0e-5;
    cfg.source_x_fraction = 0.5;
    cfg.source_y_fraction = 0.5;
    cfg.source_temperature = 1.0;
    return cfg;
}

// ---------------------------------------------------------------------------
// Test 1: disabled pre-sim returns empty UncertaintyMap
// ---------------------------------------------------------------------------
static int test_disabled_returns_empty() {
    shardsim::SimulationConfig cfg = make_base_config(32, 32);
    cfg.presim_steps = 0;  // disabled

    const auto grid = shardsim::mesh::make_non_uniform_grid(cfg);
    const auto result = shardsim::presim::run_presim(grid, cfg);

    if (!result.scores.empty()) {
        std::cerr << "[FAIL] test_disabled_returns_empty: expected empty scores\n";
        return 1;
    }
    if (result.nx != 0 || result.ny != 0) {
        std::cerr << "[FAIL] test_disabled_returns_empty: expected nx==ny==0\n";
        return 1;
    }
    std::cout << "[PASS] test_disabled_returns_empty\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 2: enabled pre-sim returns map with correct full-grid dimensions
// ---------------------------------------------------------------------------
static int test_output_dimensions() {
    constexpr std::size_t NX = 32;
    constexpr std::size_t NY = 24;

    shardsim::SimulationConfig cfg = make_base_config(NX, NY);
    cfg.presim_steps = 50;
    cfg.presim_coarsening_factor = 4;

    const auto grid = shardsim::mesh::make_non_uniform_grid(cfg);
    const auto result = shardsim::presim::run_presim(grid, cfg);

    if (result.nx != NX || result.ny != NY) {
        std::cerr << "[FAIL] test_output_dimensions: expected (" << NX << "," << NY
                  << ") got (" << result.nx << "," << result.ny << ")\n";
        return 1;
    }
    if (result.scores.size() != NX * NY) {
        std::cerr << "[FAIL] test_output_dimensions: scores size mismatch\n";
        return 1;
    }
    std::cout << "[PASS] test_output_dimensions\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 3: all scores are in [0, 1]
// ---------------------------------------------------------------------------
static int test_scores_in_unit_interval() {
    shardsim::SimulationConfig cfg = make_base_config(40, 40);
    cfg.presim_steps = 100;
    cfg.presim_coarsening_factor = 4;

    const auto grid = shardsim::mesh::make_non_uniform_grid(cfg);
    const auto result = shardsim::presim::run_presim(grid, cfg);

    const double max_score = *std::max_element(result.scores.begin(), result.scores.end());
    const double min_score = *std::min_element(result.scores.begin(), result.scores.end());

    if (min_score < -1.0e-9 || max_score > 1.0 + 1.0e-9) {
        std::cerr << "[FAIL] test_scores_in_unit_interval: range [" << min_score
                  << ", " << max_score << "] outside [0,1]\n";
        return 1;
    }
    std::cout << "[PASS] test_scores_in_unit_interval\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 4: blend_presim_scores preserves size and keeps values in [0, 1]
// ---------------------------------------------------------------------------
static int test_blend_preserves_size_and_range() {
    constexpr std::size_t N = 16;
    constexpr std::size_t M = 16;

    shardsim::presim::UncertaintyMap umap;
    umap.nx = N;
    umap.ny = M;
    umap.scores.assign(N * M, 0.7);  // constant presim score

    std::vector<double> heuristic(N * M, 0.3);  // constant heuristic score

    shardsim::presim::blend_presim_scores(heuristic, umap, N, M, 0.5);

    if (heuristic.size() != N * M) {
        std::cerr << "[FAIL] test_blend_preserves_size_and_range: size changed\n";
        return 1;
    }
    for (std::size_t k = 0; k < heuristic.size(); ++k) {
        if (heuristic[k] < -1.0e-9 || heuristic[k] > 1.0 + 1.0e-9) {
            std::cerr << "[FAIL] test_blend_preserves_size_and_range: value out of range\n";
            return 1;
        }
    }
    // Expected blended value: 0.5*0.3 + 0.5*0.7 = 0.5
    const double expected = 0.5;
    if (std::abs(heuristic[0] - expected) > 1.0e-9) {
        std::cerr << "[FAIL] test_blend_preserves_size_and_range: expected " << expected
                  << " got " << heuristic[0] << "\n";
        return 1;
    }
    std::cout << "[PASS] test_blend_preserves_size_and_range\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 5: coarsening factor of 1 produces a coarse grid of approximately full size
// ---------------------------------------------------------------------------
static int test_coarsening_factor_one() {
    constexpr std::size_t NX = 16;
    constexpr std::size_t NY = 16;

    shardsim::SimulationConfig cfg = make_base_config(NX, NY);
    cfg.presim_steps = 20;
    cfg.presim_coarsening_factor = 1;  // no downscaling

    const auto grid = shardsim::mesh::make_non_uniform_grid(cfg);
    const auto result = shardsim::presim::run_presim(grid, cfg);

    if (result.nx != NX || result.ny != NY) {
        std::cerr << "[FAIL] test_coarsening_factor_one: wrong size\n";
        return 1;
    }
    std::cout << "[PASS] test_coarsening_factor_one\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Test 6: blend_presim_scores is a no-op when UncertaintyMap is empty
// ---------------------------------------------------------------------------
static int test_blend_noop_on_empty_map() {
    constexpr std::size_t N = 8;
    shardsim::presim::UncertaintyMap empty_map;  // default-constructed: scores empty

    std::vector<double> heuristic(N * N, 0.42);
    const std::vector<double> original = heuristic;

    shardsim::presim::blend_presim_scores(heuristic, empty_map, N, N, 0.5);

    if (heuristic != original) {
        std::cerr << "[FAIL] test_blend_noop_on_empty_map: scores changed unexpectedly\n";
        return 1;
    }
    std::cout << "[PASS] test_blend_noop_on_empty_map\n";
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    int failures = 0;
    failures += test_disabled_returns_empty();
    failures += test_output_dimensions();
    failures += test_scores_in_unit_interval();
    failures += test_blend_preserves_size_and_range();
    failures += test_coarsening_factor_one();
    failures += test_blend_noop_on_empty_map();

    if (failures == 0) {
        std::cout << "All presim tests passed.\n";
    } else {
        std::cerr << failures << " presim test(s) failed.\n";
    }
    return failures;
}
