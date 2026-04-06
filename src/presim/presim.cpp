#include "shardsim/presim/presim.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace shardsim::presim {

namespace {

// ---------------------------------------------------------------------------
// Coarsened grid helpers
// ---------------------------------------------------------------------------

struct CoarseGrid {
    std::size_t nx {1};
    std::size_t ny {1};
    double dx {1.0};
    double dy {1.0};
};

CoarseGrid make_coarse_grid(std::size_t full_nx, std::size_t full_ny,
                             std::size_t coarsening) {
    const std::size_t factor = std::max(std::size_t{1}, coarsening);
    CoarseGrid g;
    g.nx = std::max(std::size_t{2}, full_nx / factor);
    g.ny = std::max(std::size_t{2}, full_ny / factor);
    g.dx = static_cast<double>(full_nx) / static_cast<double>(g.nx);
    g.dy = static_cast<double>(full_ny) / static_cast<double>(g.ny);
    return g;
}

inline std::size_t idx(std::size_t i, std::size_t j, std::size_t ny) {
    return i * ny + j;
}

// ---------------------------------------------------------------------------
// Minimal single-rank explicit heat solve on a uniform coarse grid.
// Boundary conditions: Dirichlet 0 on x=0 row, Neumann zero-flux elsewhere.
// The source cell is set to source_temperature and held fixed each step.
// ---------------------------------------------------------------------------
std::vector<double> run_coarse_heat(const CoarseGrid& g,
                                    std::size_t steps,
                                    double alpha,
                                    double source_x_frac,
                                    double source_y_frac,
                                    double source_temp) {
    const std::size_t nx = g.nx;
    const std::size_t ny = g.ny;

    // dt chosen to be CFL-stable.
    const double h2 = std::min(g.dx * g.dx, g.dy * g.dy);
    const double dt = 0.24 * h2 / std::max(std::abs(alpha), 1.0e-12);

    std::vector<double> cur(nx * ny, 0.0);
    std::vector<double> nxt(nx * ny, 0.0);

    // Single interior source cell.
    const std::size_t src_i = static_cast<std::size_t>(
        std::clamp(source_x_frac, 0.0, 1.0) * static_cast<double>(nx - 1));
    const std::size_t src_j = static_cast<std::size_t>(
        std::clamp(source_y_frac, 0.0, 1.0) * static_cast<double>(ny - 1));
    // Avoid placing source at i=0 (Dirichlet zero column).
    const std::size_t src_i_safe = std::max(std::size_t{1}, src_i);
    cur[idx(src_i_safe, src_j, ny)] = source_temp;

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t i = 1; i + 1 < nx; ++i) {
            // Neumann at j=0 and j=ny-1 (zero-flux): use one-sided diff.
            // Interior: standard 5-point Laplacian.
            for (std::size_t j = 1; j + 1 < ny; ++j) {
                const double t  = cur[idx(i, j, ny)];
                const double d2x = (cur[idx(i + 1, j, ny)] - 2.0 * t + cur[idx(i - 1, j, ny)])
                                   / (g.dx * g.dx);
                const double d2y = (cur[idx(i, j + 1, ny)] - 2.0 * t + cur[idx(i, j - 1, ny)])
                                   / (g.dy * g.dy);
                nxt[idx(i, j, ny)] = t + dt * alpha * (d2x + d2y);
            }
            // j = 0: Neumann (zero flux), mirror from j=1.
            nxt[idx(i, 0, ny)] = cur[idx(i, 1, ny)];
            // j = ny-1: Neumann, mirror from j=ny-2.
            nxt[idx(i, ny - 1, ny)] = cur[idx(i, ny - 2, ny)];
        }

        // x = 0 column: Dirichlet zero.
        for (std::size_t j = 0; j < ny; ++j) {
            nxt[idx(0, j, ny)] = 0.0;
        }

        // x = nx-1 column: Neumann (extrapolate from interior).
        for (std::size_t j = 0; j < ny; ++j) {
            nxt[idx(nx - 1, j, ny)] = nxt[idx(nx - 2, j, ny)];
        }

        // Re-inject source.
        nxt[idx(src_i_safe, src_j, ny)] = source_temp;

        std::swap(cur, nxt);
    }

    return cur;
}

// ---------------------------------------------------------------------------
// Compute normalised uncertainty scores from a field.
// score(i,j) = 0.5 * norm(grad T) + 0.5 * abs(laplacian T)
// Both components are normalised by their respective domain maxima.
// ---------------------------------------------------------------------------
std::vector<double> compute_uncertainty_scores(const std::vector<double>& field,
                                               const CoarseGrid& g) {
    const std::size_t nx = g.nx;
    const std::size_t ny = g.ny;

    std::vector<double> scores(nx * ny, 0.0);

    double max_grad = 1.0e-12;
    double max_lap  = 1.0e-12;

    for (std::size_t i = 1; i + 1 < nx; ++i) {
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            const double t   = field[idx(i, j, ny)];
            const double dTx = (field[idx(i + 1, j, ny)] - field[idx(i - 1, j, ny)]) / (2.0 * g.dx);
            const double dTy = (field[idx(i, j + 1, ny)] - field[idx(i, j - 1, ny)]) / (2.0 * g.dy);
            const double lap = (field[idx(i + 1, j, ny)] - 2.0 * t + field[idx(i - 1, j, ny)]) / (g.dx * g.dx)
                             + (field[idx(i, j + 1, ny)] - 2.0 * t + field[idx(i, j - 1, ny)]) / (g.dy * g.dy);

            const double grad_mag = std::sqrt(dTx * dTx + dTy * dTy);
            scores[idx(i, j, ny)] = grad_mag;
            max_grad = std::max(max_grad, grad_mag);
            max_lap  = std::max(max_lap,  std::abs(lap));

            (void)max_lap;  // used below
            scores[idx(i, j, ny)] = (grad_mag + std::abs(lap)) * 0.5;
        }
    }

    // Re-normalise to [0, 1].
    double max_score = *std::max_element(scores.begin(), scores.end());
    if (max_score < 1.0e-12) {
        max_score = 1.0;
    }
    for (auto& s : scores) {
        s /= max_score;
    }
    return scores;
}

// ---------------------------------------------------------------------------
// Bilinear up-sample from (cnx, cny) to (fnx, fny).
// ---------------------------------------------------------------------------
std::vector<double> upsample(const std::vector<double>& coarse,
                              std::size_t cnx, std::size_t cny,
                              std::size_t fnx, std::size_t fny) {
    std::vector<double> fine(fnx * fny, 0.0);

    for (std::size_t fi = 0; fi < fnx; ++fi) {
        for (std::size_t fj = 0; fj < fny; ++fj) {
            // Map fine coordinates to coarse float coords.
            const double cx_f = static_cast<double>(fi) * static_cast<double>(cnx - 1)
                                / static_cast<double>(fnx > 1 ? fnx - 1 : 1);
            const double cy_f = static_cast<double>(fj) * static_cast<double>(cny - 1)
                                / static_cast<double>(fny > 1 ? fny - 1 : 1);

            const std::size_t cx0 = static_cast<std::size_t>(cx_f);
            const std::size_t cy0 = static_cast<std::size_t>(cy_f);
            const std::size_t cx1 = std::min(cx0 + 1, cnx - 1);
            const std::size_t cy1 = std::min(cy0 + 1, cny - 1);

            const double tx = cx_f - static_cast<double>(cx0);
            const double ty = cy_f - static_cast<double>(cy0);

            const double v00 = coarse[idx(cx0, cy0, cny)];
            const double v10 = coarse[idx(cx1, cy0, cny)];
            const double v01 = coarse[idx(cx0, cy1, cny)];
            const double v11 = coarse[idx(cx1, cy1, cny)];

            fine[idx(fi, fj, fny)] = (1.0 - tx) * (1.0 - ty) * v00
                                   +         tx  * (1.0 - ty) * v10
                                   + (1.0 - tx) *         ty  * v01
                                   +         tx  *         ty  * v11;
        }
    }
    return fine;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

UncertaintyMap run_presim(const mesh::Grid2D& full_grid,
                          const SimulationConfig& config) {
    if (config.presim_steps == 0) {
        return {};  // empty → caller uses heuristic-only path
    }

    const std::size_t fnx = full_grid.size.x;
    const std::size_t fny = full_grid.size.y;
    if (fnx < 2 || fny < 2) {
        return {};
    }

    const CoarseGrid cg = make_coarse_grid(fnx, fny, config.presim_coarsening_factor);

    const std::vector<double> coarse_field = run_coarse_heat(
        cg,
        config.presim_steps,
        config.alpha,
        config.source_x_fraction,
        config.source_y_fraction,
        config.source_temperature);

    const std::vector<double> coarse_scores = compute_uncertainty_scores(coarse_field, cg);

    const std::vector<double> fine_scores = upsample(coarse_scores, cg.nx, cg.ny, fnx, fny);

    UncertaintyMap out;
    out.scores = fine_scores;
    out.nx = fnx;
    out.ny = fny;
    return out;
}

void blend_presim_scores(std::vector<double>& heuristic_scores,
                         const UncertaintyMap& presim_map,
                         std::size_t nx,
                         std::size_t ny,
                         double weight) {
    if (presim_map.scores.empty()) {
        return;  // no pre-sim data → pass through
    }
    if (presim_map.nx != nx || presim_map.ny != ny) {
        throw std::runtime_error("presim UncertaintyMap size does not match heuristic scores grid");
    }
    if (heuristic_scores.size() != nx * ny) {
        throw std::runtime_error("heuristic_scores size mismatch in blend_presim_scores");
    }

    const double w = std::clamp(weight, 0.0, 1.0);
    for (std::size_t k = 0; k < heuristic_scores.size(); ++k) {
        heuristic_scores[k] = (1.0 - w) * heuristic_scores[k] + w * presim_map.scores[k];
    }
}

}  // namespace shardsim::presim
