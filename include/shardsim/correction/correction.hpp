#pragma once

#include <string>

#include "shardsim/config.hpp"
#include "shardsim/types.hpp"

namespace shardsim::correction {

/// Apply a trained ML correction model to a coarse field and return the
/// corrected field: corrected[i,j] = coarse[i,j] + model_prediction[i,j].
///
/// The native linear variant (`correction_policy = "linear"`) reads a
/// `.txt` model in the same format produced by `train_linear_policy.py`:
///   patch_radius, bias, feature_mean, feature_scale, weights
/// It evaluates a local-patch Ridge regression at every interior cell and
/// adds the predicted discrepancy to the coarse value.
///
/// The Python variant (`correction_policy = "python"`) shells out to
/// `correction_script_path` (default: `scripts/apply_correction.py`) with
/// the coarse field written to a temp binary and corrections read back.
///
/// Returns the coarse field unchanged when `correction_policy = "none"` or
/// when the model path is empty.
Field2D apply_correction(const Field2D& coarse,
                         const SimulationConfig& config);

}  // namespace shardsim::correction
