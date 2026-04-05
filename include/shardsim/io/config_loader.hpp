#pragma once

#include <string>

#include "shardsim/config.hpp"

namespace shardsim::io {

SimulationConfig load_config(const std::string& path);

}  // namespace shardsim::io
