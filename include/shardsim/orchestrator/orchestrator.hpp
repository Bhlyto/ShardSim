#pragma once

#include "shardsim/config.hpp"
#include "shardsim/metrics/metrics.hpp"

namespace shardsim::orchestrator {

class Orchestrator {
  public:
    explicit Orchestrator(SimulationConfig config);
    metrics::RunSummary run() const;

  private:
    SimulationConfig config_;
};

}  // namespace shardsim::orchestrator
