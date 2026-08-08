from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from shardsim.contracts import ModelDescriptor, Prediction, SimulationCase
from shardsim.pipeline import ReferenceSample


class DeltaSurrogate(Protocol):
    @property
    def descriptor(self) -> ModelDescriptor: ...

    def fit(self, samples: Sequence[ReferenceSample]) -> ModelDescriptor: ...

    def predict(self, case: SimulationCase, coarse_on_nominal: np.ndarray) -> Prediction: ...


class PersistentDeltaSurrogate(DeltaSurrogate, Protocol):
    def save(self, path: str | Path) -> Path: ...
