from __future__ import annotations

from typing import Protocol

from shardsim.contracts import Fidelity, SimulationCase, SimulationResult


class Solver(Protocol):
    def solve(
        self,
        case: SimulationCase,
        fidelity: Fidelity,
        grid_shape: tuple[int, int],
    ) -> SimulationResult: ...


class SolverAdapter(Solver, Protocol):
    @property
    def adapter_id(self) -> str: ...

    def supports(self, case: SimulationCase) -> bool: ...

    def is_available(self) -> bool: ...
