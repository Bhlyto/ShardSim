from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shardsim.solvers.openfoam import OpenFOAMAdapter
from shardsim.verification.heat import run_heat_verification


def main() -> None:
    adapter = OpenFOAMAdapter()
    if not adapter.is_available():
        raise SystemExit("Docker or the pinned OpenFOAM image is unavailable.")
    report = run_heat_verification(openfoam=adapter)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
