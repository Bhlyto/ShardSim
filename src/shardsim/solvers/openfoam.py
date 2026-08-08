from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from shutil import rmtree, which
from subprocess import CompletedProcess, run
from tempfile import mkdtemp
from time import perf_counter
import re

import numpy as np

from shardsim.canonical import DIFFUSIVITY, TEMPERATURE, FieldLocation
from shardsim.contracts import Fidelity, SimulationCase, SimulationResult


OPENFOAM_IMAGE = (
    "opencfd/openfoam-run:2606"
    "@sha256:4229997e74defb81548222d511b8e3b95b98305e5df41b8e88b031813fe47eeb"
)


class OpenFOAMExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenFOAMDiscretization:
    rows: int
    columns: int
    dx: float
    dy: float
    dt: float
    n_steps: int


def sample_cell_centers(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("field must be a two-dimensional structured grid.")
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must contain positive cell counts.")

    rows, columns = shape
    row_positions = (np.arange(rows, dtype=np.float64) + 0.5) * (values.shape[0] - 1) / rows
    column_positions = (np.arange(columns, dtype=np.float64) + 0.5) * (
        values.shape[1] - 1
    ) / columns
    row_lower = np.floor(row_positions).astype(int)
    column_lower = np.floor(column_positions).astype(int)
    row_upper = np.minimum(row_lower + 1, values.shape[0] - 1)
    column_upper = np.minimum(column_lower + 1, values.shape[1] - 1)
    row_weight = (row_positions - row_lower)[:, None]
    column_weight = (column_positions - column_lower)[None, :]

    top = (1.0 - column_weight) * values[np.ix_(row_lower, column_lower)] + column_weight * values[
        np.ix_(row_lower, column_upper)
    ]
    bottom = (1.0 - column_weight) * values[np.ix_(row_upper, column_lower)] + column_weight * values[
        np.ix_(row_upper, column_upper)
    ]
    return (1.0 - row_weight) * top + row_weight * bottom


def parse_openfoam_scalar_field(path: Path, shape: tuple[int, int]) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    expected_size = shape[0] * shape[1]
    uniform_match = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
    if uniform_match:
        values = np.full(expected_size, float(uniform_match.group(1)), dtype=np.float64)
    else:
        nonuniform_match = re.search(
            r"internalField\s+nonuniform\s+(?:List|Field)<scalar>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;",
            text,
            flags=re.DOTALL,
        )
        if nonuniform_match is None:
            raise ValueError(f"Cannot parse internalField from {path}.")
        declared_size = int(nonuniform_match.group(1))
        values = np.fromstring(nonuniform_match.group(2), sep=" ", dtype=np.float64)
        if declared_size != values.size:
            raise ValueError("OpenFOAM field size does not match its declared size.")
    if values.size != expected_size:
        raise ValueError(f"Expected {expected_size} OpenFOAM cells, found {values.size}.")
    return np.flipud(values.reshape(shape))


@dataclass(frozen=True, slots=True)
class OpenFOAMHeatCaseBuilder:
    temporal_factor: float = 0.25

    def discretization(
        self,
        case: SimulationCase,
        grid_shape: tuple[int, int],
        delta_t: float | None = None,
    ) -> OpenFOAMDiscretization:
        if case.problem.domain != "heat-2d":
            raise ValueError(f"OpenFOAM heat adapter cannot solve {case.problem.domain!r}.")
        if len(grid_shape) != 2 or min(grid_shape) < 2:
            raise ValueError("OpenFOAM grid_shape must contain at least 2x2 cells.")
        alpha = case.problem.parameter("alpha")
        if alpha <= 0:
            raise ValueError("Thermal diffusivity alpha must be positive.")
        rows, columns = grid_shape
        length_x, length_y = case.problem.extent
        dx = length_x / columns
        dy = length_y / rows
        requested_dt = delta_t or self.temporal_factor * min(dx * dx, dy * dy) / alpha
        if requested_dt <= 0:
            raise ValueError("delta_t must be positive.")
        n_steps = max(1, ceil(case.problem.t_end / requested_dt))
        return OpenFOAMDiscretization(
            rows=rows,
            columns=columns,
            dx=dx,
            dy=dy,
            dt=case.problem.t_end / n_steps,
            n_steps=n_steps,
        )

    def write(
        self,
        case: SimulationCase,
        grid_shape: tuple[int, int],
        case_directory: Path,
        delta_t: float | None = None,
    ) -> OpenFOAMDiscretization:
        discretization = self.discretization(case, grid_shape, delta_t)
        for directory in ("0", "constant", "system"):
            (case_directory / directory).mkdir(parents=True, exist_ok=True)

        initial_cells = sample_cell_centers(case.initial_field, grid_shape)
        files = {
            case_directory / "0" / "T": self._temperature(case, initial_cells),
            case_directory / "constant" / "transportProperties": self._transport(case),
            case_directory / "system" / "blockMeshDict": self._block_mesh(case, discretization),
            case_directory / "system" / "controlDict": self._control(case, discretization),
            case_directory / "system" / "fvSchemes": self._schemes(),
            case_directory / "system" / "fvSolution": self._solution(),
        }
        for path, content in files.items():
            path.write_text(content, encoding="utf-8", newline="\n")
        return discretization

    @staticmethod
    def _header(object_name: str, class_name: str, location: str) -> str:
        return f"""FoamFile
{{
    format      ascii;
    class       {class_name};
    location    \"{location}\";
    object      {object_name};
}}
"""

    def _temperature(self, case: SimulationCase, initial_cells: np.ndarray) -> str:
        values = np.flipud(initial_cells).ravel()
        value_lines = "\n".join(f"{value:.17g}" for value in values)
        boundaries = case.boundaries
        return self._header("T", "volScalarField", "0") + f"""
dimensions      {TEMPERATURE.to_openfoam()};
internalField   nonuniform List<scalar>
{values.size}
(
{value_lines}
);
boundaryField
{{
    top
    {{
        type fixedValue;
        value uniform {boundaries.top:.17g};
    }}
    bottom
    {{
        type fixedValue;
        value uniform {boundaries.bottom:.17g};
    }}
    left
    {{
        type fixedValue;
        value uniform {boundaries.left:.17g};
    }}
    right
    {{
        type fixedValue;
        value uniform {boundaries.right:.17g};
    }}
    frontAndBack
    {{
        type empty;
    }}
}}
"""

    def _transport(self, case: SimulationCase) -> str:
        return self._header("transportProperties", "dictionary", "constant") + f"""
DT              {DIFFUSIVITY.to_openfoam()} {case.problem.parameter('alpha'):.17g};
"""

    def _block_mesh(
        self,
        case: SimulationCase,
        discretization: OpenFOAMDiscretization,
    ) -> str:
        length_x, length_y = case.problem.extent
        thickness = min(discretization.dx, discretization.dy)
        return self._header("blockMeshDict", "dictionary", "system") + f"""
scale 1;
vertices
(
    (0 0 0)
    ({length_x:.17g} 0 0)
    ({length_x:.17g} {length_y:.17g} 0)
    (0 {length_y:.17g} 0)
    (0 0 {thickness:.17g})
    ({length_x:.17g} 0 {thickness:.17g})
    ({length_x:.17g} {length_y:.17g} {thickness:.17g})
    (0 {length_y:.17g} {thickness:.17g})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({discretization.columns} {discretization.rows} 1)
        simpleGrading (1 1 1)
);
edges ();
boundary
(
    bottom
    {{
        type wall;
        faces ((0 1 5 4));
    }}
    right
    {{
        type wall;
        faces ((1 2 6 5));
    }}
    top
    {{
        type wall;
        faces ((2 3 7 6));
    }}
    left
    {{
        type wall;
        faces ((3 0 4 7));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);
mergePatchPairs ();
"""

    def _control(self, case: SimulationCase, discretization: OpenFOAMDiscretization) -> str:
        return self._header("controlDict", "dictionary", "system") + f"""
application     laplacianFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {case.problem.t_end:.17g};
deltaT          {discretization.dt:.17g};
writeControl    timeStep;
writeInterval   {discretization.n_steps};
purgeWrite      0;
writeFormat     ascii;
writePrecision  17;
writeCompression off;
timeFormat      general;
timePrecision   12;
runTimeModifiable false;
"""

    def _schemes(self) -> str:
        return self._header("fvSchemes", "dictionary", "system") + """
ddtSchemes
{
    default Euler;
}
gradSchemes
{
    default Gauss linear;
}
divSchemes
{
    default none;
}
laplacianSchemes
{
    default none;
    laplacian(DT,T) Gauss linear orthogonal;
}
interpolationSchemes
{
    default linear;
}
snGradSchemes
{
    default orthogonal;
}
fluxRequired
{
    default no;
    T;
}
"""

    def _solution(self) -> str:
        return self._header("fvSolution", "dictionary", "system") + """
solvers
{
    T
    {
        solver PCG;
        preconditioner DIC;
        tolerance 1e-12;
        relTol 0;
    }
}
"""


@dataclass(frozen=True, slots=True)
class OpenFOAMAdapter:
    image: str = OPENFOAM_IMAGE
    docker_executable: str = "docker"
    timeout_seconds: float = 300.0
    work_root: Path | None = None
    keep_cases: bool = False
    builder: OpenFOAMHeatCaseBuilder = OpenFOAMHeatCaseBuilder()

    @property
    def adapter_id(self) -> str:
        return "openfoam-laplacian-v2606"

    @property
    def output_location(self) -> FieldLocation:
        return FieldLocation.CELL

    def supports(self, case: SimulationCase) -> bool:
        return case.problem.domain == "heat-2d" and case.problem.equation == "du/dt=alpha*laplacian(u)"

    def is_available(self) -> bool:
        if which(self.docker_executable) is None:
            return False
        completed = run(
            [self.docker_executable, "image", "inspect", self.image],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode == 0

    def solve(
        self,
        case: SimulationCase,
        fidelity: Fidelity,
        grid_shape: tuple[int, int],
    ) -> SimulationResult:
        if not self.supports(case):
            raise ValueError(f"{self.adapter_id} does not support this problem specification.")
        if which(self.docker_executable) is None:
            raise OpenFOAMExecutionError("Docker executable is not available.")

        case_directory = Path(
            mkdtemp(prefix="shardsim-openfoam-", dir=str(self.work_root) if self.work_root else None)
        )
        try:
            discretization = self.builder.write(case, grid_shape, case_directory)
            command = [
                self.docker_executable,
                "run",
                "--rm",
                "--mount",
                f"type=bind,source={case_directory.resolve()},target=/case",
                self.image,
                "bash",
                "-lc",
                "cd /case && blockMesh && laplacianFoam",
            ]
            started_at = perf_counter()
            completed = run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            runtime_seconds = perf_counter() - started_at
            self._raise_on_failure(completed)
            result_path = self._latest_field(case_directory)
            field = parse_openfoam_scalar_field(result_path, grid_shape)
            metadata: dict[str, object] = {
                "solver": "laplacianFoam",
                "solver_adapter": self.adapter_id,
                "container_image": self.image,
                "field_location": "cell",
                "alpha": case.problem.parameter("alpha"),
                "dx": discretization.dx,
                "dy": discretization.dy,
                "log_tail": "\n".join(completed.stdout.splitlines()[-12:]),
            }
            if self.keep_cases:
                metadata["case_directory"] = str(case_directory)
            return SimulationResult(
                case_id=case.case_id,
                fidelity=fidelity,
                field=field,
                t_end=case.problem.t_end,
                dt=discretization.dt,
                n_steps=discretization.n_steps,
                runtime_seconds=runtime_seconds,
                field_location=FieldLocation.CELL,
                metadata=metadata,
            )
        finally:
            if not self.keep_cases:
                rmtree(case_directory, ignore_errors=True)

    @staticmethod
    def _raise_on_failure(completed: CompletedProcess[str]) -> None:
        if completed.returncode == 0:
            return
        output = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-40:])
        raise OpenFOAMExecutionError(f"OpenFOAM failed with exit code {completed.returncode}:\n{output}")

    @staticmethod
    def _latest_field(case_directory: Path) -> Path:
        time_directories: list[tuple[float, Path]] = []
        for path in case_directory.iterdir():
            if not path.is_dir():
                continue
            try:
                time_value = float(path.name)
            except ValueError:
                continue
            if time_value > 0 and (path / "T").is_file():
                time_directories.append((time_value, path))
        if not time_directories:
            raise OpenFOAMExecutionError("OpenFOAM produced no final T field.")
        return max(time_directories, key=lambda item: item[0])[1] / "T"
