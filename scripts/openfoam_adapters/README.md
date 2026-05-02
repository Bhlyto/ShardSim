# OpenFOAM Adapter Plugins

Adapter plugins extend case-copy adaptation in `run_openfoam_shardsim_comparison.py`.

## Contract

Each plugin file must define:

```python
def register_adapters(register_adapter):
    ...
```

`register_adapter(name, fn)` registers an adapter callable:

- `name: str`
- `fn(case_dir: Path) -> list[str]`

The callable may patch only the copied workspace case directory and should return
human-readable warning/trace messages describing applied changes.

## Example

```python
from pathlib import Path

def my_adapter(case_dir: Path) -> list[str]:
    # mutate files under case_dir as needed
    return ["my_adapter applied"]


def register_adapters(register_adapter):
    register_adapter("my_adapter", my_adapter)
```

## Strict Registry Behavior

If a solver profile references an adapter that is not registered, orchestration fails
with an explicit error listing known adapters.
