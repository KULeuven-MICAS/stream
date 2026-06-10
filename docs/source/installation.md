# Installation

## Requirements

- Python **≥ 3.11**
- `git` and `pip`
- **Graphviz** (the `dot` binary) — the pipeline renders the workload and schedule graphs to PNG via `pydot`, which calls `dot`. Install it with your system package manager, e.g. `sudo apt-get install graphviz` (Debian/Ubuntu) or `brew install graphviz` (macOS).

## Clone

```bash
git clone https://github.com/KULeuven-MICAS/stream_aie.git
cd stream_aie
```

## Install

Stream is an installable package (`stream-dse`); install it in editable mode from the repo root. The authoritative dependency list is `pyproject.toml`.

**Base install** — everything needed for the constraint-optimization pipeline:

```bash
pip install -e .
```

This pulls in `zigzag-dse`, `ortools>=9.15` (the default MILP backend), `gurobipy`, `pydantic`, and the xDSL/SNAX MLIR packages.

**With the MCP server** (for driving Stream from an AI agent — see [Using Stream with an AI agent](ai-agents.md)):

```bash
pip install -e ".[mcp]"
```

The `[mcp]` extra adds `fastmcp`.

**With the AMD AIE toolchain** (only needed for AIE-target MLIR codegen and on-device tracing):

```bash
pip install -e ".[aie]"
bash setup_aie_python_extras.sh      # aie.extras.context — not installable as a pip extra
source setup_mlir_aie_pythonpath.sh  # adds mlir_aie/python to PYTHONPATH
```

The `[aie]` extra is **Linux x86_64 only** (manylinux wheels) and not supported on Python 3.11 — use 3.12 or 3.13 for it. The base CO pipeline has no such restriction.

## Solver license note

The default backend `ortools_gscip` is open-source and needs **no license**. Gurobi requires a separate commercial license: the package installs fine, but selecting `backend="gurobi"` errors at solve time without a valid license. For most users the default is all you need.

## Optional: pre-commit hooks

```bash
pip install pre-commit
pre-commit install   # runs ruff check + ruff format on every commit
```

Once installed, head to [Getting Started](getting-started.md).
