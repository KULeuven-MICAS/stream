# Installation

## Requirements

- Python **≥ 3.12**
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

This pulls in `zigzag-dse`, `ortools>=9.15` (the default, license-free MILP backend), `pydantic`, `pydot`, and `xdsl`.

**With the MCP server** (for driving Stream from an AI agent — see [Using Stream with an AI agent](ai-agents.md)):

```bash
pip install -e ".[mcp]"
```

The `[mcp]` extra adds `fastmcp`.

**With the Gurobi solver** (optional — OR-Tools GSCIP is the default, license-free backend):

```bash
pip install -e ".[gurobi]"
```

The `[gurobi]` extra adds `gurobipy` (a Gurobi license is still required at solve time; see the note below).

**With the AMD AIE toolchain** (only needed for AIE-target MLIR codegen and on-device tracing):

```bash
pip install -e .       # or, once published: pip install stream-dse
stream-setup-aie       # installs the AIE toolchain into the current environment
```

The base install carries no AIE dependencies — those are git/URL installs that PyPI does not allow in package metadata. The `stream-setup-aie` console script installs the toolchain (`mlir_aie`, `llvm-aie`, `xdsl-aie`, `snax-mlir`, and `aie-python-extras`) and writes a `.pth` file so the `mlir_aie` bindings are importable with no further `PYTHONPATH` setup. Run `stream-setup-aie --dry-run` to preview the exact steps. It is **Linux x86_64 only** (manylinux wheels), **CPython 3.12 or 3.13**. After it completes, the AIE codegen entry points and `enable_codegen=True` work; the base CO pipeline has no such restriction.

## Solver license note

The default backend `ortools_gscip` is open-source and needs **no license**. Gurobi support requires the `[gurobi]` extra (`pip install -e ".[gurobi]"`) **and** a separate commercial license: selecting `backend="gurobi"` errors at solve time without a valid license. For most users the default is all you need.

## Optional: pre-commit hooks

```bash
pip install pre-commit
pre-commit install   # runs ruff check + ruff format on every commit
```

Once installed, head to [Getting Started](getting-started.md).
