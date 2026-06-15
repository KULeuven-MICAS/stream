# Using Stream with an AI agent

Stream is built to be driven by an AI coding agent (such as Claude Code) as well as by a human. There are two layers of programmatic support: an **MCP server** that lets an agent run and inspect optimizations as a tool, and **typed IR models** for structured results.

---

## The MCP server

Stream ships an MCP (Model Context Protocol) server so an agent can submit and inspect TETRA constraint-optimization jobs as tool calls. It needs the `[mcp]` extra:

```bash
pip install -e ".[mcp]"
```

Launch it (STDIO / JSON-RPC transport) from the repo root:

```bash
python3 -c "from stream.mcp.server import mcp; mcp.run(transport='stdio')"
```

The server (`stream/mcp/server.py`, name `stream-aie`) exposes six tools:

| Tool | Purpose |
|------|---------|
| `run_optimization(hardware, workload, mapping, output_path, backend, ...)` | Submit a CO job; returns a `job_id` immediately and solves in the background. |
| `poll_optimization(job_id)` | Check status: `pending` / `running` / `complete` / `failed` / `not_found`. |
| `get_workload_ir(workload=None, experiment_id=None)` | The workload graph as `WorkloadIR` JSON. |
| `get_accelerator_ir(hardware=None, experiment_id=None)` | The hardware model as `AcceleratorIR` JSON. |
| `get_allocation_ir(job_id)` | The allocation result as `AllocationIR` JSON (three persona views). |
| `get_solve_stats(job_id)` | MILP solve statistics (objective, time, gap, node count, backend). |

Typical flow: `run_optimization(...)` → poll `poll_optimization(job_id)` until `complete` → inspect with `get_allocation_ir(job_id)` and `get_solve_stats(job_id)`.

---

## Typed IR for structured results

When an agent (or any tool) needs machine-readable output instead of console text, use the IR models. They wrap the internal objects and serialize cleanly to JSON:

```python
from stream.ir import WorkloadIR, AcceleratorIR, AllocationIR

# after running optimize_allocation_co_generic(...) -> ctx
workload_ir    = WorkloadIR.from_internal(ctx.get("workload"))
accelerator_ir = AcceleratorIR.from_internal(ctx.get("accelerator"))
allocation_ir  = AllocationIR.from_internal(ctx.get("scheduler"))

allocation_data = allocation_ir.model_dump()   # JSON-compatible dict
```

`AllocationIR` offers persona-specific views of the same result - `.algorithmic_view()`, `.hardware_view()`, `.compiler_view()` - so a consumer asks for the shape it needs (the algorithmic graph, the hardware placement, or the compiler-facing detail).

---

## In short

- **Running optimizations as a tool?** Use the MCP server.
- **Consuming results programmatically?** Use the IR models.
