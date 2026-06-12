# Outputs

Every entry point returns a `StageContext`, and writes a set of files to the run's output directory (`<output_path>/<experiment_id>/`). This page covers both.

## The result context

After a run, read results off the returned context with `ctx.get(...)`:

| Key | What it is |
|-----|-----------|
| `total_latency` | Total scheduled latency (cycles) for the workload. |
| `group_latencies` | Per-fusion-group latency breakdown. |
| `scheduler` | The `SteadyStateScheduler` - the full schedule and timing. |
| `workload` | The parsed computation graph. |
| `accelerator` | The parsed hardware model. |

```python
ctx = optimize_allocation_co_generic(...)
print(ctx.get("total_latency"))     # e.g. 14344.0
print(ctx.get("group_latencies"))
scheduler = ctx.get("scheduler")
```

## Files written to disk

- **`summary.yaml`** - a machine-readable summary of the run (e.g. `total_latency`, per-group latencies).
- **Visualizations (PNG)** - workload graph, tiling, and the schedule, written into the run directory.

## Schedule trace (Perfetto)

The schedule can be exported as a Perfetto JSON trace and opened at <https://ui.perfetto.dev> to inspect each core's timeline and the inter-core transfers. See `stream/visualization/` for the trace and plotting helpers.

## Typed IR (for tools and agents)

For structured, JSON-serializable output, convert the context's objects into the typed IR models. These are the same models the [MCP server](ai-agents.md) returns:

```python
from stream.ir import WorkloadIR, AcceleratorIR, AllocationIR

workload_ir    = WorkloadIR.from_internal(ctx.get("workload"))
accelerator_ir = AcceleratorIR.from_internal(ctx.get("accelerator"))
allocation_ir  = AllocationIR.from_internal(ctx.get("scheduler"))

allocation_data = allocation_ir.model_dump()      # JSON-compatible dict
```

`AllocationIR` exposes persona views - `.algorithmic_view()`, `.hardware_view()`, `.compiler_view()` - each shaping the same result for a different consumer. The performance view surfaces bottleneck (compute- vs transfer-bound) cycles and per-node utilization. See [Using Stream with an AI agent](ai-agents.md) and the `ir` skill for details.
