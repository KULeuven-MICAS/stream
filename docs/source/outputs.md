# Outputs

Every entry point returns a `MappingEstimate` and writes a set of files under its output directory, one `group_<index>/` folder per fused group. This page covers both.

## The result

A `MappingEstimate` holds `cycles`, the fused groups' estimates plus the reconfiguration the hardware declares, the per-group `group_cycles`, and the solved `context`. Read the rest off the context with `ctx.get(...)`:

| Key | What it is |
|-----|-----------|
| `group_latencies` | Per-fusion-group latency breakdown. |
| `scheduler` | The `SteadyStateScheduler` - the full schedule and timing. |
| `workload` | The parsed computation graph. |
| `accelerator` | The parsed hardware model. |

```python
estimate = evaluate_mapping(...)
print(estimate.cycles)
ctx = estimate.context
scheduler = ctx.get("scheduler")
```

## Files written to disk

- **Visualizations (PNG)** - the tiling and the schedule of each fused group, written into its `group_<index>/` folder.

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

`AllocationIR` exposes persona views - `.algorithmic_view()`, `.hardware_view()`, `.compiler_view()` - each shaping the same result for a different consumer. The performance view surfaces bottleneck (compute- vs transfer-bound) cycles and per-node utilization. See [Using Stream with an AI agent](ai-agents.md) for details.
