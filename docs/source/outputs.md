# Outputs

Every entry point returns a `MappingEstimate` and writes a set of files under its output directory, one `group_<index>/` folder per fused group. This page covers both.

## The result

A `MappingEstimate` holds `cycles`, the fused groups' estimates plus the reconfiguration the hardware declares, the per-group `group_cycles`, and the solved `context`. Read the rest off the context with `ctx.get(...)`:

| Key | What it is |
|-----|-----------|
| `group_latencies` | Per-fusion-group latency breakdown. |
| `allocation` | The `SteadyStateSchedule` - the solved workload, mapping and iteration spaces, and its `solution` (placements, routes, latencies, solve statistics, performance report). |
| `workload` | The parsed computation graph. |
| `accelerator` | The parsed hardware model. |

```python
estimate = evaluate_mapping(...)
print(estimate.cycles)
ctx = estimate.context
schedule = ctx.get("allocation")
print(schedule.solution.latency.total)
```

## Files written to disk

- **Visualizations (PNG)** - the tiling and the schedule of each fused group, written into its `group_<index>/` folder.

## Allocation artifacts

With `SolveOptions(instrumentation={"allocation_artifacts": {}})`, each allocation solve also writes into `group_<index>/tetra/`: the schedule as Perfetto JSON traces (`steady_state_trace.json` and `steady_state_trace_compact.json`, open them at <https://ui.perfetto.dev> to inspect each core's timeline and the inter-core transfers), a picture of the solved steady-state workload (`steady_state_workload_final.svg`), the solver's progress and metrics (`optimization_progress.png`, `optimization_trace.yaml`, `optimization_metrics.yaml`) and where each slot's latency goes (`slot_latency_breakdown.yaml`). They are off by default, so a sweep pays nothing for them.

## Typed IR (for tools and agents)

For structured, JSON-serializable output, convert the context's objects into the typed IR models. These are the same models the [MCP server](ai-agents.md) returns:

```python
from stream.ir import WorkloadIR, AcceleratorIR, AllocationIR

workload_ir    = WorkloadIR.from_internal(ctx.get("workload"))
accelerator_ir = AcceleratorIR.from_internal(ctx.get("accelerator"))
allocation_ir  = AllocationIR.from_internal(ctx.get("allocation"))

allocation_data = allocation_ir.model_dump()      # JSON-compatible dict
```

`AllocationIR` exposes persona views - `.algorithmic_view()`, `.hardware_view()`, `.compiler_view()` - each shaping the same result for a different consumer. The performance view surfaces bottleneck (compute- vs transfer-bound) cycles and per-node utilization. See [Using Stream with an AI agent](ai-agents.md) for details.
