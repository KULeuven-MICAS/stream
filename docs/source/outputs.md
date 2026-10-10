# Outputs

Every entry point returns a `MappingEstimate` and writes a set of files under its output directory, one `group_<index>/` folder per fused group. This page covers both.

## The result

A `MappingEstimate` holds `cycles`, the fused groups' estimates plus the reconfiguration the hardware declares, the per-group `group_cycles`, and the solved `context`. Read the rest off the context with `ctx.get(...)`:

| Key | What it is |
|-----|-----------|
| `group_latencies` | Per-fusion-group latency breakdown. |
| `allocation` | The `Allocation` - the `problem` it solves (the steady-state workload, the candidate placements and routes, the iteration spaces), the solved mapping and iteration spaces, and its `solution` (placements, routes, latencies, solve statistics and metrics, reports). |
| `workload` | The parsed computation graph. |
| `accelerator` | The parsed hardware model. |

```python
estimate = evaluate_mapping(...)
print(estimate.cycles)
ctx = estimate.context
allocation = ctx.get("allocation")
print(allocation.solution.latency.total)
```

## Files written to disk

- **Per fused group**, in its `group_<index>/` folder: a picture of its tiled workload (`tiled_workload.svg`), the cost of each node on each core (`core_cost_lut.yaml`, with its `core_cost_lut.pickle` cache) and its allocation artifacts; under a tile search, those of the candidate it chose.

## Allocation artifacts

Each allocation solve writes into `group_<index>/allocation/`:

- `reports/` - the solver's metrics (`optimization_metrics.yaml`) and, for Gurobi, its progress (`optimization_trace.yaml`), and where each slot's latency goes (`slot_latency_breakdown.yaml`).
- `traces/` - the allocation as Perfetto JSON traces (`steady_state_trace.json` and `steady_state_trace_compact.json`); open them at <https://ui.perfetto.dev> to inspect each core's timeline and the inter-core transfers.
- `figures/` - the solver's progress for Gurobi (`optimization_progress.png`) and a picture of the solved steady-state workload (`steady_state_workload_final.svg`).

A sweep that only needs the estimates turns these off with `SolveOptions(artifacts=False)`. A solve that has no solution writes its model for diagnosis instead, whatever `artifacts` says: Gurobi's irreducible infeasible subsystem as `allocation/model.ilp`, OR-Tools' whole model as `allocation/model.mps`.

## YAML reference

Every key of every YAML file a solve writes, one table per file. A nested key is a dotted path, an item of a list is `[]`, and a key that names something of the solve (a core id, a metadata entry) is in angle brackets. A test solves a small mapping and checks that these are exactly the keys the files hold. The `mapping.yaml` the generic mapping generator writes is a mapping in the input format, described in [Mapping](mapping.md).

### `core_cost_lut.yaml`

The cost of each computation node on each core it may run on, next to the `core_cost_lut.pickle` cache it summarises.

| Key | Meaning | Unit |
|-----|---------|------|
| `nodes` | the computation nodes the cost LUT holds | |
| `nodes[].node` | the node's name | |
| `nodes[].cores` | the cores the node has a cost on | |
| `nodes[].cores[].core_id` | the core's id | |
| `nodes[].cores[].core_type` | the core's namespaced type, such as `aie2.compute` | |
| `nodes[].cores[].latency_cycles` | the node's latency on the core, memory stalls included | cycles |
| `nodes[].cores[].ideal_cycles` | the latency at full use of the core's MAC units | cycles |
| `nodes[].cores[].ideal_temporal_cycles` | the latency without memory stalls, under the spatial unrolling the mapping gives the core | cycles |
| `nodes[].cores[].energy_pj` | the node's energy on the core; 0 where the backend estimates none | pJ |
| `nodes[].cores[].metadata` | what the core-cost backend records about the estimate | |
| `nodes[].cores[].metadata.backend` | the backend that estimated it: `zigzag`, `ideal-cycle` (ZigZag could not cost the pair) or `aie` | |
| `nodes[].cores[].metadata.<key>` | a further fact the backend records; `aie` records `computed_fraction` (the share of the core's steps that compute anything), and `family` (the kernel family whose rate priced it) or `measured_symbol` (the kernel whose measured call did) | |

### `allocation/reports/optimization_metrics.yaml`

The outcome of the allocation solve, on every backend, and the size of its model, which only Gurobi reports; a value the backend does not report is null.

| Key | Meaning | Unit |
|-----|---------|------|
| `status` | how the solve ended, such as `OPTIMAL` or `TIME_LIMIT` | |
| `backend` | the solver backend, such as `GUROBI` or `ORTOOLS_GSCIP` | |
| `solver` | the solver behind it, such as `gurobi` or `gscip` | |
| `search` | how much the solver searched | |
| `search.nodes_explored` | branch-and-bound nodes explored | count |
| `search.simplex_iterations` | simplex iterations | count |
| `solution` | what the solver found | |
| `solution.objective_value` | the objective of the best solution; for Stream's lexicographic objective, its highest level, the latency | objective |
| `solution.mip_gap` | the relative gap between that objective and the best bound | ratio |
| `effort` | what the solve cost | |
| `effort.runtime_s` | the solver's runtime | s |
| `model` | the size of the model | |
| `model.variables` | its variables | |
| `model.variables.total` | all variables | count |
| `model.variables.integer` | integer variables, binaries included | count |
| `model.variables.binary` | binary variables | count |
| `model.constraints` | its constraints | |
| `model.constraints.linear` | linear constraints | count |
| `model.constraints.general` | general constraints (min, max, indicator, ...) | count |
| `model.nonzeros` | nonzero coefficients of the constraint matrix | count |

### `allocation/reports/optimization_trace.yaml`

Every point a Gurobi solve's progress callback reported, in time order; other backends write no trace. A key a point's callback does not report is null.

| Key | Meaning | Unit |
|-----|---------|------|
| `trace` | the points, in time order | |
| `trace[].time_s` | the solver's runtime at the point | s |
| `trace[].event` | the callback that reported it: `PRESOLVE` (presolve progress), `MIP` (search progress) or `MIPSOL` (a new incumbent) | |
| `trace[].incumbent_objective` | the objective of the best solution so far, null before the first; a lexicographic solve optimizes its levels in turn, so the values restart at each level | objective |
| `trace[].objective_bound` | the best bound so far, null before the first | objective |
| `trace[].mip_gap` | the relative gap between the two, null while either is | ratio |
| `trace[].nodes_explored` | branch-and-bound nodes explored so far | count |
| `trace[].nodes_left` | branch-and-bound nodes still open (`MIP`) | count |
| `trace[].simplex_iterations` | simplex iterations so far (`MIP`) | count |
| `trace[].cuts_applied` | cutting planes applied so far (`MIP`) | count |
| `trace[].work_units` | Gurobi's deterministic work measure so far | work units |
| `trace[].rows_removed` | constraints presolve removed so far (`PRESOLVE`) | count |
| `trace[].columns_removed` | variables presolve removed so far (`PRESOLVE`) | count |
| `trace[].bound_changes` | variable bounds presolve changed so far (`PRESOLVE`) | count |
| `trace[].coefficient_changes` | coefficients presolve changed so far (`PRESOLVE`) | count |

### `allocation/reports/slot_latency_breakdown.yaml`

Where the latency of one steady-state iteration goes: its totals, each resource's slack, each tensor's reuse, and what sets each slot's latency.

| Key | Meaning | Unit |
|-----|---------|------|
| `totals` | the latency of the schedule | |
| `totals.iteration_latency_cycles` | one iteration, the sum of its slot latencies | cycles |
| `totals.overlap_cycles` | how much of an iteration the next one overlaps | cycles |
| `totals.initiation_interval_cycles` | the time between the starts of two iterations, the iteration latency less the overlap | cycles |
| `totals.total_latency_cycles` | the latency of every iteration together, fill included | cycles |
| `totals.shared_busy_cycles` | per core whose bandwidth its transfers share, the time it spends on one iteration's transfers | |
| `totals.shared_busy_cycles.<core_id>` | that time for the core with this id; it bounds the initiation interval from below | cycles |
| `resource_slack` | per core and link, least first, its idle time within one iteration; the overlap is at most the least of them | |
| `resource_slack[].resource` | the core or link | |
| `resource_slack[].kind` | `core` or `link` | |
| `resource_slack[].slack_cycles` | its idle time | cycles |
| `tensor_reuse` | per tensor, largest first, the reuse the solve chose | |
| `tensor_reuse[].tensor` | the tensor's name | |
| `tensor_reuse[].size_bits` | its size | bits |
| `tensor_reuse[].reuse_factor` | for how many iterations it stays resident; 1 is re-fetched every iteration | iterations |
| `tensor_reuse[].reuse_stop_level` | the loop level its reuse stops at | |
| `tensor_reuse[].on_chip_tiles` | the tiles it takes on chip at that level | count |
| `tensor_reuse[].loop_nest_out_to_in` | its loop nest, outermost first | |
| `slots` | the slots of one iteration, in order | |
| `slots[].slot` | the slot's index | |
| `slots[].slot_latency_cycles` | its latency, that of the slowest node or transfer in it | cycles |
| `slots[].compute_contributors` | the computation nodes in the slot | |
| `slots[].compute_contributors[].node` | the node's name | |
| `slots[].compute_contributors[].cost_lut_core_count` | the cores the cost LUT prices it on | count |
| `slots[].compute_contributors[].lut_latency_cycles` | its latency on the slowest of them | cycles |
| `slots[].compute_contributors[].active_fraction` | the share of the iterations it is not idle on a loop it is absent from | ratio |
| `slots[].compute_contributors[].active_latency_cycles` | its LUT latency times its active fraction | cycles |
| `slots[].transfer_contributors` | the transfers in the slot, on the routes the solve chose | |
| `slots[].transfer_contributors[].transfer` | the transfer's name | |
| `slots[].transfer_contributors[].tensor_bits` | the size of the tensor it moves | bits |
| `slots[].transfer_contributors[].min_link_bandwidth_bits_per_cycle` | the bandwidth of the narrowest link on its route | bits/cycle |
| `slots[].transfer_contributors[].path_cycles` | the time its route takes to move the tensor | cycles |
| `slots[].transfer_contributors[].active_latency_cycles` | the route time times its active fraction | cycles |
| `slots[].transfer_contributors[].reuse_factor` | the iterations one firing of it serves | iterations |
| `slots[].transfer_contributors[].latency_contribution_cycles` | what it adds to the slot's latency | cycles |

## Typed IR (for tools and agents)

For structured, JSON-serializable output, convert the context's objects into the typed IR models. These are the same models the [MCP server](ai-agents.md) returns:

```python
from stream.ir import WorkloadIR, AcceleratorIR, AllocationIR

workload_ir    = WorkloadIR.from_internal(ctx.get("workload"))
accelerator_ir = AcceleratorIR.from_internal(ctx.get("accelerator"))
allocation_ir  = AllocationIR.from_internal(ctx.get("allocation"))

allocation_data = allocation_ir.model_dump()      # JSON-compatible dict
```

`AllocationIR` exposes persona views - `.algorithmic_view()`, `.hardware_view()`, `.compiler_view()` - each shaping the same result for a different consumer. The performance view surfaces bottleneck (compute- vs transfer-bound) cycles and per-node utilization, and, under `layouts`, every transfer that lays its tile out anew on its chosen route: the axis orders it reads and writes (`source_order`, `target_order`, outermost first) and its contiguous bytes per run on each side (see [Data layout](data_layout.md)). See [Using Stream with an AI agent](ai-agents.md) for details.
