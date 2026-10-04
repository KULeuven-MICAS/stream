# Stages

Stream's mapping flow is a **pipeline of stages**. Each stage does one job - parse an input, generate tilings, estimate cost, run the MILP allocation - and passes shared state to the next through a `StageContext`. This makes the flow easy to read, configure, and extend.

The framework lives in `stream/stages/`.

---

## Execution model

- **`Stage`** - the base unit of work. A **`LeafStage`** does work and yields results; a **`MainStage`** owns an ordered list of sub-stages and runs them as a pipeline.
- **`StageContext`** (`stream/stages/context.py`) - the shared, mutable state threaded through the run. Inputs (hardware/workload/mapping paths, backend, output path) go in; results (`total_latency`, `group_latencies`, `allocation`, `workload`, `accelerator`, …) come out. You read results with `ctx.get("…")`.

The public API functions in `stream/api.py` assemble the right stage list for you - you normally don't build a `MainStage` by hand.

---

## The CO pipeline

`evaluate_mapping`, `select_mapping` and `generate_code` run one pipeline:

1. **`AcceleratorParserStage`** - the accelerator model (cores, memories, interconnect), already parsed from the hardware YAML so the plugins below can be chosen for it. The workload comes from the frontend that loads it (see [Workload](workload.md)).
2. **Mapping** - either:
   - **`FixedMappingGenerationStage`** (a mapping was given) - split the workload at the mapping's fused groups and build each group's mapping; or
   - the stages of the **mapping generator** that claims the accelerator (no mapping given) - for any accelerator, `ExpandNormalizationStage` + `GenericMappingGenerationStage` propose one.
3. **`FusionGroupIterationStage`** - run the stages below once per fused group, on that group's workload and mapping.
4. The stage of the **code generation backend** that claims the accelerator, for `generate_code` only - for AIE2 arrays, `AIECodeGenerationStage`.
5. **`PlacementGenerationStage`** - place a group whose kernels are known but whose cores are not, and offer one variant per other compiled block.
6. **`KernelStateStage`** - the state a kernel carries between iterations.
7. **`TileSearchStage`** - with `tile_search`, price the tile candidates around the mapping's seed and keep the fastest.
8. **`TilingGenerationStage`** - generate the intra-/inter-core tilings for each node.
9. **`CoreCostEstimationStage`** - estimate per-(node, core) cost through the core-cost backend that claims each core.
10. **`SteadyStateLoweringStage`** - lower the group to its steady state (`stream.allocation.lowering`): make the transfers explicit with the placements and routes each may take, and fix the iteration spaces and timeslots, as a `SteadyStateProblem`.
11. **`AllocationStage`** - build the MILP (`TransferAndTensorAllocator`, TETRA) for that problem from its [constraint families](#constraint-families) and solve it: decide tensor placement and transfer paths, producing the `SteadyStateSchedule` the context carries as `allocation`.
12. **`MemoryAccessesEstimationStage`** - estimate memory traffic for the chosen allocation.

The mapping generators (`stream.mapping_generators`) and code generation backends (`stream.codegen_backends`) are entry-point groups: an object with a `name`, a `priority`, a `claims(accelerator)` predicate and `stages()` or `stage()` extends the pipeline for a new kind of hardware, the highest priority among those that claim it winning.

---

## Constraint families

The allocation model is built from constraint families, each a group of constraints and the quantities they define, registered in the `stream.constraint_families` entry-point group. `SolveOptions(families=...)` lists the families of a solve, by name or as `{name: options}`, the options being the family's constructor arguments. Without it a solve builds the default set, `stream.api.default_families(hardware)`: Stream's own families and those of each core namespace the accelerator has, such as `aie2`'s.

| Family | What it constrains | Options |
|--------|--------------------|---------|
| `placement` | each movable tensor takes one of its placements | |
| `path_choice` | each transfer takes one route, whose ends hold the tensors it moves; the route length is the last objective level | |
| `reuse_rates` | how many iterations one firing of a transfer serves | |
| `link_contention` | a link carries at most one transfer per slot | |
| `memory_capacity` | what each memory holds fits in its capacity; a memory too small for the tensors pinned to it fails the solve before the model is built | |
| `object_fifo_depth` | the object-fifo depth each core's tensors need; the buffering depth is an objective level | |
| `buffer_descriptors` | the buffer descriptors each core's transfers need | |
| `slot_latency` | a slot lasts as long as the slowest node or transfer in it | |
| `reuse_levels`, `output_reuse` | a tensor handed between cores, and a final output, are held up to their outermost irrelevant loop | |
| `reuse_compatibility` | the reuse levels on either side of a memory-to-compute transfer agree | |
| `spatial_reuse` | reuse covers every temporal loop inside a tensor's outermost spatial loop | |
| `overlap` | how much of an iteration the next one overlaps, the fill before the first, and the latency they add up to | `model` (`occupancy` or `span`), `transfer_contention`, `offchip_contention` |
| `dma_channels` | the DMA channels each core drives, whose peaks the latency objective charges | |
| `offchip_traffic` | the bits crossing the off-chip boundary, an objective level and charged in the latency objective | |
| `aie2_object_fifo_depth`, `aie2_buffer_descriptors` | an AIE2 tile's fifo depth and buffer descriptors stay within `max_object_fifo_depth` | |
| `aie2_memory_reuse` | a memory tile outlives its reader only where one replay expresses the re-read | |
| `aie2_dma_channels` | a tile drives at most its DMA channels in each direction | `max_compute_tile_dma_channels` (2), `max_mem_tile_dma_channels` (6), `max_shim_tile_dma_channels` (2) |
| `memory_ports` | each memory port's bits fit in its rate, see [Memory ports](hardware.md#memory-ports); not in the default set | `interval`, `burst` |

A family declares the quantities it `requires` and those it `provides`, and the builder runs each one after every family that provides what it requires: Stream's own families keep the order of the table, and any other family runs as soon as its requirements allow, so a namespace's limit lands right after the quantity it bounds. A family that defines quantities others read before its own constraints can be built, as `memory_ports` declares the slot latencies its ports can force before the overlap is built, does that in a separate `declare` step. A selection that leaves out what a family requires is rejected, so switching a constraint group off means leaving out its family and those that need it, which `default_families(hardware, without=[...])` does:

```python
from stream.api import SolveOptions, default_families

hardware = "stream/inputs/aie/hardware/whole_array_strix.yaml"
no_dma = SolveOptions(families=default_families(hardware, without=["dma_channels"]))  # also drops aie2_dma_channels
span = SolveOptions(families=[*default_families(hardware, without=["overlap"]), {"overlap": {"model": "span"}}])
ports = SolveOptions(families=[*default_families(hardware), "memory_ports"])
```

A family's `build(ctx, q)` (and `declare`) receives a `FormulationContext`: `ctx.space`, the read-only problem and the choices derived from it (each tensor's placements, each transfer's routes and the links they use, each tensor's reuse stops and the tiles they hold); `ctx.vars`, the core decision variables (`x` places a tensor, `y` routes a transfer, `z_stop` stops a tensor's reuse, `z_single` holds its window in one buffer, `slot_latency`); `ctx.model`, the `SolverModel`; `ctx.quantities`, the `QuantityRegistry` passed as `q`; and the modelling helpers families share, such as `binary_product` and `tensor_uses_core_var`. A family that creates a constraint with `ctx.add_constr(expr, name=..., resource=..., kind=..., subject=..., rule=..., bound=...)` states what it stands for: the core or link it binds, the hardware limit (`ResourceKind`) it is part of and that limit's `bound`, the tensor whose demand it carries, or the `StructuralRule` it enforces. When a model has no solution, the diagnosis maps the solver's IIS back to cores, links and causes through these tags alone. A family can also contribute to the objective: `objective(ctx, q)` runs once every family has built and returns `ObjectiveLevel`s; the levels of one name are summed and the solve minimizes them lexicographically, highest priority first. Stream's levels are `latency` (priority 4: the run's latency from `overlap`, the DMA peaks from `dma_channels` and the weighted off-chip traffic from `offchip_traffic`), `offchip_traffic` (3), `buffering` (2, from `object_fifo_depth`) and `route_hops` (1, from `path_choice`). And `screen(ctx)`, run before any family builds, fails a solve its constraints cannot satisfy by raising an `InfeasibleAllocationError`, as `memory_capacity` does.

A core namespace (a `NamespaceConstraints` in the `stream.constraints` group) contributes its families by naming them in its `families`, next to the facts the model reads of it: which cores share memory, what the toolchain reserves, and what a dispatch of several designs costs.

---

## Writing a custom stage

To add behaviour, subclass `Stage` (or `LeafStage`), accept the downstream stages as your sub-stage list, and yield `(result, info)` tuples as you iterate them:

```python
from stream.stages.stage import LeafStage

class MyStage(LeafStage):
    def run(self):
        for result, info in self.sub_stage.run():
            # transform / measure / filter here
            yield result, info
```

Insert your stage at the right position in the list passed to `MainStage`. If your stage reduces (keeps only the best result), `yield` once **after** the loop rather than inside it.
