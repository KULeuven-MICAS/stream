# Stages

Stream's mapping flow is a **pipeline of stages**. Each stage does one job - parse an input, generate tilings, estimate cost, run the MILP allocation - and passes shared state to the next through a `StageContext`. This makes the flow easy to read, configure, and extend.

The framework lives in `stream/stages/`.

---

## Execution model

- **`Stage`** - the base unit of work. Each stage runs the stages after it in its `list_of_callables` and yields what they yield; a **`LeafStage`** ends the list. A **`MainStage`** takes the ordered list and a context and runs the pipeline.
- **`StageContext`** (`stream/stages/context.py`) - the fields the stages hand each other. Inputs (hardware, workload, mapping path, backend, output path) go in; results (`group_cycles`, `group_latencies`, `allocation`, `workload`, `accelerator`, ...) come out. You read results with `ctx.get("...")`.

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
10. **`SteadyStateLoweringStage`** - lower the group to its steady state (`stream.allocation.lowering`): make the transfers explicit with the placements and routes each may take, and fix the iteration spaces and timeslots, as an `AllocationProblem`.
11. **`AllocationStage`** - build the MILP (`AllocationModel`) for that problem from its [constraint families](#constraint-families) and solve it: decide tensor placement and transfer paths, producing the `Allocation` the context carries as `allocation`.
12. **`MemoryAccessesEstimationStage`** - estimate memory traffic for the chosen allocation.

The mapping generators (`stream.mapping_generators`) and code generation backends (`stream.codegen_backends`) are entry-point groups: an object with a `name`, a `priority`, a `claims(accelerator)` predicate and `stages()` or `stage()` extends the pipeline for a new kind of hardware, the highest priority among those that claim it winning.

### Windowed operators

A convolution or a pool reads each spatial axis of its input through a sliding window, `s*o + d*f - p` of its
output dim `o` and kernel dim `f`; the same per-axis strides, dilations and padding serve every windowed parser and
frontend. The pipeline handles such a read like any other affine access:

- **Couplings.** A reader that indexes a producer's axis as `s*o + windows + c` merges the producer's dim into its
  own `o`, with a remainder dim of size `s` when `s > 1`, so fused convolutions and pools share their row and column
  axes; the window and the constant stay in the access maps.
- **Halos.** A tile's footprint is its interior window, not clipped at the tensor's origin, and a transfer's copy
  holds the window of the node it reaches. Each loop that slides that window carries its halo, the window less its
  step, on its `IterationVariable`: the rows a fused loop shares between iterations, the columns cores share.
- **Line buffer.** A sliding loop keeps its halo resident: the copy holds the window (times its buffering) and the
  transfer moves only the step, an input's tile per iteration being what its readers' windows advance by. A target
  core is fed by every source core whose tile its window overlaps, and the halo it takes from a neighbour counts on
  the links and DMA channels like any transfer. ZigZag costs the interior tile, which needs no border padding.
- **Boundary iterations.** A producer's first tile also covers what its reader's first window reaches ahead (conv1
  computes 9, 8, 8, 7 rows under a 3x3 conv tiled to 8 rows), and its last is shorter
  (`Workload.get_sliding_work`). The fill adds what the longer first tiles cost at the interior rate per element: a
  reader waits for its transfer's, and a node's readers on cores it does not run on wait for its; the shorter last
  tiles end behind the sink's interior one.

---

## Constraint families

The allocation model is built from constraint families, each a group of constraints and the quantities they define. Stream's own families are built in; another package registers its own in the `stream.constraint_families` entry-point group. `SolveOptions(families=...)` lists the families of a solve, by name or as `{name: options}`, the options being the family's constructor arguments. Without it a solve builds the default set, `stream.api.default_families(hardware)`: Stream's own families and those of each core namespace the accelerator has, such as `aie2`'s.

| Family | What it constrains | Options |
|--------|--------------------|---------|
| `placement` | each movable tensor takes one of its placements | |
| `path_choice` | each transfer takes one route, whose ends hold the tensors it moves; the route length is the last objective level | |
| `reuse_rates` | how many iterations one firing of a transfer serves | |
| `link_contention` | a link carries at most one transfer per slot | |
| `memory_capacity` | what each memory holds fits in its capacity | |
| `object_fifo_depth` | the object-fifo depth each core's tensors need, and the buffering depth, an objective level | `depth` (without it only the buffering level) |
| `buffer_descriptors` | the buffer descriptors each core's transfers need | |
| `slot_latency` | a slot lasts as long as the slowest node or transfer in it | |
| `reuse_levels`, `output_reuse` | a tensor handed between cores, and a final output, are held up to their outermost irrelevant loop | |
| `reuse_compatibility` | the reuse levels on either side of a memory-to-compute transfer agree | |
| `spatial_reuse` | reuse covers every temporal loop inside a tensor's outermost spatial loop | |
| `overlap` | how much of an iteration the next one overlaps, the fill before the first, and the latency they add up to | `model` (`occupancy` or `span`), `transfer_contention`, `offchip_contention` |
| `dma_channels` | the DMA channels each core drives, whose peaks the latency objective charges | |
| `offchip_traffic` | the bits crossing the off-chip boundary, an objective level and charged in the latency objective | `charge` (without it only the level) |
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

A family's `build(ctx)` (and `declare`) receives a `FormulationContext`: `ctx.space`, the read-only problem and the choices derived from it (each tensor's placements, each transfer's routes and the links they use, each tensor's reuse stops and the tiles they hold); `ctx.vars`, the core decision variables (`x` places a tensor, `y` routes a transfer, `z_stop` stops a tensor's reuse, `z_single` holds its window in one buffer, `slot_latency`); `ctx.model`, the `SolverModel`; `ctx.quantities`, the `QuantityRegistry` of the quantities the families provide; and the modelling helpers families share, such as `binary_product` and `tensor_uses_core_var`. A family that creates a constraint with `ctx.add_constr(expr, name=..., resource=..., kind=..., subject=..., rule=..., bound=...)` states what it stands for: the core or link it binds, the hardware limit (`ResourceKind`) it is part of and that limit's `bound`, the tensor whose demand it carries, or the `StructuralRule` it enforces. When a model has no solution, the diagnosis maps the solver's IIS back to cores, links and causes through these tags alone. A family can also contribute to the objective: `objective(ctx)` runs once every family has built and returns `ObjectiveLevel`s; the levels of one name are summed and the solve minimizes them lexicographically, highest priority first. Stream's levels are `latency` (priority 4: the run's latency from `overlap`, the DMA peaks from `dma_channels` and the weighted off-chip traffic from `offchip_traffic`), `offchip_traffic` (3), `buffering` (2, from `object_fifo_depth`) and `route_hops` (1, from `path_choice`). A family's `report(ctx)` adds sections to the solved allocation's performance report; one that fails is logged and leaves its section None. Before any family builds, the model checks that every memory fits the tensors pinned to it under some reuse choice, whichever families a solve selects, and fails one that cannot with an `InfeasibleAllocationError`.

A core namespace (a `HardwareNamespace`) contributes its families by naming them in its `families`, next to the facts the model reads of it: which cores share memory, what the toolchain reserves, and what a dispatch of several designs costs. Another package plugs into the allocation model through two entry-point groups:

- `stream.constraint_families` - a family factory under the family's `name`, whose keyword arguments are its options.
- `stream.namespaces` - a `HardwareNamespace` subclass under the core namespace it describes, built with `from_config` whenever the accelerator has a core of that namespace.

Stream's own families and its `aie2` namespace are built in, so an install whose entry points are stale still has them.

### Migrating from Stream 1.x

Stream 2.0 replaces `ConstraintSelection` with the family list; each of its toggles has an exact equivalent, which builds the same model:

| `ConstraintSelection(...)` | `SolveOptions(families=default_families(hardware, without, options))` |
|----------------------------|-------------------------------------------------------------------------|
| `memory_capacity=False` | `without=["memory_capacity"]` (the capacity screen still runs, as before) |
| `object_fifo_depth=False` | `options={"object_fifo_depth": {"depth": False}}` (the buffering level stays, as before) |
| `buffer_descriptors=False` | `without=["buffer_descriptors"]` |
| `dma_channels=False` | `without=["dma_channels"]` |
| `transfer_contention=False`, `offchip_contention=False` | `options={"overlap": {"transfer_contention": False, "offchip_contention": False}}` |
| `offchip_traffic_cost=False` | `options={"offchip_traffic": {"charge": False}}` (the traffic level stays, as before) |
| `pipelining=PipeliningModel.SPAN` | `options={"overlap": {"model": "span"}}` |
| `families=[...]` | the default set plus those families, `[*default_families(hardware), ...]` |

The rest of the 2.0 changes an extension meets: the `stream.constraints` entry-point group is `stream.namespaces` (an entry point left in the old group is ignored, with a warning), and `NamespaceConstraints` is `HardwareNamespace` (`AIE2Constraints`, `AIE2Namespace`), whose `add_*_constraints` hooks are families now (a namespace that still defines one is rejected with the family that replaces it); the context carries the solved `allocation` (an `Allocation`, read by `AllocationIR.from_internal`) instead of the `scheduler`; `AllocationIR` lists its `families` instead of a `constraint_selection`; and a stage declares its contract (below) instead of `REQUIRED_FIELDS`.

---

## Stage contracts

A stage declares the context fields it touches, as tuples of field names on the class:

- `reads` - the fields it needs when it starts; each must be in the context it is given or written by a stage before it, and not None.
- `optional_reads` - the fields it reads when they are there, such as the options a plugin's stages take through `SolveOptions(stage_options=...)`.
- `writes` - the fields it sets before the stages after it run.
- `result_reads`, `result_writes` - the fields it reads and sets on the context the stages after it yield, as `FusionGroupIterationStage` does with each group's `allocation`.

`MainStage` checks a pipeline's contracts before it runs anything: following the list, and through the stages a stage runs after it (the inner pipelines of `FusionGroupIterationStage` and `TileSearchStage` are the rest of the list), each read must be in the initial context or in the writes of a stage before it, and each result read in what the stages after it write. A pipeline that breaks the rule fails with a `StageContractError` naming the stage and the field. While a stage runs, the context lets it read only the fields its contract names and write only its writes, raising a `StageContractError` otherwise; outside a stage, as when a caller reads the result, every field is open. `ctx.data` is the raw store, for a stage that snapshots and restores the whole context, as `TileSearchStage` does between candidates.

| Stage | `reads` | `optional_reads` | `writes` | `result_reads` | `result_writes` |
|-------|---------|------------------|----------|----------------|-----------------|
| `AIECodeGenerationStage` |  | `trace_size`, `trace_max_tiles`, `trace_tiles`, `trace_group`, `npu`, `group_index` |  | `allocation`, `workload`, `accelerator`, `output_path` | `module` |
| `AcceleratorParserStage` | `accelerator` | `kernel_library` | `accelerator` |  |  |
| `AllocationStage` | `allocation_problem`, `output_path`, `backend`, `families`, `time_limit_s`, `solver_log`, `artifacts` | `total_mac_ops` | `allocation`, `workload`, `mapping` |  |  |
| `CoreCostEstimationStage` | `workload`, `accelerator`, `mapping`, `loma_lpf_limit`, `output_path`, `temporal_mapping_type` | `nb_spatial_mappings_generated`, `fusion_splits`, `loma_show_progress_bar` | `cost_lut` |  |  |
| `ExpandNormalizationStage` | `workload` |  | `workload` |  |  |
| `FixedMappingGenerationStage` | `accelerator`, `workload`, `mapping_path` |  | `sub_workloads`, `sub_mappings` |  |  |
| `FusionAnalysisStage` | `workload` |  | `fusion_edges` |  |  |
| `FusionGroupIterationStage` | `accelerator`, `output_path`, `sub_workloads`, `sub_mappings` | `memory_accesses` | `workload`, `mapping`, `output_path`, `group_index` | `allocation` | `total_latency`, `group_latencies`, `group_columns`, `group_cycles`, `group_wall_times`, `group_allocations`, `group_memory_accesses` |
| `FusionProposalStage` | `workload` | `fusion_capacity_elements` | `proposed_fusion_regions` |  |  |
| `GenericMappingGenerationStage` | `accelerator`, `workload`, `output_path` | `fusion_cut_points`, `intra_core_tiling` | `sub_workloads`, `sub_mappings` |  |  |
| `KernelStateStage` | `workload`, `mapping` | `placement_alternatives`, `placement_reserves` | `workload`, `mapping`, `placement_alternatives`, `placement_reserves` |  |  |
| `LeafStage` |  |  |  |  |  |
| `MemoryAccessesEstimationStage` | `workload`, `accelerator`, `mapping`, `allocation` |  | `memory_accesses` |  |  |
| `ONNXModelParserStage` | `workload_path`, `output_path` |  | `onnx_model`, `workload` |  |  |
| `PlacementGenerationStage` | `workload`, `mapping`, `accelerator` |  | `mapping`, `placement_alternatives`, `placement_reserves` |  |  |
| `SteadyStateLoweringStage` | `workload`, `accelerator`, `mapping`, `cost_lut`, `fusion_splits`, `nb_cols_to_use` |  | `allocation_problem` |  |  |
| `StructuralDedupStage` | `workload` |  | `block_classes` |  |  |
| `TileSearchStage` | `workload`, `mapping`, `output_path` | `tile_search` | `mapping`, `output_path`, `placement_alternatives`, `placement_reserves` | `allocation` | `output_path` |
| `TilingGenerationStage` | `workload`, `mapping`, `output_path` |  | `workload`, `mapping`, `fusion_splits`, `total_mac_ops` |  |  |

A test checks this table against the stages' declarations. Constraint families follow the same rule one level down: a family declares the quantities it `requires` and `provides`, and the model builder orders and checks them (see [Constraint families](#constraint-families)).

---

## Writing a custom stage

Subclass `Stage`, declare its contract, and run the stages after it:

```python
from stream.stages.stage import Stage

class LayerCountStage(Stage):
    reads = ("workload",)
    writes = ("layer_count",)

    def run(self):
        self.ctx.set(layer_count=len(self.ctx.get("workload").get_computation_nodes()))
        yield from self.list_of_callables[0](self.list_of_callables[1:], self.ctx).run()
```

Insert it at the right position in the list passed to `MainStage`. A stage that reduces (keeps only the best of several results) yields once **after** its loop rather than inside it. Out-of-tree stages, such as those a mapping generator's `stages()` or a code generation backend's `stage()` returns, declare their contracts the same way. A stage that declares none, such as one written against Stream 1.x, and a stage callable that is not a `Stage` class, such as a factory or a `functools.partial`, run unchecked: the pipeline logs once that it cannot check them, and checks the fields of the stages after them only as they run.
