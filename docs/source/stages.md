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
10. **`ConstraintOptimizationAllocationStage`** - build and solve the MILP (`TransferAndTensorAllocator`, TETRA): decide tensor placement and transfer paths, producing the schedule.
11. **`MemoryAccessesEstimationStage`** - estimate memory traffic for the chosen allocation.

The mapping generators (`stream.mapping_generators`) and code generation backends (`stream.codegen_backends`) are entry-point groups: an object with a `name`, a `priority`, a `claims(accelerator)` predicate and `stages()` or `stage()` extends the pipeline for a new kind of hardware, the highest priority among those that claim it winning.

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
