# Stages

Stream's mapping flow is a **pipeline of stages**. Each stage does one job - parse an input, generate tilings, estimate cost, run the MILP allocation - and passes shared state to the next through a `StageContext`. This makes the flow easy to read, configure, and extend.

The framework lives in `stream/stages/`; the deep-dive is the [`pipeline` skill](ai-agents.md) (`.claude/skills/pipeline/`).

---

## Execution model

- **`Stage`** - the base unit of work. A **`LeafStage`** does work and yields results; a **`MainStage`** owns an ordered list of sub-stages and runs them as a pipeline.
- **`StageContext`** (`stream/stages/context.py`) - the shared, mutable state threaded through the run. Inputs (hardware/workload/mapping paths, backend, output path) go in; results (`total_latency`, `group_latencies`, `scheduler`, `workload`, `accelerator`, …) come out. You read results with `ctx.get("…")`.

The public API functions in `stream/api.py` assemble the right stage list for you - you normally don't build a `MainStage` by hand.

---

## The CO pipeline

`optimize_allocation_co_generic` (auto-mapping) and `optimize_allocation_co_with_mapping` (manual mapping) run essentially these stages:

1. **`AcceleratorParserStage`** - parse the hardware YAML into the accelerator model (cores, memories, interconnect).
2. **`ONNXModelParserStage`** - parse the ONNX workload into a computation graph (see [Workload](workload.md)).
3. **Mapping** - either:
   - **`GenericMappingGenerationStage`** + **`FusionGroupIterationStage`** (generic path) - auto-generate a mapping and iterate the inner pipeline once per fusion group; or
   - **`MappingParserStage`** (manual path) - parse the supplied mapping YAML.
4. **`TilingGenerationStage`** - generate the intra-/inter-core tilings for each node.
5. **`CoreCostEstimationStage`** - estimate per-(node, core) cost. ZigZag cores are costed with ZigZag's model; AIE cores use the native AIE cost model.
6. **`ConstraintOptimizationAllocationStage`** - build and solve the MILP (`TransferAndTensorAllocator`, TETRA): decide tensor placement and transfer paths, producing the schedule.
7. **`MemoryAccessesEstimationStage`** - estimate memory traffic for the chosen allocation.

For a workload with several fusion groups, the CO runs **once per group**; this is the loop that dominates wall-clock time on large workloads.

`optimize_mapping` wraps this pipeline in an outer DSE loop that enumerates mapping variants and evaluates each one.

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

Insert your stage at the right position in the list passed to `MainStage`. If your stage reduces (keeps only the best result), `yield` once **after** the loop rather than inside it. See `.claude/skills/pipeline/` for the full contract and worked examples.
