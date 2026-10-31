# Mapping

The mapping ties a [workload](workload.md) to a [hardware](hardware.md) system: it says **which cores** each operator may run on and **how its work is split** across them. The constraint-optimization (CO) pipeline then uses this as the search space and decides the concrete tensor placement and routing.

There are two things to keep distinct:

- The **spatial mapping (dataflow)** - how a single operator is unrolled across a core's MAC array - lives on the **core**, in the hardware file (`operational_array.dimensions` / `sizes`). It is a property of the core, not of the mapping file.
- The **mapping file** decides **core allocation**, **inter-core tiling** (splitting an operator across multiple cores), and **intra-core tiling / fusion** (how operators group and tile within a core).

A mapping can be **auto-generated** by the pipeline or **hand-written**. The format is validated by `stream/parser/mapping_validator.py`.

---

## Auto-generated mapping (the default)

If you don't pass a mapping, or pass one with only `runtime_args` (see [Where inputs and outputs live](#where-inputs-and-outputs-live)), `evaluate_mapping` asks the mapping generator that claims the hardware for one; for any accelerator that is the generic generator. This is the recommended starting point.

The generic generator (`stream/stages/generation/generic_mapping_generation.py`):

- **Selects eligible cores per node** from the hardware. A core with an `operator_types` list only accepts those op types (so pooling goes to the pooling core, SiLU/Mul to the SIMD core); a core without the list accepts anything. A core listing an operator also runs the operators it is expanded into and its parts (a core listing `Softmax` runs `PartialSoftmax`). A node no core runs is refused, naming its operator. Memory cores are never chosen for computation.
- **Splits work across cores** when several compute cores are eligible, by factoring the node's dimensions into an inter-core tiling.
- **Follows the layouts** of a mapping file with only `runtime_args`: it splits the cores first along the dimension indexing the axis the layouts split the most bits along, so each core reads the block of the off-chip memory beside it; an operator reading what such a split made, in its group or across a fusion cut, follows that split in turn, so the data stays where it was made. It keeps the tiler from walking that dimension over time, and carries the layouts into each group's mapping.
- **Writes per-fusion-group `mapping.yaml` files** under the run's output directory, so you can inspect (and later hand-edit) exactly what it chose.

This is enough to run any of the example workloads on any of the example architectures - see the matrix in the README.

---

## Hand-written mapping

A mapping file is a list of layer entries (optionally wrapped in `layers:`), with optional `fused_groups:`.

```yaml
layers:
  - name: Conv
    core_allocation:
      - [0, 1, 2, 3]          # candidate cores for Conv nodes
    inter_core_tiling:
      - - dim: D6
          split: 4            # split dimension D6 across 4 cores

  - name: Gemm
    core_allocation:
      - [0, 1, 2, 3]
    inter_core_tiling:
      - - dim: D2
          split: 4

  - name: MaxPool
    core_allocation:
      - [4]                   # the pooling core

  - name: Add
    core_allocation:
      - [5]                   # the SIMD core

fused_groups:
  - name: Fused_Group_1
    layers: [Conv]
    intra_core_tiling:
      - dim: Conv.D0
        tile: 1
```

### Matching entries to nodes

For each node in the workload, Stream looks for a mapping entry in this order:

1. **Exact name** - the entry `name` equals the node's name (e.g. `Gemm_Left`).
2. **Operator type** - the entry `name` equals the node's op type (e.g. `Gemm`, `Conv`, `Add`). This is the common case, letting one entry cover all nodes of a type.

If neither matches, validation fails - every node must resolve to an entry (use a type-level entry to catch the rest).

### Layer fields

| Field | Required | Meaning |
|-------|:---:|---------|
| `name` | yes | Node name or operator type to match (see above). |
| `core_allocation` | yes | A list of **candidate core-id groups**. `[[0,1,2,3]]` is one group of four cores; the MILP allocator chooses the actual placement within that candidate set. A single-core role is just `[[4]]`. |
| `inter_core_tiling` | no | How to split the operator **across** cores. Each inner entry is `{dim: D<n>, split: k}` - split loop dimension `D<n>` (0-indexed in the node's loop nest) by factor `k`. |
| `kernel` | no | Kernel used by the AIE codegen path: `{name: <kernel>, kwargs: {<field>: <value>}}`, described by the accelerator's [kernel library](kernel_library.md). Ignored by the non-AIE CO pipeline. |

### Fused-group fields

`fused_groups` declares which layers are scheduled together as one fusion group and how they tile **within** a core.

| Field | Required | Meaning |
|-------|:---:|---------|
| `name` | yes | Group label. |
| `layers` | yes | Names of the layers in this group. |
| `intra_core_tiling` | no | Per-dimension temporal tiling, each `{dim: <Node>.D<n>, tile: size}` - note the **fully-qualified** dimension name (e.g. `Conv.D0`). |

---

## Where inputs and outputs live

An accelerator with several off-chip memories (see [Hardware](hardware.md#memories)) needs to know where each workload input and output lives. The mapping file says so per edge under `runtime_args`, keyed by the tensor's name:

```yaml
runtime_args:
  K: {memories: [hbm_0, hbm_1, hbm_2, hbm_3], split: 1}   # block s along axis 1 in hbm_s
  q: {memories: [hbm_0, hbm_1, hbm_2, hbm_3]}             # a copy in each
  O: {memories: [hbm_0]}                                  # all of it in one
```

| Layout | Meaning |
|--------|---------|
| `split: <axis>` | The whole tensor cut into one contiguous block per memory along that axis (0-indexed, as the workload declares the tensor), block `s` in `memories[s]`. The axis must split evenly. |
| several `memories`, no `split` | A copy of the whole tensor in each; every reader takes it from the nearest. Inputs only. |
| one memory | All of it there, as with a single off-chip memory. |

These entries live beside the code generators' `layout` and are read apart from it.

- **Without a layout.** With several off-chip memories, an input without one is refused, naming it. An output without one lands a block per producing core in the off-chip memory the fewest hops from where that core holds it. The input end of a fusion cut without one reads it back from there.
- **Fusion cuts.** The two ends of a cut, `<node>_cut_out` and `<node>_cut_in`, are one tensor: a layout declared for either holds for both, and ends declaring different memories are refused.
- **Time and traffic.** Each memory serves what its blocks hold of what each core reads. Where the iterations walk a split axis, one iteration may read from fewer memories than the walk does: the latency takes the busiest iteration, which waits on the memories it reads while the others idle, while traffic and energy take the average over the walk, each memory its share. The busiest iteration is an upper bound: a schedule prefetching more than one tile ahead may beat it.
- **Capacity.** Before solving, each off-chip memory must hold what lives in it: a block of each tensor split over it, all of each copy, each tensor counted once over the fused groups and both ends of a cut. Otherwise the run is refused, naming the memory and the sum, e.g. `hbm_0 holds K + V + q/4 + O/4 = 32 MiB > 16 MiB`.

A layout the cores' split follows keeps each core's data in its own slice. Split along an axis the cores do not split, the data crosses the on-chip links; split along an axis the iterations walk, each iteration waits on one memory. `tests/test_decode_kv.py` shows all three for decode attention (`stream/inputs/testing/workload/make_decode_attention.py`) on `stream/inputs/testing/hardware/tpu_like_ring_hbm.yaml` and `tpu_v7_ironwood_hbm_per_chiplet.yaml`.

---

## How the mapping feeds the optimizer

`core_allocation` defines the **candidate set**, not a fixed assignment (unless a role has only one core). The MILP allocator (`AllocationModel`) then chooses, within those candidates, where each tensor lives and which links carry each transfer - minimizing latency subject to memory and bandwidth constraints. `inter_core_tiling` determines how many parallel pieces exist to place; `fused_groups` / `intra_core_tiling` determine what is co-scheduled and how it is temporally tiled on a core.

For a workload with multiple fusion groups, the pipeline runs the CO once per group (see [Stages](stages.md)).

---

## Reusing an allocation

The auto-generated `mapping.yaml` files written into a run's output directory are valid hand-written mappings. To pin a result, copy the generated mapping out, edit the `core_allocation` candidate sets down to the chosen cores, and pass it back as `evaluate_mapping`'s `mapping`.
