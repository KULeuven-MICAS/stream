# Mapping

The mapping ties a [workload](workload.md) to a [hardware](hardware.md) system: it says **which cores** each operator may run on and **how its work is split** across them. The constraint-optimization (CO) pipeline then uses this as the search space and decides the concrete tensor placement and routing.

There are two things to keep distinct:

- The **spatial mapping (dataflow)** — how a single operator is unrolled across a core's MAC array — lives on the **core**, in the hardware file (`operational_array.dimensions` / `sizes`). It is a property of the core, not of the mapping file.
- The **mapping file** decides **core allocation**, **inter-core tiling** (splitting an operator across multiple cores), and **intra-core tiling / fusion** (how operators group and tile within a core).

A mapping can be **auto-generated** by the pipeline or **hand-written**. The format is validated by `stream/parser/mapping_validator.py`.

---

## Auto-generated mapping (the default)

If you don't pass a mapping, `optimize_allocation_co_generic` (and `scripts/main_stream_co.py` without `--mapping`) builds one for you. This is the recommended starting point.

The generic generator (`stream/stages/generation/generic_mapping_generation.py`):

- **Selects eligible cores per node** from the hardware. A core with an `operator_types` list only accepts those op types (so pooling goes to the pooling core, SiLU/Mul to the SIMD core); a core without the list accepts anything. Off-chip and shim cores are never chosen for computation.
- **Splits work across cores** when several compute cores are eligible, by factoring the node's dimensions into an inter-core tiling.
- **Writes per-fusion-group `mapping.yaml` files** under the run's output directory, so you can inspect (and later hand-edit) exactly what it chose.

This is enough to run any of the example workloads on any of the example architectures — see the matrix in the README.

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

1. **Exact name** — the entry `name` equals the node's name (e.g. `Gemm_Left`).
2. **Operator type** — the entry `name` equals the node's op type (e.g. `Gemm`, `Conv`, `Add`). This is the common case, letting one entry cover all nodes of a type.

If neither matches, validation fails — every node must resolve to an entry (use a type-level entry to catch the rest).

### Layer fields

| Field | Required | Meaning |
|-------|:---:|---------|
| `name` | yes | Node name or operator type to match (see above). |
| `core_allocation` | yes | A list of **candidate core-id groups**. `[[0,1,2,3]]` is one group of four cores; the MILP allocator chooses the actual placement within that candidate set. A single-core role is just `[[4]]`. |
| `inter_core_tiling` | no | How to split the operator **across** cores. Each inner entry is `{dim: D<n>, split: k}` — split loop dimension `D<n>` (0-indexed in the node's loop nest) by factor `k`. |
| `kernel` | no | Kernel hint used by the AIE codegen path: `{name: <kernel>, kwargs: {utilization: <pct>}}`. Ignored by the non-AIE CO pipeline. |

### Fused-group fields

`fused_groups` declares which layers are scheduled together as one fusion group and how they tile **within** a core.

| Field | Required | Meaning |
|-------|:---:|---------|
| `name` | yes | Group label. |
| `layers` | yes | Names of the layers in this group. |
| `intra_core_tiling` | no | Per-dimension temporal tiling, each `{dim: <Node>.D<n>, tile: size}` — note the **fully-qualified** dimension name (e.g. `Conv.D0`). |

---

## How the mapping feeds the optimizer

`core_allocation` defines the **candidate set**, not a fixed assignment (unless a role has only one core). The MILP allocator (`TransferAndTensorAllocator`) then chooses, within those candidates, where each tensor lives and which links carry each transfer — minimizing latency subject to memory and bandwidth constraints. `inter_core_tiling` determines how many parallel pieces exist to place; `fused_groups` / `intra_core_tiling` determine what is co-scheduled and how it is temporally tiled on a core.

For a workload with multiple fusion groups, the pipeline runs the CO once per group (see [Stages](stages.md)).

---

## Reusing an allocation

The auto-generated `mapping.yaml` files written into a run's output directory are valid hand-written mappings. To pin a result, copy the generated mapping out, edit the `core_allocation` candidate sets down to the chosen cores, and pass it back with `--mapping` (or to `optimize_allocation_co_with_mapping`).
