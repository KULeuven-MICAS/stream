# Hardware

A hardware target in Stream is a **system of heterogeneous dataflow cores**. You describe it with YAML in two layers:

1. An **accelerator file** - names the system, lists its cores, and defines how the cores are connected.
2. One **core file per core type** - describes a single core's compute array, memory hierarchy, and role.

Accelerator files live in `stream/inputs/examples/hardware/`; the core files they reference live in `stream/inputs/examples/hardware/cores/`. Several cores in an accelerator may point at the *same* core file (e.g. four identical compute cores).

The format is validated by `stream/parser/accelerator_validator.py` and built by `stream/parser/accelerator_factory.py` - those files are the authoritative schema if anything below is ambiguous.

---

## Accelerator file

```yaml
name: tpu_like_quad_core

cores:
  0: ./cores/tpu_like.yaml   # compute A
  1: ./cores/tpu_like.yaml   # compute B
  2: ./cores/tpu_like.yaml   # compute C
  3: ./cores/tpu_like.yaml   # compute D
  4: ./cores/pooling.yaml    # pooling engine
  5: ./cores/simd.yaml       # SIMD / vector unit
  6: ./cores/offchip.yaml    # DRAM controller

core_coordinates:
  0: [0, 0]
  1: [1, 0]
  2: [1, 1]
  3: [0, 1]
  4: [2, 0]
  5: [2, 1]
  6: [1, 2]

offchip_core_id: 6      # core that fronts external memory
unit_energy_cost: 0     # default per-transfer energy for the links below

core_connectivity:
  # 2-D mesh among the four compute cores
  - type: link
    cores: [0, 1]
    bandwidth: 32
  - type: link
    cores: [1, 2]
    bandwidth: 32
  # … more links …

  # shared bus to off-chip memory (all cores ↔ DRAM)
  - type: bus
    cores: [0, 1, 2, 3, 4, 5, 6]
    bandwidth: 128
```

| Field | Required | Meaning |
|-------|:---:|---------|
| `name` | yes | A label for the system. |
| `cores` | yes | Map of integer **core id → core file**. Paths may be relative (`./cores/foo.yaml`); a bare filename is resolved against `<accelerator_dir>/cores/` then `<accelerator_dir>/`. |
| `offchip_core_id` | yes | The id of the core that fronts external memory (DRAM). Must appear in `cores`. No computation is ever placed here. |
| `core_connectivity` | yes | List of links and buses connecting the cores (see below). |
| `unit_energy_cost` | no | Default energy per transferred word for every connection, unless overridden per-connection. Defaults to `0`. |
| `core_coordinates` | no | Map of core id → `[col, row]`. Used for placement-aware models; required for the AIE namespace, optional otherwise. |
| `core_memory_sharing` | no | Groups of core ids that share L1 memory, e.g. `["0, 1", "2, 3"]`. |

### Connections: `link` vs `bus`

Each entry in `core_connectivity` is one connection:

| Field | Required | Meaning |
|-------|:---:|---------|
| `type` | no | `link` (point-to-point between exactly two cores) or `bus` (shared medium between two or more cores). Defaults to `link`. |
| `cores` | yes | The connected core ids. Exactly two for a `link`; two or more for a `bus`. |
| `bandwidth` | yes | Peak bandwidth of the connection (bits per cycle), `> 0`. |
| `unit_energy_cost` | no | Per-word energy for this connection; inherits the accelerator-level default if omitted. |

Connections are how the MILP allocator routes tensors between cores. A typical accelerator has fast point-to-point `link`s on-chip and one wider `bus` to the off-chip core.

---

## Core file

Every core declares a `type` of the form `<namespace>.<kind>`:

- **namespace** - `zigzag` (a full dataflow core modelled with ZigZag's cost model) or `aie2` (an AMD AIE tile).
- **kind** - `compute`, `memory`, `shim`, or `offchip`.

A bare kind (e.g. `type: compute`) is accepted and defaults to the `zigzag` namespace.

### ZigZag cores (compute / offchip)

A ZigZag core has a memory hierarchy and an operational (MAC) array:

```yaml
name: tpu_like
type: zigzag.compute

memories:
  rf_2B:                       # a register file serving outputs
    size: 16                   # capacity in bits
    r_cost: 0.021              # read energy (pJ)
    w_cost: 0.021              # write energy (pJ)
    area: 0
    latency: 1
    operands: [O]              # which operands this memory holds
    ports:
      - name: r_port_1
        type: read             # read | write | read_write
        bandwidth_min: 16      # bits / cycle
        bandwidth_max: 16
        allocation:
          - O, tl              # (operand, port-direction tag)
      - name: w_port_1
        type: write
        bandwidth_min: 16
        bandwidth_max: 16
        allocation:
          - O, fh
    served_dimensions: [D2]    # array dimensions this memory feeds

  sram_2MB:
    size: 16777216
    r_cost: 416.16
    w_cost: 378.4
    latency: 1
    operands: [I1, I2, O]      # weights, activations, outputs
    ports:
      - name: r_port_1
        type: read
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation: [I1, tl, I2, tl, O, tl, O, th]
      - name: w_port_1
        type: write
        bandwidth_min: 64
        bandwidth_max: 2048
        allocation: [I1, fh, I2, fh, O, fh, O, fl]
    served_dimensions: [D1, D2]

operational_array:
  unit_energy: 0.04            # pJ per MAC
  unit_area: 1
  dimensions: [D1, D2]         # spatial (unrolled) dimensions of the array
  sizes: [32, 32]              # 32 × 32 systolic array
```

**Operands.** Stream uses three algorithmic operands: `I1` (weights / first input), `I2` (activations / second input), and `O` (output). A memory's `operands` field lists which of these it stores.

**Ports and `allocation`.** Each memory exposes `ports`. An `allocation` entry pairs an operand with a port-direction tag - `fh` (write the *high* / final value), `fl` (write the *low* / partial value), `tl` (read *low* / from lower level), `th` (read *high* / to higher level). This is how Stream knows which port carries which data movement when it builds the cost model.

**`served_dimensions`.** The operational-array dimensions (`D1`, `D2`, …) that this memory feeds. An empty list means the memory is innermost (feeds a single MAC lane).

**`operational_array`.** The spatial compute fabric: `dimensions` names the unrolled axes, `sizes` gives their extent, and `unit_energy` / `unit_area` are the per-MAC cost.

### The off-chip core

The off-chip / DRAM core is just a ZigZag core with `type: zigzag.offchip`, a single large DRAM memory holding all operands, and a zero-size operational array (it never computes):

```yaml
name: offchip
type: zigzag.offchip
memories:
  dram:
    size: 10000000000
    r_cost: 1000
    w_cost: 1000
    latency: 1
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: 64
        bandwidth_max: 64
        allocation: [I1, fh, I1, tl, I2, fh, I2, tl, O, fh, O, tl, O, fl, O, th]
    served_dimensions: [D1, D2]
operational_array:
  unit_energy: 0
  unit_area: 0
  dimensions: [D1, D2]
  sizes: [0, 0]
```

### Restricting a core to certain operators

A compute core can be limited to specific operator types with an `operator_types` list. For example a pooling engine accepts only pooling ops:

```yaml
name: pooling
type: zigzag.compute
# … memories, operational_array …
operator_types: [MaxPool, AveragePool, GlobalAveragePool]
```

If `operator_types` is omitted, the core accepts **any** operator. The auto-mapper (see [Mapping](mapping.md)) uses this field to decide which nodes a core is eligible for - e.g. SiLU/Mul go to the SIMD core, pooling to the pooling core, and Conv/Gemm to the general compute cores.

### AIE cores

AIE tiles use the `aie2` namespace and a lighter schema describing tile-local memory and the object-FIFO depth, rather than a full ZigZag hierarchy. These are used by the AIE codegen entry points and live under `stream/inputs/aie/hardware/`.

```yaml
name: aie_tile
type: aie2.compute
max_object_fifo_depth: 12
memory:
  capacity: 524288     # bits
  bandwidth_min: 512   # bits / cycle
  bandwidth_max: 512
```

---

## Example systems

Eight non-AIE example accelerators ship under `stream/inputs/examples/hardware/` and are exercised across both example workloads in the test matrix (see the [Workload × Hardware matrix](https://github.com/KULeuven-MICAS/stream_aie#workload--hardware-matrix) in the README):

`eyeriss_like_single_core`, `eyeriss_like_dual_core`, `eyeriss_like_quad_core`, `tpu_like_quad_core`, `simba_small`, `simba` (a 36-core chiplet mesh), `fusemax`, and `meta_prototype_dual_core_simd_offchip`.

AIE example systems (`single_core`, single column, full array, Strix variant) live under `stream/inputs/aie/hardware/`.
