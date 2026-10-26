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

core_coordinates:
  0: [0, 0]
  1: [1, 0]
  2: [1, 1]
  3: [0, 1]
  4: [2, 0]
  5: [2, 1]

unit_energy_cost: 0     # default per-transfer energy for the links below

memories:
  tpu_like_0:
    offchip: false
    views: [0.sram_2MB]           # <core id>.<top-level memory instance>
    ports:                        # each a port core 0's file declares on sram_2MB
      r_port_1: {from: r_port_1, cores: [0]}
      w_port_1: {from: w_port_1, cores: [0]}
  # … tpu_like_1 to tpu_like_3, pooling and simd alike …
  offchip:
    offchip: true
    views: []                     # no core computes out of DRAM
    capacity_bits: 10000000000
    ports:
      rw_port_1: {bandwidth: 64, serves: {any: [read, write]}, read_energy: 15.625, write_energy: 15.625}

connections:
  # 2-D mesh among the four compute cores
  - type: link
    ends:                         # each memory it joins, with the ports it enters and leaves it through
      tpu_like_0: [r_port_1, w_port_1]
      tpu_like_1: [r_port_1, w_port_1]
    bandwidth: 32
  # … more links …

  # shared bus to off-chip memory (all cores ↔ DRAM)
  - type: bus
    ends:
      tpu_like_0: [r_port_1, w_port_1]
      # … tpu_like_1 to tpu_like_3, pooling and simd alike …
      offchip: [rw_port_1]
    bandwidth: 128
```

| Field | Required | Meaning |
|-------|:---:|---------|
| `name` | yes | A label for the system. |
| `cores` | yes | Map of integer **core id → core file**; ids need not be consecutive. Paths may be relative (`./cores/foo.yaml`); a bare filename is resolved against `<accelerator_dir>/cores/` then `<accelerator_dir>/`. |
| `kernel_library` | no | Path, relative to this file, to the [kernel library](kernel_library.md) the target's AIE kernels are described by. |
| `bandwidth` | no | Map of memory name to the measured bandwidth every transfer into or out of that memory shares, in bits per cycle: `ceiling` for reads and writes together, `contiguous` one way, and `strided.read` / `strided.write` per contiguous span in bytes, or, where nothing is measured, `burst`, the bytes the memory moves per access, of which a shorter contiguous run uses only its share (see [Data layout](data_layout.md)). The allocation then runs no transfer through the memory faster than its access pattern allows, and bounds every step by the time the memory spends on one iteration's transfers. A key can also name one memory port, see [Memory ports](#memory-ports). |
| `reconfiguration` | no | What reconfiguring the array between the designs of one dispatch costs, read by the cores' namespace; for `aie2`, `cycles_per_column` per design and `reset_cycles` once. |
| `memories` | yes | Map of memory name → whether it is off-chip, its views `<core id>.<instance>` and its ports, see [Memories](#memories). |
| `connections` | yes | List of links and buses connecting the memories (see below). |
| `unit_energy_cost` | no | Default energy per transferred word for every connection, unless overridden per-connection. Defaults to `0`. |
| `core_coordinates` | no | Map of core id → `[col, row]`. Used for placement-aware models; required for the AIE namespace, optional otherwise. |

### Memories

Data lives in memories, not in cores: a core computes out of its top-level memories, and several cores may compute out of the same one. `memories` names each memory and writes it out in full:

| Field | Required | Meaning |
|-------|:---:|---------|
| `offchip` | yes | Whether it is the off-chip memory, where the workload's inputs and outputs live. An accelerator has at most one. |
| `views` | yes | For each core using it, `<core id>.<instance>`, the name the core's file gives the top-level memory instance it sees the memory as, e.g. `[0.l1, 1.tcdm]`. Empty for a memory no core computes out of, such as DRAM. |
| `capacity_bits` | for a memory without views | Its capacity. A memory with views takes the one its views declare, and declares none. |
| `ports` | yes | Its ports by name, see [Memory ports](#memory-ports). |

Everything outside the core files refers to the memory by its name; a core's own names stay inside it.

Each top-level instance of a core, the highest level of some operand, is a view of exactly one memory; a core with an activation and a weights SRAM, such as `meta_prototype`, binds both, to two memories. A node reads each input, and writes its output, in the memory its core's top-level instance for that operand is a view of: its `k`-th operand in the one holding `I<k>`, the output in the one holding `O`. A multiply-accumulate's operands are its first two inputs; a further input, such as a bias, is added into the output, and state a node carries is updated in place as an accumulator is, so both land in the output's memory. A core with one top-level instance holds every operand there. Inner levels are the core's own: no transfer reaches them and Stream models none of their ports. The views of one memory declare the same capacity, and every memory has a connection: one nothing reaches holds nothing a transfer can bring.

A memory belongs to none of its cores. The allocation places tensors in memories, a placement naming the memory at each of its node's positions, the same memory repeated for cores that share it. A memory's capacity bounds everything its cores hold in it together: split tiles add up, a tensor each core holds whole counts once. Data between its cores moves in place, without a route. Connections join memories, and a `bandwidth` model of a memory bounds every transfer into or out of it, whichever of its cores the transfer serves; its macro is priced once.

### Memory ports

`ports` lists every physical port of a memory, each one of two kinds:

```yaml
ports:
  # A port the views' cores declare: the port `from` that cores `cores` declare on their view, one port they all reach.
  rw_a: {from: rw_port_1, cores: [0, 1]}   # cores 0 and 1 share one port
  rw_b: {from: rw_port_1, cores: [2]}      # core 2 has its own
  # A port of the memory's own, as DRAM's: its width in bits per cycle, the directions it serves per operand
  # (`input<k>`, `output`, or `any` for every operand), and its energies in pJ per bit.
  rw_port_1: {bandwidth: 64, serves: {any: [read, write]}, read_energy: 15.625, write_energy: 15.625}
```

Nothing is shared unless said so: every port a view's core declares on the memory is in exactly one of its ports, and the cores of one port must declare it alike, with the same width and energies. A port its views declare takes its width, energies and what it serves from the core files. A transfer leaves and enters a memory over the connections its route takes there, and passes one of the ports those connections name for that memory (see [Connections](#connections-link-vs-bus)): the one its operand is allocated in that direction (`fh` into the memory, `th` out of it; a tile DMA serves every operand), else the one serving that direction, as a ZigZag hierarchy allocates an input no port out of its top level. Two candidate ports, as a connection through the ports of two cores each with their own names, are an error, since which one the bits pass is then undeclared; a connection naming only the one it passes settles it. A `bandwidth` key `<memory>.<port>` (e.g. `dram.rw_port_1`) gives the measured model to one port. It replaces that port's width, in bits per cycle. A memory with a measured model has no per-port model, so it takes no port key.

The `memory_ports` [constraint family](stages.md#constraint-families), in the default set, uses these rates. Its `interval` option (default true) fits each port's bits per iteration in its rate times the initiation interval; `burst` (default true) fits each step's bits in its rate times that step's latency.

### Connections: `link` vs `bus`

Each entry in `connections` is one connection:

| Field | Required | Meaning |
|-------|:---:|---------|
| `type` | no | `link` (point-to-point between exactly two memories) or `bus` (shared medium between two or more memories). Defaults to `link`. |
| `ends` | yes | Map of each connected memory's name → the ports of it the connection enters and leaves it through, of those the memory declares. Exactly two memories for a `link`; two or more for a `bus`. A memory a transfer leaves or enters over the connection is charged at the port of these serving its operand; an empty list charges none. |
| `bandwidth` | yes | Peak bandwidth of the connection (bits per cycle), `> 0`. |
| `unit_energy_cost` | no | Per-word energy for this connection; inherits the accelerator-level default if omitted. |

Connections are how the MILP allocator routes tensors between memories. A typical accelerator has fast point-to-point `link`s on-chip and one wider `bus` to the off-chip memory.

A transfer's route serves each target from the nearest memory holding the data it reads, then joins each source's targets in one tree (a Steiner tree, grown from the shortest paths with each link weighted by the time the transfer already keeps it busy), so data shared by several targets crosses each link once. Partial sums, of a computation split over a dimension it reduces, are added up on their way to their target; one headed off-chip is completed in the on-chip memories next to it, a slice each, as a reduce-scatter. A route never passes through the off-chip memory, and a bus carries what each memory sends over it separately. Each link is charged the share of the transfer's data crossing it.

---

## Core file

Every core declares a `type` of the form `<namespace>.<kind>`:

- **namespace** - `zigzag` (a full dataflow core modelled with ZigZag's cost model) or `aie2` (an AMD AIE tile).
- **kind** - `compute` or `memory`. Off-chip memory is no core: `memories` declares it.

A bare kind (e.g. `type: compute`) is accepted and defaults to the `zigzag` namespace.

### ZigZag cores

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

### Restricting a core to certain operators

A compute core can be limited to specific operator types with an `operator_types` list. For example a pooling engine accepts only pooling ops:

```yaml
name: pooling
type: zigzag.compute
# … memories, operational_array …
operator_types: [MaxPool, AveragePool, GlobalAveragePool]
```

If `operator_types` is omitted, the core accepts **any** operator. The auto-mapper (see [Mapping](mapping.md)) uses this field to decide which nodes a core is eligible for - e.g. SiLU/Mul go to the SIMD core, pooling to the pooling core, and Conv/Gemm to the general compute cores.

### Operand precision

A compute core can declare the element types it computes in:

```yaml
operand_precision:
  input: bf16
  accumulator: fp32
```

The `accumulator` sets how wide a matmul's or convolution's partial sums are while the core accumulates them; the
result is written out at its output tensor's type. A core without an `accumulator` keeps partial sums at the output's
type. `input` names the multiplier format; it is recorded in the core's hardware cost report and does not change latency or energy. The known types are `int4`,
`int8`, `uint8`, `int16`, `int24`, `bf16`, `fp16`, `int32` and `fp32`.

### Memory banks

A memory level can declare how it is banked, for the layout report to place each tile's bytes in banks: in a ZigZag
core under `memories.<level>`, in an AIE tile under `memory`. The views of one memory must agree on its banks.

```yaml
memories:
  sram:
    # size, ports, ...
    banks:
      count: 8     # banks
      width: 16    # bytes per bank word
      stripe: 512  # words a bank holds before the next one does; 1 (the default) interleaves words
```

Byte address `a` sits in bank `a // width // stripe % count`. An AIE-ML tile's eight banks are each one contiguous
8 KB range (`stripe: 512`); a memory that interleaves words across its banks takes the default. The allocation does
not price bank conflicts: the banks only describe where data lands.

### Systolic arrays

An operational array whose operands move from one unit to the next, one unit per cycle, rather than being broadcast,
lists the dimensions they move along:

```yaml
operational_array:
  dimensions: [D1, D2]
  sizes: [256, 256]
  systolic_dimensions: [D1, D2]
```

ZigZag then adds the cycles the last results take to leave the array, one per unit in use along each such dimension
(510 on a full 256 x 256 array). A run's tiles follow each other through the array, so Stream charges this drain once
per run, in its fill, rather than on every tile. A node that reads another's results waits for them to drain, while
one that does not streams into its array behind it, so the fill adds the drains along the run's longest chain of
dependent nodes: a SwiGLU's gate and up projections drain together, then its down projection. The TPU-like, TPU7x MXU
and FuseMax arrays are systolic.

### AIE cores

AIE tiles use the `aie2` namespace and a lighter schema describing tile-local memory and the object-FIFO depth, rather than a full ZigZag hierarchy. These are used by the AIE codegen entry points and live under `stream/inputs/aie/hardware/`. A tile's one memory is its instance `memory`, so its view is `<core id>.memory`, and the memory is named after its tile, `tile_<col>_<row>`: the code generator names the memory a transfer lands in, and the AIE lowering reads the tile off that name. DRAM is the one exception: no core views it, and its two ports, `mm2s` and `s2mm`, are one shim tile's DMA, two 64-bit channels each way. The shim tiles are no cores: the generated code reaches DRAM from the runtime sequence, through the shim of the column the transfer's other end sits in, with the buffer descriptors and iterations an AIE2 shim DMA has.

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

Eight non-AIE example accelerators ship under `stream/inputs/examples/hardware/` and are exercised across both example workloads in the test matrix (see the [Workload × Hardware matrix](https://github.com/KULeuven-MICAS/stream#workload--hardware-matrix) in the README):

`eyeriss_like_single_core`, `eyeriss_like_dual_core`, `eyeriss_like_quad_core`, `tpu_like_quad_core`, `simba_small`, `simba` (a 36-core chiplet mesh), `fusemax`, and `meta_prototype_dual_core_simd_offchip`.

AIE example systems (`single_core`, single column, full array, Strix variant) live under `stream/inputs/aie/hardware/`.
