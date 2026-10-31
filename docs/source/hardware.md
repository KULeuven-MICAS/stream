# Hardware

A hardware target in Stream is a **system of heterogeneous dataflow cores** that compute out of **memories**, with the memories joined by links and buses. You describe it with YAML in two layers:

1. An **accelerator file** - names the system, lists its cores and memories, says which memory each core computes out of, and connects the memories.
2. One **core file per core type** - describes a single core's compute array and memory hierarchy.

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
  5: ./cores/simd.yaml       # SIMD unit

core_coordinates:
  0: [0, 0]
  1: [1, 0]
  2: [1, 1]
  3: [0, 1]
  4: [2, 0]
  5: [2, 1]

unit_energy_cost: 0     # default energy for the connections

memories:
  tpu_like_0:
    offchip: false
    views: [0.sram_2MB]           # <core id>.<top-level memory instance in that core's file>
    ports:                        # ports core 0's file declares on sram_2MB
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
  # 2-D mesh among the four compute cores' memories
  - type: link
    ends:                         # each memory it joins, with the ports the link passes there
      tpu_like_0: [r_port_1, w_port_1]
      tpu_like_1: [r_port_1, w_port_1]
    bandwidth: 32
  # … more links …

  # shared bus to off-chip memory
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
| `cores` | yes | Map of integer **core id → core file**. Paths may be relative (`./cores/foo.yaml`); a bare filename is resolved against `<accelerator_dir>/cores/` then `<accelerator_dir>/`. |
| `memories` | yes | Map of memory name → its description, see [Memories](#memories). |
| `connections` | yes | List of links and buses between memories, see [Connections](#connections-link-vs-bus). |
| `core_coordinates` | no | Map of core id → `[col, row]`. Used for placement-aware models; required for the AIE namespace, optional otherwise. |
| `bandwidth` | no | Measured bandwidth models, keyed by memory name or by `<memory>.<port>`, see [Measured bandwidth](#measured-bandwidth). |
| `unit_energy_cost` | no | Default energy per transferred word for every connection, unless overridden per-connection. Defaults to `0`. |
| `kernel_library` | no | Path, relative to this file, to the [kernel library](kernel_library.md) the target's AIE kernels are described by. |
| `reconfiguration` | no | What reconfiguring the array between the designs of one dispatch costs, read by the cores' namespace; for `aie2`, `cycles_per_column` per design and `reset_cycles` once. |

### Memories

Data lives in memories. A core computes out of its top-level memory instances (per operand, the highest level in its core file), and each such instance is a **view** of exactly one memory. Several cores may view one memory, as the MXU, VPU and VMEM cores of `tpu_v7_ironwood.yaml` do; data between them moves in place, without a route. Lower levels of a core's hierarchy are internal to the core.

| Field | Required | Meaning |
|-------|:---:|---------|
| `offchip` | yes | Whether it is an off-chip memory, where the workload's inputs and outputs live. Every other memory has a route from one, and no core views it. |
| `views` | yes | The core instances that see this memory, as `<core id>.<instance>`, e.g. `[0.l1, 1.tcdm]`. Empty for a memory no core computes out of, such as DRAM. |
| `capacity_bits` | only without views | Its capacity in bits. A memory with views takes the size its views declare. |
| `ports` | yes | Its ports by name, see [Memory ports](#memory-ports). |
| `dma` | no | The DMA behind its `channels` ports: `buffer_descriptors` and `iterations` per descriptor. Given exactly when a port has `channels`. Only the `aie2` namespace bounds DMA channels: there the off-chip memory has `dma` and every one of its ports has `channels`. The AIE code generator chains re-reads within these limits. |

An accelerator may have several off-chip memories, each with its own capacity, ports and bandwidth (and its own [measured bandwidth](#measured-bandwidth), keyed by its name or port), such as the HBM stacks of each chiplet. `Accelerator.offchip_memories` lists them in the order they are declared. Every on-chip memory must be reachable from at least one; data one core reads from another's off-chip memory crosses the on-chip links between them. The mapping says where each workload input and output lives over them (see [Mapping](mapping.md#where-inputs-and-outputs-live)). With several, the latency times the links of each through the `overlap` family's off-chip contention, so a selection without it is refused, and the AIE code generator, whose runtime sequence reaches one, refuses the hardware. `stream/inputs/testing/hardware/` has two such systems: `tpu_like_ring_hbm` (four slices on a ring, an HBM stack each) and `tpu_v7_ironwood_hbm_per_chiplet` (`tpu_v7_ironwood` with an HBM per chiplet).

The allocation places tensors in memories; a memory's capacity bounds what all its cores hold in it together. A node reads input `k` from the memory its core's `I<k>` instance views, and writes to the one its `O` instance views.

### Memory ports

`ports` lists every physical port of a memory. A port either comes from the viewing cores' files or is declared by the memory itself:

```yaml
ports:
  # Port `rw_port_1` that the listed cores declare on their view; cores in one entry share the port.
  rw_a: {from: rw_port_1, cores: [0, 1]}
  rw_b: {from: rw_port_1, cores: [2]}
  # Declared here: width in bits per cycle, directions served per operand (`input<k>`, `output` or `any`),
  # energies in pJ per bit.
  rw_port_1: {bandwidth: 64, serves: {any: [read, write]}, read_energy: 15.625, write_energy: 15.625}
  # A DMA port: each of its `channels` carries one transfer at a time.
  mm2s: {bandwidth: 128, channels: 2, serves: {any: [read]}, read_energy: 0, write_energy: 0}
```

Every port a viewing core declares must belong to exactly one entry. A transfer passes, of the ports its connection lists for the memory, the one serving its operand in its direction; list at each end a port serving reads and one serving writes (unless the memory has a measured bandwidth), and only one such port per operand and direction.

The `memory_ports` [constraint family](stages.md#constraint-families), in the default set, uses these rates. Its `interval` option (default true) fits each port's bits per iteration in its rate times the initiation interval; `burst` (default true) fits each step's bits in its rate times that step's latency.

### Measured bandwidth

`bandwidth` gives a measured model, in bits per cycle: `ceiling` for reads and writes together, `contiguous` one way, and `strided.read` / `strided.write` per contiguous span in bytes, or, where nothing is measured, `burst`, the bytes moved per access, of which a shorter contiguous run uses only its share (see [Data layout](data_layout.md)). Its key is either:

- a **memory** (e.g. `dram`): the model bounds every transfer into or out of it, in place of its ports' widths, so its views need not list their ports;
- a **port** `<memory>.<port>` (e.g. `hbm.rw_port_1`): the model replaces that port's width.

### Connections: `link` vs `bus`

Each entry in `connections` is one connection between memories:

| Field | Required | Meaning |
|-------|:---:|---------|
| `type` | no | `link` (point-to-point between exactly two memories) or `bus` (shared medium between two or more memories). Defaults to `link`. |
| `ends` | yes | Map of each connected memory → the ports of that memory the connection passes. |
| `bandwidth` | yes | Peak bandwidth of the connection (bits per cycle), `> 0`. |
| `unit_energy_cost` | no | Per-word energy for this connection; inherits the accelerator-level default if omitted. |

Connections are how the MILP allocator routes tensors between memories. A typical accelerator has fast point-to-point `link`s on-chip and one wider `bus` to the off-chip memory.

A transfer's route serves each target from the nearest memory holding the data it reads, then joins each source's targets in one tree (a Steiner tree, grown from the shortest paths with each link weighted by the time the transfer already keeps it busy, and a link that already carries all the next target reads by nothing), so data shared by several targets crosses each link once. A transfer from the off-chip memory to several targets also has a striped route: the data every target the off-chip memory links to reads is cut into a part per such target, each part enters over that target's link and takes the shortest paths on chip to the others; the allocator picks between the routes. A memory passing data on reads from its ports what it holds of that data itself, unless the namespace of its cores switches the data onto the next link unread. Partial sums, of a computation split over a dimension it reduces, are added up on their way to their target; one headed off-chip is completed in the on-chip memories next to it, a slice each, as a reduce-scatter. A route never passes through the off-chip memory, and a bus carries what each memory sends over it separately. Each link is charged, in time and in energy, the share of the transfer's data crossing it.

---

## Core file

Every core declares a `type` of the form `<namespace>.<kind>`:

- **namespace** - `zigzag` (a full dataflow core modelled with ZigZag's cost model) or `aie2` (an AMD AIE tile).
- **kind** - `compute` or `memory` (a core that holds data but runs no nodes, such as an AIE memory tile). Off-chip memory is not a core: the accelerator's `memories` declares it.

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

AIE tiles use the `aie2` namespace and a lighter schema describing tile-local memory, the tile DMA's channels and the object-FIFO depth, rather than a full ZigZag hierarchy. These are used by the AIE codegen entry points and live under `stream/inputs/aie/hardware/`.

```yaml
name: aie_tile
type: aie2.compute
max_object_fifo_depth: 12
memory:
  capacity: 524288     # bits
  bandwidth_min: 512   # bits / cycle
  bandwidth_max: 512
dma:                   # tile DMA channels
  mm2s: 2
  s2mm: 2
  channel_bits: 64
```

A tile's view is `<core id>.memory`, with ports `mm2s` and `s2mm`. On `aie2` each on-chip memory is a view of exactly one tile (the shipped files name it `tile_<col>_<row>`); a neighbouring tile reads it without a view.

DRAM has no views. Its ports are the shim DMAs, one `mm2s`/`s2mm` pair per column (`mm2s_c<col>`, `s2mm_c<col>`), each with `channels: 2`, and each column's link to its memory tile passes that column's pair; a shim port serves one column. DRAM's `dma` gives the shim DMA's buffer descriptors and iterations (16 and 64 on AIE2).

---

## Converting a hardware file from the core format

Stream used to describe an accelerator as cores joined to cores: `offchip_core_id` named a core standing in for
DRAM, `core_connectivity` joined cores, and `core_memory_sharing` and `memory_aliases` said which cores' memories were
one. Such a file no longer parses. The core files stay as they are, except for `banks`; the accelerator file gains
`memories` and `connections`. [The conversion prompt](convert_hardware_prompt.md) has an agent convert one file, and
`scripts/check_hardware_conversion.py` checks the result against the Stream that read the old file.

### Key by key

| Core format | Memories format |
|---|---|
| `offchip_core_id: k` | Core `k` leaves `cores`, `core_coordinates` and every connection; a memory with `offchip: true`, `views: []` takes its place. |
| ZigZag off-chip core: its top level's `size` | That memory's `capacity_bits`. |
| its port `p`: `bandwidth_max`, `allocation`, `r_cost`, `w_cost` | `p: {bandwidth: <bandwidth_max>, serves: ..., read_energy: <r_cost / bandwidth_max>, write_energy: <w_cost / bandwidth_max>}`: an access moved `bandwidth_max` bits. `serves` lists per operand (`I1` is `input1`, `I2` `input2`, `O` `output`) `read` for `th`/`tl` and `write` for `fh`/`fl`; `any` for all three alike. |
| `aie2.shim` off-chip core | `capacity_bits: <memory.capacity>`, `dma: {buffer_descriptors, iterations}`, ports `mm2s` and `s2mm` of `channels: <dma.mm2s>`, `bandwidth: <mm2s x channel_bits>`, `serves: {any: [read]}` (`[write]` for `s2mm`), energies 0. |
| A core's top-level instance `i` (per operand its highest level; `memory` on an AIE tile) | A memory with `views: [<core>.i]` and, per port `p` of `i`, `p: {from: p, cores: [<core>]}`; on an AIE tile `mm2s` and `s2mm`. |
| `core_memory_sharing: ["a, b, c"]` | One memory, `views: [a.i, b.i, c.i]`, each port `{from: p, cores: [a, b, c]}`. |
| `memory_aliases: [[a.x, b.y]]` | One memory, `views: [a.x, b.y]`. |
| `core_connectivity` entry `cores: [a, b, ...]` | `connections` entry, `ends:` the memories of those cores, each with the ports a transfer passes there: one per operand and direction, usually all of the memory's. `type`, `bandwidth` and `unit_energy_cost` as they were. |
| `bandwidth` key `k` (a core) | The name of `k`'s memory; its views need not list their ports. |
| `bandwidth` key `k.i.p` | `<memory>.<port name>`. |
| A core file's top-level `banks` | Under the level it describes: `memories.<level>.banks`, or `memory.banks` on an AIE tile. |
| Core kinds `offchip`, `shim` | Gone: a core is `compute` or `memory`. |

`name`, `cores` (minus the off-chip core), `core_coordinates`, `kernel_library`, `reconfiguration`,
`technology_node` and `unit_energy_cost` are unchanged. Core ids keep their values; they need not be contiguous.

### What the old Stream did with a file

The old Stream did not always build what a file says. A faithful conversion builds what it built; doing what the file
says instead is a modelling change, which the checker reports.

| Case | The old Stream | Faithful conversion |
|---|---|---|
| Two connections join the same two cores, such as links declared before a bus over all cores | Kept one edge per pair of cores, the last declared: the links never carried data (every `eyeriss_like_*`, `tpu_like_quad_core`, `simba_small` and `meta_prototype` file) | Leave such links out |
| A link at a core of a `core_memory_sharing` group other than its first | Entered and left the group through its first core where that one had links, so the link went unused (`fusemax`'s second link to DRAM) | Leave it out; two links between the same memories are two resources |
| A core in no connection and no sharing group | Never built it (`whole_array`'s shim cores 6, 12, 18) | Drop it |
| A core with several top-level instances | One memory, sized by the instance holding `I1`, its links reaching all of them (`meta_prototype`: activations and weights) | One memory per instance; a link joins the `I1` one, a bus all; capacity and occupancy now split, which no faithful form avoids |
| `memory_aliases` | Costing only: each aliased memory kept its capacity and the links between their cores, and all used the first one's ports | One memory with its capacity once and no link inside; cores declaring a port differently need a port entry each. No faithful form |
| An `aie2.shim` linked to every column | One shim's 2 + 2 DMA channels for the whole array | One `mm2s` / `s2mm` pair on every column's link; a pair per column (`mm2s_c<col>`) gives each column its own channels |
| A link between two cores that now view one memory | A route between them | Leave it out: the data moves in place |

What the memories format has no place for: the off-chip core's `bandwidth_min`, `latency`, `area`,
`served_dimensions`, `operational_array`, name, id and coordinates; a shim's `max_object_fifo_depth` and
`memory.bandwidth_min`/`bandwidth_max`. Delete the off-chip core file if no other accelerator uses it.

### Checking a conversion

Install the Stream that read the old file next to the current one, for example
`python -m venv /opt/stream-old && /opt/stream-old/bin/pip install "stream-dse @ git+https://github.com/KULeuven-MICAS/stream@d03bfd10"`
(or, with the same dependencies, `--old-python "env PYTHONPATH=<old checkout> python"`), then

```bash
python scripts/check_hardware_conversion.py check OLD.yaml NEW.yaml --old-python /opt/stream-old/bin/python
```

Each file is summarised by its own Stream and the summaries compared. A memory is known by its views
(`0.vmem+1.vmem`), the off-chip one as `offchip`, and ports and connections by their cores, operands, rates and
energies, so names may change. Compared are: per core its type, coordinates and operators; per memory its capacity,
ports, measured bandwidth, off-chip DMA and banks; every connection with the ports at each end; and the latency,
energy and per-memory occupancy of one solve (the 2-conv test workload unless `--workload`, `--mapping` and `--cols`
say otherwise; `--no-solve` skips it), within `--rtol` (1 %), the occupancy up to a permutation of equal memories.
It prints each difference and a last line with the latency and energy of both, and exits with 1 if anything
differs. `summarize` writes one file's summary and `compare` compares two, for a summary recorded elsewhere.

The shipped files keep the links the old Stream ignored, give `meta_prototype`'s weights memories of their own and
each AIE column its own shim ports; against the old files the checker reports these and nothing else.

---

## Example systems

Nine non-AIE example accelerators ship under `stream/inputs/examples/hardware/` and are exercised across both example workloads in the test matrix (see the [Workload × Hardware matrix](https://github.com/KULeuven-MICAS/stream#workload--hardware-matrix) in the README):

`eyeriss_like_single_core`, `eyeriss_like_dual_core`, `eyeriss_like_quad_core`, `tpu_like_quad_core`, `simba_small`, `simba` (a 36-core chiplet mesh), `fusemax`, `meta_prototype_dual_core_simd_offchip`, and `tpu_v7_ironwood` (memories shared by several cores).

AIE example systems (`single_core`, single column, full array, Strix variant) live under `stream/inputs/aie/hardware/`.
