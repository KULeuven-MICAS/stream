# AccelForge core costs

[AccelForge](https://github.com/Accelergy-Project/accelforge) (MIT) is a second per-core cost model that Stream can use in place of ZigZag on ZigZag compute cores. It is optional and never chosen by default, and serves as an independent cross-check of a ZigZag result: on the same per-action energies the two agree when they choose the same mapping. It is not more accurate than ZigZag on anything Stream can measure, and its mapper is slower: a 2-conv workload on four TPU-like cores costs about a second more, the same workload on a single-core Eyeriss 544 s against ZigZag's few seconds.

The backend costs a node on a core as the ZigZag backend does, at the core's extent of every dim: the node becomes one Einsum, the core an arch with Stream's own energies, and the mapping with the fewest cycles, then the least energy, becomes the entry. AccelForge's fusion mapper (FFM) is not used: Stream splits a fused group over cores and prices, places and moves the tensors between nodes itself, so mapping the group with FFM would assume one core and price those tensors twice.

## Install

```bash
pip install "stream-dse[accelforge]"
pip install --no-deps "accelforge @ git+https://github.com/Accelergy-Project/accelforge.git@bfc12161157912bea8bdb0cab58d15cee1323431"
```

AccelForge 1.0.521 cannot be a requirement of the extra: its own requirements include `hwcomponents-cacti` (HP CACTI licence, no sale for compensation), `hwcomponents-neurosim` (CC BY-NC 4.0) and `hwcomponents-library` (both). The backend runs no component model, so the extra lists only what AccelForge imports, and AccelForge is installed without its requirements; `pip check` reports them missing.

## Use

```python
SolveOptions(stage_options={"core_cost_backend": "accelforge", "core_cost_options": {"accelforge": {"memory_order": "forced"}}})
```

`core_cost_backend` names the backend that costs every core it claims; the others keep the backend the priority rule chooses. A run naming a backend that claims none of its cores is refused, as when AccelForge is not installed. A node no Einsum can express (a stride that skips past its window, a product of dims, mod, floordiv or ceildiv) is costed by ZigZag, with a warning and the reason in `metadata["accelforge_refused"]`. Any other failure raises `AccelForgeEstimationError` naming the node and the core.

| Option | Default | Meaning |
|---|---|---|
| `memory_order` | `"free"` | `free` lets the storage of different tensors nest in any memory order, as ZigZag does; where AccelForge rejects one of its own free-order templates (tpu_like, tpu_v7_mxu, meta_prototype) the node is mapped in `forced` order and the entry records `memory_order_fallback`. |
| `imperfect_spatial_loops` | `False` | lets a fanout unroll part of a dim it does not divide, at many times the mapper time. |

An entry is cached with its options, the backend's revision and AccelForge's version, so changing them reprices the nodes. Its metadata holds the mapping, the mapper's wall time, the memory order, and the port traffic and spatial unrolling the allocation reads. Stream imports AccelForge only when it maps a node, and undoes the root logging handler, warning filters and environment variables its import adds.

## How the conversion differs from ZigZag

Energies are pJ per access in Stream and J per action in AccelForge, set from the core's `r_cost`, `w_cost` and MAC `unit_energy`; arches run at 1 Hz, so AccelForge's seconds are Stream's cycles. Where the two differ:

- **Memory order**: ZigZag orders each operand's memories on their own; in forced order AccelForge orders storage by the tree across tensors.
- **Access granularity**: ZigZag rounds each transfer to whole accesses and truncates totals to whole words; AccelForge rounds only where a transfer moves one value into one unit and elsewhere charges by the bit. Small transfers through wide ports differ most.
- **Latency**: ZigZag adds onloading, offloading and port stalls; AccelForge takes the slowest component. Systolic dimensions become broadcast fanouts, without fill or drain.
- **Ports**: several ports of one type become one read and one write throughput, the sum of their widths, at the first port's access width and energy.
- **Memories whose served dimensions do not nest** (tpu_v7_mxu, meta_prototype) are copied over the fanouts above them, with their size and bandwidth split.
- **Partial sums** take the accumulator precision on every memory below the output's top memory.
- **Tensors**: an index's padding is dropped, as ZigZag drops it, so a padded input carries its halo; an axis indexed by one strided dim is compacted to the values read; inputs beyond `I1` and `I2` live in the outermost memory; a tensor read twice through one map is read once.
- **Not modelled**: fixed dataflows (AccelForge searches spatial mappings itself), double buffering, per-access memory latency and area.
- **Ops**: an Einsum computes one action per point; the backend scales the compute cycles and the MAC energy by the operators a point expands into, as ZigZag's entries do.

In-memory-compute arrays, an operand whose memories are copied more often further from the array, and a memory without a read and a write port are refused, and the backend does not claim such a core.
