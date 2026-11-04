# Prompt: convert a hardware file from the core format

Give this page, an old accelerator file and its core files to an LLM agent working in a Stream checkout. The rules
behind each step are in [Hardware](hardware.md#converting-a-hardware-file-from-the-core-format).

---

You convert one Stream accelerator file from the core format (it has `offchip_core_id` and `core_connectivity`) into
the memories format (`memories` and `connections`), keeping every number the old file and its core files give.
The result must describe the accelerator the old Stream built from the old file, not the one its comments suggest.
Where the old file cannot be expressed exactly, or where the old Stream ignored something the file says, keep the
old behaviour and record the case under **Decisions**; change it only if the user asks.

## Inputs

- `OLD`: the accelerator file, and every core file its `cores` names (resolved as Stream resolves them: `./x.yaml`
  relative to the file; a bare `x.yaml` in `cores/` next to it, then next to it).
- The new format: `docs/source/hardware.md` and `stream/parser/accelerator_validator.py`.

## Steps

1. **Read the old accelerator as Stream built it.** For every core note its id, file, `type`, and its top-level
   memory instances: per operand (`I1`, `I2`, `O`) the last level in `memories` that lists it; an `aie2` core has
   one, `memory`. Note the instance holding `I1` (else the first one): the old Stream sized the core's memory by it.
   Build the old core graph: go through `core_connectivity` in order; a link joins its two cores, a bus every pair
   of its cores; for each pair of cores only the **last** connection that joins them survives. Drop every core no
   surviving connection reaches and that is in no `core_memory_sharing` group (the old Stream never built it).
2. **Keep the cores.** `cores` keeps every core except the off-chip core (`offchip_core_id`) and dropped cores, with
   the same ids and files; ids need not be contiguous. `core_coordinates` loses the same ids. `name`,
   `kernel_library`, `reconfiguration`, `technology_node`, `unit_energy_cost` and any key Stream does not define (one an
   extension declares) are copied unchanged.
3. **One memory per physical memory.** Group the top-level instances `<core>.<instance>`:
   - every `core_memory_sharing` group `"a, b, c"` joins `a.i`, `b.i`, `c.i` for each instance name `i` they share
     (else the instance holding `I1`);
   - every `memory_aliases` group `[a.x, b.y]` joins `a.x` and `b.y`;
   - every other instance is a memory of its own.

   Each group is one entry of `memories`: `offchip: false`, `views:` the group's instances. Name it after what it is
   and its first core (`sram_3`, `vmem_8`; `tile_<col>_<row>` for an AIE tile). Its
   `ports`: for each port name `p` the views' core files declare on the instance, `p: {from: p, cores: [every core
   of the group whose instance declares p]}`; an `aie2` view declares `mm2s` and `s2mm` when its core file has a
   `dma`. Cores that declare one port differently (width or energy) need an entry each.
4. **The off-chip memory.** Replace the off-chip core by one memory named after it (its `name`, or `dram` for a
   shim), `offchip: true`, `views: []`:
   - a ZigZag off-chip core: `capacity_bits` is the `size` of its top level holding `I1`; each of its ports `p`
     becomes `p: {bandwidth: <bandwidth_max>, serves: ..., read_energy: <r_cost / bandwidth_max>,
     write_energy: <w_cost / bandwidth_max>}`, the energies written in full (`32000 / 3200 = 10`,
     `23675 / 9624 = 2.4599958437240232`). `serves` maps each `allocation` entry's operand (`I1` -> `input1`,
     `I2` -> `input2`, `O` -> `output`) to `read` for `th` and `tl` and to `write` for `fh` and `fl`; a port serving
     all three operands the same way serves `{any: [...]}`.
   - an `aie2.shim` core: `capacity_bits` is its `memory.capacity`, `dma: {buffer_descriptors, iterations}` its
     `dma`'s, and two ports: `mm2s: {bandwidth: <mm2s x channel_bits>, channels: <mm2s>, serves: {any: [read]},
     read_energy: 0, write_energy: 0}` and `s2mm` the same for `write`. Keep these two ports on every column's
     connection, as the old shim was one; per-column ports are a Decision.

   If no other core file uses the off-chip core's file, delete that file.
5. **Measured bandwidth.** A `bandwidth` key that is a core id becomes the name of that core's memory (the off-chip
   core's: the off-chip memory). A key `<core>.<instance>.<port>` becomes `<memory>.<port name in the new file>`.
   Models are copied unchanged. A memory with a measured bandwidth needs no view ports: give it `ports: {}`, or only
   its DMA ports.
6. **Connections.** For each surviving old connection, in the old order:
   - skip a link at a core that is not the first of its `core_memory_sharing` group when the first one has
     connections of its own: the old Stream entered and left the group through that core;
   - skip a link whose two cores now view one memory: its data moves in place;
   - a bus joins every memory of its cores, each once, the off-chip core's being the off-chip memory;
   - a link joins, for each of its two cores, the memory of its instance holding `I1` (a Decision where the core
     has several memories);
   - two old links that now join the same two memories stay two connections, each a resource of its own;
   - at each end list the ports of that memory a transfer passes: all of them, except that of several ports serving
     the same operand and direction keep only the first by name, as the old Stream did (a level's port serves an
     operand's read where its `allocation` pairs the operand with `th`, its write where with `fh`);
   - copy `type`, `bandwidth` and `unit_energy_cost`.
7. **Core files.** Move a core file's top-level `banks:` under the memory level it describes (`memories.<level>`
   of the instance holding `I1`, or `memory` for an `aie2` core). Change nothing else.
8. **Write** the new accelerator file, keeping the old one until the checks pass. Rewrite its comments to talk about
   memories, without mentioning the old format.

## Decisions to report

Name each that applies, with the files' numbers, and what you did:

- a link the old Stream never used (step 1: a later connection joins the same cores; step 6: a sharing group's
  other core); kept out, it reproduces the old results; put in, it models what the file says;
- a core whose top-level instances are several memories now (step 3): its capacity was the `I1` instance's and its
  links reached all of them; now each instance holds what the solve puts there, and only the memory you name at a
  link end is on that link;
- `memory_aliases`: the aliased memories had their own capacities and were joined by their cores' links; now they are
  one memory, holding its capacity once, with no link inside;
- an `aie2.shim` serving several columns: per-column DMA ports (`mm2s_c<col>`, `s2mm_c<col>`, each on its column's
  link) give every column its own channels;
- what has no place in the new file: the off-chip core's `bandwidth_min`, `latency`, `area`, `operational_array`,
  name and coordinates; a shim's `max_object_fifo_depth` and `memory.bandwidth_*`; dropped cores.

## Checks

Run every one and report its output; the conversion is done when each passes or every difference is a Decision.

1. The new file parses:
   `python -c "from stream.stages.parsing.accelerator_parser import parse_accelerator as p; p('NEW.yaml')"`.
2. The structure and one solve match the old Stream's. Install the Stream that read the old file into a second
   environment (`pip install "stream-dse @ git+https://github.com/KULeuven-MICAS/stream@<its commit>"`), then run
   `python scripts/check_hardware_conversion.py check OLD.yaml NEW.yaml --old-python "<that environment's python>"`
   (add `--mapping` and `--cols` for an AIE array, or `--no-solve` where no workload maps). It prints every
   difference, then a line with the latency and energy of both. Each remaining difference must be a Decision you
   listed; anything else is a conversion error to fix.
3. If you changed a core file, the check above covers it; if another accelerator uses the same core file, check that
   one too.

Finish with: the new file, the files you changed or deleted, the Decisions, and the checker's last line.
