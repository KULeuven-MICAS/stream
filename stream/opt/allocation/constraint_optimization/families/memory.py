"""What each memory holds, and the object-fifo depth and buffer descriptors each memory's tensors need."""

from __future__ import annotations

from collections import defaultdict
from math import ceil, inf
from typing import TYPE_CHECKING, Any, ClassVar

from xdsl.ir.affine import AffineDimExpr

from stream.hardware.architecture.memory import Memory
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.mapping.mapping import OFFCHIP_CAPACITY, whole_bits
from stream.opt.allocation.constraint_optimization.diagnosis import (
    ConstraintTag,
    ResourceKind,
    structural_infeasibility,
)
from stream.opt.allocation.constraint_optimization.families import BUFFERING, OFFCHIP_FOOTPRINT
from stream.opt.allocation.constraint_optimization.utils import resource_key
from stream.opt.solver import ObjectiveLevel, SolverModel, SolverVar, SolverVarType
from stream.workload.iterator_type import is_state_operand

if TYPE_CHECKING:
    from collections.abc import Iterable

    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.workload.workload import Tensor, TransferNode

Held = list[tuple[SolverVar, int, str]]
"""What holding a tensor in a memory takes: per term an indicator, the bits it holds when 1, and the tensor."""

MEMORY_CAPACITY = ResourceKind(
    "memory_capacity",
    "on-chip memory capacity exceeded",
    unit="bytes",
    demand_label="tensors resident in",
    bound_label="capacity of",
    demand_input="workload tensor sizes × mapping intra-core tiling",
    bound_input="hardware spec: memory size",
    term_detail="min resident (1 tile)",
    levers=(
        "Increase {resource} (hardware spec)",
        "Tile the fused group finer so fewer / smaller tiles are resident (mapping intra_core_tiling)",
        "Reduce the workload's tensor sizes (fewer channels, smaller spatial, shorter sequence)",
    ),
)
OBJECT_FIFO_DEPTH = ResourceKind(
    "object_fifo_depth",
    "object-FIFO depth exceeded",
    unit="FIFO slots",
    demand_label="concurrently buffered tiles on",
    bound_label="object-FIFO depth of",
    demand_input="mapping reuse levels × fused tiles",
    bound_input="hardware spec: tile object-FIFO depth",
    term_detail="min buffered tiles",
    levers=(
        "Increase {resource}'s object-FIFO depth (hardware spec)",
        "Lower buffering: reduce reuse / double-buffering (mapping)",
        "Route fewer tensors through {resource} (mapping)",
    ),
)
BUFFER_DESCRIPTORS = ResourceKind(
    "buffer_descriptors",
    "buffer-descriptor count exceeded",
    unit="descriptors",
    demand_label="buffer descriptors on",
    bound_label="buffer-descriptor budget of",
    demand_input="mapping tiling × transfers through the tile",
    bound_input="hardware spec: tile buffer-descriptor count",
    term_detail="min descriptors",
    levers=(
        "Increase {resource}'s buffer-descriptor budget (hardware spec)",
        "Reduce distinct transfers / reuse levels through {resource} (mapping)",
        "Fuse fewer tensors through {resource} (mapping)",
    ),
)


def _held_bits(terms: Held) -> Any:
    """Bits a list of residency terms holds."""
    return sum(bits * indicator._raw for indicator, bits, _ in terms)


def capacity_screen(space: DecisionSpace, model: SolverModel) -> None:
    """Fail before the model is built when a memory cannot fit the tensors pinned to it under any reuse choice beside
    the inputs and outputs placed in it; a copy pinned with its source or a placement counts only what it holds beyond
    it, as in :func:`_memory_loads`."""
    held: dict[Tensor, tuple[Memory, int]] = {}
    for node in space.workload.get_iteration_space_nodes():
        carried = [x for x in node.inputs if is_state_operand(node, x)]
        for t in (*node.outputs, *carried):
            candidates = space.candidate_memories(t)
            if len(candidates) != 1:
                continue
            (memory,) = candidates
            if (share := _placed_share(space, t, memory)) == inf:
                continue
            tile = space.workload.get_tensor_single_core(t, node, space.mapping)
            whole = tile.size_bits() == t.size_bits()
            copies = 1 if whole else min(choice.count(memory) for choice in space.tensor_choices[t])
            least = _min_resident_bits(space, t, space.held_bits(t, tile), memory) + space.accumulator_bits(t, tile)
            held[t] = (memory, max(0, ceil(copies * least - share)))
    pinned: dict[Memory, int] = defaultdict(int)
    for memory, bits in held.values():
        pinned[memory] += bits
    for tr in space.transfer_nodes:
        if tr.inputs[0] in held:
            source_memory, source_bits = held[tr.inputs[0]]
            for copy in tr.outputs:
                if copy in held and held[copy][0] == source_memory:
                    pinned[source_memory] -= min(held[copy][1], source_bits)
    for memory, bits in pinned.items():
        placed = _placed(space, memory)
        cap = space.memory_capacity_bits(memory) - sum(placed.values())
        if bits > cap:
            beside = f" beside {', '.join(t.name for t in placed)} placed in it" if placed else ""
            raise InfeasibleAllocationError(
                structural_infeasibility(
                    f"Memory {memory.name}: tensors pinned to it need at least {bits / 8192:.1f} KB "
                    f"under every reuse choice, but it has {cap / 8192:.1f} KB{beside}",
                    model,
                )
            )


class MemoryCapacity:
    """What each memory holds fits in its capacity, less what the toolchain reserves and the workload's inputs and
    outputs placed in it hold for the whole run. The off-chip bits of an input whose placement the allocation chooses
    rank after latency, so it is copied only where that is faster."""

    name: ClassVar[str] = "memory_capacity"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("offchip_footprint", "memory_objects")

    def build(self, ctx: FormulationContext) -> None:
        """What each memory holds against its capacity: a whole tensor once however many of its positions sit there,
        a tile once per position, a handover by its writer and by a reader that cannot reach it, and an in-place copy
        only what it holds beyond its source."""
        space, ledger = ctx.space, ctx.ledger
        z_stop, z_single = ctx.vars.z_stop, ctx.vars.z_single
        held: dict[tuple[Tensor, Memory], Held] = defaultdict(list)
        objects: list[tuple[Memory, Any, int]] = []
        least_held: dict[tuple[Tensor, Memory], tuple[int, dict[str, Any]]] = {}
        for node in space.workload.get_iteration_space_nodes():
            carried = [x for x in node.inputs if is_state_operand(node, x)]
            for t in (*node.outputs, *carried):
                tile = space.workload.get_tensor_single_core(t, node, space.mapping)
                tensor_size, accumulator = space.held_bits(t, tile), space.accumulator_bits(t, tile)
                tile_dims, tile_dtype = _tile_shape(space, node, t, tile)
                whole = tile.size_bits() == t.size_bits()
                holders = (
                    [(m, 1, ctx.tensor_in_memory_var(t, m)) for m in space.candidate_memories(t)]
                    if whole
                    else _holdings(ctx, t)
                )
                for memory, copies, u in holders:
                    name = f"{t.name}_{resource_key(memory)}" + ("" if whole else f"_x{copies}")
                    min_req: int | None = None
                    staged = memory.kind == "memory"
                    for stop in space.stops(t):
                        beside = space.halo_bits(t, stop) + accumulator * space.summed_tiles(t, stop)
                        resident = space.resident_tiles(t, stop, staged=staged)
                        req_size = copies * (ceil(resident * tensor_size) + beside)
                        single = (t, stop) in z_single
                        resident = space.resident_tiles(t, stop, single, staged)
                        least = copies * (ceil(resident * tensor_size) + beside)
                        min_req = least if min_req is None else min(min_req, least)
                        uz = ctx.binary_product(
                            a=u,
                            b=z_stop[(t, stop)],
                            base_name=f"memload_{name}_L{stop}",
                            tag=ConstraintTag(memory, MEMORY_CAPACITY, t.name),
                        )
                        held[(t, memory)].append((uz, req_size, t.name))
                        objects.append((memory, uz, ceil(space.resident_tiles(t, stop, True, staged) * tensor_size)))
                        if single:
                            us = ctx.binary_product(
                                a=u,
                                b=z_single[(t, stop)],
                                base_name=f"memsingle_{name}_L{stop}",
                                tag=ConstraintTag(memory),
                            )
                            held[(t, memory)].append((us, least - req_size, t.name))
                    if min_req is not None and min_req < least_held.get((t, memory), (inf,))[0]:
                        least_held[(t, memory)] = (min_req, {"dims": tile_dims, "dtype": tile_dtype})

        load = _memory_loads(ctx, held)
        _record_least(ctx, least_held)
        ctx.quantities.add("memory_objects", tuple(objects))

        for one, other, bits in space.handovers:
            for memory in dict.fromkeys((one,) if space.hardware.shares_memory(one, other) else (one, other)):
                load[memory] = load[memory] + bits
                ledger.handover_bits[memory] = ledger.handover_bits.get(memory, 0) + bits
                terms = ledger.terms[(MEMORY_CAPACITY, memory)]
                handed = terms.get("handover", {}).get("value", 0) + bits / 8
                terms["handover"] = {"value": handed, "dims": (), "dtype": ""}

        for memory, bits in space.kernel_scratch.items():
            load[memory] = load[memory] + bits
            ledger.terms[(MEMORY_CAPACITY, memory)]["kernel scratch"] = {"value": bits / 8, "dims": (), "dtype": ""}

        for memory, expr in load.items():
            cap = space.memory_capacity_bits(memory)
            placed = _placed(space, memory)
            terms = ledger.terms[(MEMORY_CAPACITY, memory)]
            for tensor, bits in placed.items():
                if tensor.name in terms:
                    terms[f"{tensor.name} (outside its placement)"] = terms.pop(tensor.name)
                terms[tensor.name] = {"value": bits / 8, "dims": (), "dtype": ""}
            ctx.add_constr(
                expr <= cap - sum(placed.values()),
                name=f"mem_cap_{resource_key(memory)}",
                resource=memory,
                kind=MEMORY_CAPACITY,
                bound=cap / 8,
            )
        _offchip_capacity(ctx)

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        if "offchip_footprint" not in ctx.quantities:
            return []
        footprint = ctx.quantities.get("offchip_footprint").expr
        return [ObjectiveLevel(expr=footprint, priority=OFFCHIP_FOOTPRINT, name="offchip_footprint")]


def _offchip_capacity(ctx: FormulationContext) -> None:
    """Each off-chip memory fits the settled workload tensors and what the chosen route's placement holds of each input
    the allocation places, diagnosed at a block; those bits, relative to the largest such input, are the
    ``offchip_footprint``."""
    space, ledger = ctx.space, ctx.ledger
    blocks = len(space.accelerator.offchip_memories)
    chosen: dict[Memory, list[tuple[float, SolverVar]]] = defaultdict(list)
    for tr in space.transfer_nodes:
        bits = whole_bits(tr.inputs[0])
        options = {choice: choice.placement.shares() for choice in space.path_choices[tr] if choice.placement}
        for memory in dict.fromkeys(memory for shares in options.values() for memory in shares):
            chosen[memory] += [(float(s.get(memory, 0) * bits), ctx.vars.y[(tr, c)]) for c, s in options.items()]
            ledger.terms[(OFFCHIP_CAPACITY, memory)][tr.inputs[0].name] = bits / blocks / 8
    for memory, terms in chosen.items():
        settled = space.problem.held.get(memory, {})
        ledger.terms[(OFFCHIP_CAPACITY, memory)] |= {tensor.name: bits / 8 for tensor, bits in settled.items()}
        cap = space.memory_capacity_bits(memory)
        ctx.add_constr(
            ctx.model.quicksum(bits * y._raw for bits, y in terms) <= cap - sum(settled.values()),
            name=f"offchip_cap_{resource_key(memory)}",
            resource=memory,
            kind=OFFCHIP_CAPACITY,
            bound=cap / 8,
        )
    if chosen:
        largest = max(bits for terms in chosen.values() for bits, _ in terms)
        footprint = ctx.model.quicksum(bits / largest * y._raw for terms in chosen.values() for bits, y in terms)
        ctx.quantities.add("offchip_footprint", footprint._raw)


def _placed(space: DecisionSpace, memory: Memory) -> dict[Tensor, float]:
    """The bits of each workload input and output placed in on-chip ``memory``, held there for the whole run."""
    return {} if memory.offchip else space.problem.held.get(memory, {})


def _placed_share(space: DecisionSpace, t: Tensor, memory: Memory) -> float:
    """The bits ``memory`` holds, placed, of ``t`` or of the tensor ``t`` copies (by name, which an output's copy shares
    with the workload's output): 0 for none, ``inf`` where the placement holds it whole."""
    placed = {tensor.name: (tensor, bits) for tensor, bits in _placed(space, memory).items()}
    names = (t.name, *(tr.inputs[0].name for tr in space.transfer_nodes if t in tr.outputs))
    tensor, bits = next((placed[name] for name in names if name in placed), (None, 0.0))
    return inf if tensor is not None and bits >= whole_bits(tensor) else bits


def _beyond(terms: Held, share: float) -> Held:
    """The residency terms of ``terms`` charging only what each holds beyond ``share`` bits."""
    beyond: Held = []
    req = 0
    for indicator, bits, name in terms:
        # A negative term trims the single-buffered stop of the term before it.
        if bits >= 0:
            req = bits
            beyond.append((indicator, max(0, ceil(bits - share)), name))
        else:
            beyond.append((indicator, max(0, ceil(req + bits - share)) - max(0, ceil(req - share)), name))
    return beyond


def _holdings(ctx: FormulationContext, t: Tensor) -> list[tuple[Memory, int, SolverVar]]:
    """For each memory ``t`` may sit in and each number of positions it can take there: the binary that places it
    so."""
    space = ctx.space
    return [
        (memory, copies, ctx.tensor_in_memory_var(t, memory, copies))
        for memory in space.candidate_memories(t)
        for copies in sorted({choice.count(memory) for choice in space.tensor_choices[t]} - {0})
    ]


def _min_resident_bits(space: DecisionSpace, t: Tensor, tensor_size: int, memory: Memory) -> int:
    """The least this tensor can keep resident in ``memory`` under any reuse-stop and buffering choice, with the halos
    its sliding loops keep."""
    staged = memory.kind == "memory"
    return min(
        ceil(space.resident_tiles(t, stop, space.may_single_buffer(t, stop), staged) * tensor_size)
        + space.halo_bits(t, stop)
        for stop in space.stops(t)
    )


def _memory_loads(ctx: FormulationContext, held: dict[tuple[Tensor, Memory], Held]) -> dict[Memory, Any]:
    """Bits each memory holds: every tensor's residency, a copy in the memory of its source (in place, even where the
    transfer also reaches other memories) only what it holds beyond the source, and a tensor placed in the memory or a
    copy of one only what it holds beyond the placement's block there (:func:`_placed_share`)."""
    space, ledger = ctx.space, ctx.ledger
    load: dict[Memory, Any] = defaultdict(int)
    charges = _counted_beyond(space, held)
    charged = {key: _beyond(held[key], share) for key, (share, _) in charges.items()}
    for (t, memory), terms in charged.items():
        ledger.memory[memory] += terms
        source = charged[source_key] if (source_key := charges[(t, memory)][1]) else None
        if not source:
            load[memory] = load[memory] + _held_bits(terms)
            continue
        name = f"inplace_{t.name}_{resource_key(memory)}"
        extra = ctx.model.add_var(vtype=SolverVarType.CONTINUOUS, name=name)
        ctx.add_constr(extra >= _held_bits(terms) - _held_bits(source), name=f"{name}_ge", resource=memory)
        load[memory] = load[memory] + extra._raw
        ledger.memory[memory] += [(indicator, -bits, t.name) for indicator, bits, _ in source]
    return load


def _record_least(ctx: FormulationContext, least: dict[tuple[Tensor, Memory], tuple[int, dict[str, Any]]]) -> None:
    """Each tensor's least residency in a memory as a diagnosis term, counted as :func:`_memory_loads` counts it:
    beyond the placement's block there, and for a copy in its source's memory beyond the source."""
    charges = _counted_beyond(ctx.space, least)
    beyond = {key: max(0, least[key][0] - share) for key, (share, _) in charges.items()}
    for (t, memory), bits in beyond.items():
        value = max(0, bits - beyond[source_key]) if (source_key := charges[(t, memory)][1]) else bits
        ctx.ledger.terms[(MEMORY_CAPACITY, memory)][t.name] = {"value": value / 8, **least[(t, memory)][1]}


def _counted_beyond(
    space: DecisionSpace, keys: Iterable[tuple[Tensor, Memory]]
) -> dict[tuple[Tensor, Memory], tuple[float, tuple[Tensor, Memory] | None]]:
    """For each (tensor, memory) of ``keys`` a placement there does not hold whole: the placement's share of it, and
    for a copy whose source is also among them in that memory, the source's key, which it is charged beyond."""
    copies = {copy: tr.inputs[0] for tr in space.transfer_nodes for copy in tr.outputs}
    shares = {key: share for key in keys if (share := _placed_share(space, *key)) != inf}
    return {
        (t, memory): (share, (copies[t], memory) if t in copies and (copies[t], memory) in shares else None)
        for (t, memory), share in shares.items()
    }


def _tile_shape(space: DecisionSpace, node: Any, tensor: Tensor, tile: Tensor) -> tuple[list[tuple[str, int]], str]:
    """The per-dimension tile sizes of ``tensor`` on one core, each labelled by its loop-dim symbol (the symbols the
    affine graph view shows) or its axis, and the dtype -- so a memory term shows why a tile is large."""
    node_dims = space.workload.get_dims(node)
    shape = tuple(tile.shape)
    dims = [
        (
            str(node_dims[r.position]) if isinstance(r, AffineDimExpr) and r.position < len(node_dims) else f"axis{i}",
            int(size),
        )
        for i, (r, size) in enumerate(zip(node.get_mapping(tensor).results, shape, strict=False))
    ]
    return dims, space.running_sum_format(tensor) or str(tile.operand_type)


class ObjectFifoDepth:
    """The object-fifo depth each memory's tensors need, counted once per position a tensor takes in it (each core
    reads its own fifo), which a namespace family bounds. Also the buffering depth, the objective level after the
    off-chip traffic; with ``depth`` False only the buffering level."""

    name: ClassVar[str] = "object_fifo_depth"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("object_fifo_depth",)

    def __init__(self, depth: bool = True) -> None:
        self.depth = depth

    def build(self, ctx: FormulationContext) -> None:
        if not self.depth:
            return
        q = ctx.quantities
        space, ledger, z_stop = ctx.space, ctx.ledger, ctx.vars.z_stop
        depth: dict[Memory, Any] = defaultdict(int)
        for tr in space.transfer_nodes:
            for t in tr.outputs:
                least: dict[Memory, int] = {}
                for memory, copies, u in _holdings(ctx, t):
                    if memory.offchip:
                        continue
                    counted = space.bds_needed_levels if memory.kind == "memory" else space.tiles_needed_levels
                    for stop in space.stops(t):
                        tiles_needed = copies * counted[(t, stop)]
                        least[memory] = min(least.get(memory, tiles_needed), tiles_needed)
                        uz = ctx.binary_product(
                            a=u,
                            b=z_stop[(t, stop)],
                            base_name=f"objfifo_{t.name}_{resource_key(memory)}_x{copies}_L{stop}",
                            tag=ConstraintTag(memory, OBJECT_FIFO_DEPTH, t.name),
                        )
                        depth[memory] = depth[memory] + tiles_needed * uz._raw
                        ledger.loads[(OBJECT_FIFO_DEPTH, memory)].append((uz, tiles_needed))
                for memory, tiles in least.items():
                    ledger.terms[(OBJECT_FIFO_DEPTH, memory)][t.name] = tiles
        for memory, expr in depth.items():
            q.add("object_fifo_depth", expr, index=memory)

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        space, z_stop = ctx.space, ctx.vars.z_stop
        buffering = ctx.model.quicksum(
            space.tiles_needed_levels[(t, s)] * z_stop[(t, s)]._raw
            for t in space.tensors_to_optimize_reuse_for
            for s in space.stops(t)
        )
        return [ObjectiveLevel(expr=buffering._raw, priority=BUFFERING, name="buffering")]


class BufferDescriptors:
    """The buffer descriptors each memory's transfers need, counted once per position a tensor takes in it, which a
    namespace family bounds: a compute memory needs one per tile it holds, a staging memory one per tile it repeats
    while its reader does not reuse it."""

    name: ClassVar[str] = "buffer_descriptors"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("buffer_descriptor_depth",)

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        space, ledger = ctx.space, ctx.ledger
        bd_depth: dict[Memory, Any] = defaultdict(int)
        for tr in space.transfer_nodes:
            for t in tr.tensors:
                least: dict[Memory, int] = {}
                for memory, copies, u in _holdings(ctx, t):
                    if memory.offchip:
                        continue
                    terms = _descriptor_terms(ctx, tr, t, memory, copies, u)
                    for indicator, bds_needed in terms:
                        bd_depth[memory] = bd_depth[memory] + bds_needed * indicator._raw
                        ledger.loads[(BUFFER_DESCRIPTORS, memory)].append((indicator, bds_needed))
                    if terms:
                        fewest = min(n for _, n in terms)
                        least[memory] = min(least.get(memory, fewest), fewest)
                for memory, descriptors in least.items():
                    ledger.terms[(BUFFER_DESCRIPTORS, memory)][t.name] = descriptors
        for memory, expr in bd_depth.items():
            q.add("buffer_descriptor_depth", expr, index=memory)


def _descriptor_terms(
    ctx: FormulationContext, tr: TransferNode, t: Tensor, c: Memory, copies: int, u: SolverVar
) -> list[tuple[SolverVar, int]]:
    """Per reuse stop of ``t`` at ``copies`` positions in ``c``: the binary that holds there, and the descriptors it
    then takes."""
    space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
    terms: list[tuple[SolverVar, int]] = []
    held = ConstraintTag(c, BUFFER_DESCRIPTORS, t.name)
    name = f"{t.name}_{resource_key(c)}_x{copies}"
    if c.kind == "compute":
        for stop in space.stops(t):
            uz = ctx.binary_product(a=u, b=z_stop[(t, stop)], base_name=f"bddepth_{name}_L{stop}", tag=held)
            terms.append((uz, copies * space.tiles_needed_levels[(t, stop)]))
        return terms
    if t in tr.outputs:
        assert len(tr.inputs) == 1
        compute_tensor = tr.inputs[0]
    elif t in tr.inputs:
        compute_tensor = tr.outputs[0]
    else:
        raise NotImplementedError("Expected tensor to be either input or output of the transfer.")
    compute_levels = len(space.ssis[compute_tensor].get_applicable_temporal_variables())
    for stop in space.stops(t):
        src_tensor_reuse = z_stop[(compute_tensor, stop)] if stop < compute_levels else 0
        gate_var = model.add_var(
            vtype=SolverVarType.BINARY, name=f"active_{compute_tensor.name}_{resource_key(c)}_x{copies}_L{stop}"
        )
        ctx.add_constr(
            gate_var == 1 - src_tensor_reuse,
            name=f"active_gate_{compute_tensor.name}_{resource_key(c)}_x{copies}_L{stop}",
            resource=c,
        )
        uz = ctx.binary_product(a=u, b=z_stop[(t, stop)], base_name=f"bddepth_{name}_L{stop}", tag=held)
        uzgate = ctx.binary_product(
            a=uz, b=gate_var, base_name=f"bddepth_active_{name}_L{stop}", tag=ConstraintTag(c, BUFFER_DESCRIPTORS)
        )
        terms.append((uzgate, copies * space.bds_needed_levels[(t, stop)]))
    return terms
