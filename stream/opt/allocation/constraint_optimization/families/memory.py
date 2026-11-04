"""What each memory holds, and the object-fifo depth and buffer descriptors each memory's tensors need."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar

from xdsl.ir.affine import AffineDimExpr

from stream.hardware.architecture.memory import Memory
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.diagnosis import (
    ConstraintTag,
    ResourceKind,
    structural_infeasibility,
)
from stream.opt.allocation.constraint_optimization.families import BUFFERING
from stream.opt.allocation.constraint_optimization.utils import resource_key
from stream.opt.solver import ObjectiveLevel, SolverModel, SolverVar, SolverVarType
from stream.workload.iterator_type import is_state_operand

if TYPE_CHECKING:
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
    """Fail before the model is built when a memory cannot fit the tensors pinned to it under any reuse choice; a copy
    pinned with its source counts only what it holds beyond it, as in :func:`_memory_loads`."""
    held: dict[Tensor, tuple[Memory, int]] = {}
    for node in space.workload.get_iteration_space_nodes():
        carried = [x for x in node.inputs if is_state_operand(node, x)]
        for t in (*node.outputs, *carried):
            candidates = space.candidate_memories(t)
            if len(candidates) != 1:
                continue
            (memory,) = candidates
            tile = space.workload.get_tensor_single_core(t, node, space.mapping)
            whole = tile.size_bits() == t.size_bits()
            copies = 1 if whole else min(choice.count(memory) for choice in space.tensor_choices[t])
            held[t] = (memory, copies * _min_resident_bits(space, t, tile.size_bits()))
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
        cap = space.memory_capacity_bits(memory)
        if bits > cap:
            raise InfeasibleAllocationError(
                structural_infeasibility(
                    f"Memory {memory.name}: tensors pinned to it need at least {bits / 8192:.1f} KB "
                    f"under every reuse choice, but it has {cap / 8192:.1f} KB",
                    model,
                )
            )


class MemoryCapacity:
    """What each memory holds fits in its capacity, less what the toolchain reserves."""

    name: ClassVar[str] = "memory_capacity"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext) -> None:
        """What each memory holds against its capacity: a whole tensor once however many of its positions sit there,
        a tile once per position, a handover by its writer and by a reader that cannot reach it, and an in-place copy
        only what it holds beyond its source."""
        space, ledger = ctx.space, ctx.ledger
        z_stop, z_single = ctx.vars.z_stop, ctx.vars.z_single
        held: dict[tuple[Tensor, Memory], Held] = defaultdict(list)
        for node in space.workload.get_iteration_space_nodes():
            carried = [x for x in node.inputs if is_state_operand(node, x)]
            for t in (*node.outputs, *carried):
                tile = space.workload.get_tensor_single_core(t, node, space.mapping)
                tensor_size = tile.size_bits()
                tile_dims, tile_dtype = _tile_shape(space, node, t, tile)
                whole = tensor_size == t.size_bits()
                holders = (
                    [(m, 1, ctx.tensor_in_memory_var(t, m)) for m in space.candidate_memories(t)]
                    if whole
                    else _holdings(ctx, t)
                )
                least_bytes: dict[Memory, float] = {}
                for memory, copies, u in holders:
                    name = f"{t.name}_{resource_key(memory)}" + ("" if whole else f"_x{copies}")
                    min_req: int | None = None
                    for stop in space.stops(t):
                        req_size = copies * ceil(space.resident_tiles(t, stop) * tensor_size)
                        single = (t, stop) in z_single
                        least = copies * ceil(space.resident_tiles(t, stop, single) * tensor_size)
                        min_req = least if min_req is None else min(min_req, least)
                        uz = ctx.binary_product(
                            a=u,
                            b=z_stop[(t, stop)],
                            base_name=f"memload_{name}_L{stop}",
                            tag=ConstraintTag(memory, MEMORY_CAPACITY, t.name),
                        )
                        held[(t, memory)].append((uz, req_size, t.name))
                        if single:
                            us = ctx.binary_product(
                                a=u,
                                b=z_single[(t, stop)],
                                base_name=f"memsingle_{name}_L{stop}",
                                tag=ConstraintTag(memory),
                            )
                            held[(t, memory)].append((us, least - req_size, t.name))
                    if min_req is not None:
                        least_bytes[memory] = min(least_bytes.get(memory, min_req / 8), min_req / 8)
                for memory, value in least_bytes.items():
                    ledger.terms[(MEMORY_CAPACITY, memory)][t.name] = {
                        "value": value,
                        "dims": tile_dims,
                        "dtype": tile_dtype,
                    }

        load = _memory_loads(ctx, held)

        for one, other, bits in space.handovers:
            for memory in dict.fromkeys((one,) if space.hardware.shares_memory(one, other) else (one, other)):
                load[memory] = load[memory] + bits
                ledger.handover_bits[memory] = ledger.handover_bits.get(memory, 0) + bits
                terms = ledger.terms[(MEMORY_CAPACITY, memory)]
                handed = terms.get("handover", {}).get("value", 0) + bits / 8
                terms["handover"] = {"value": handed, "dims": (), "dtype": ""}

        for memory, expr in load.items():
            cap = space.memory_capacity_bits(memory)
            ctx.add_constr(
                expr <= cap,
                name=f"mem_cap_{resource_key(memory)}",
                resource=memory,
                kind=MEMORY_CAPACITY,
                bound=cap / 8,
            )


def _holdings(ctx: FormulationContext, t: Tensor) -> list[tuple[Memory, int, SolverVar]]:
    """For each memory ``t`` may sit in and each number of positions it can take there: the binary that places it
    so."""
    space = ctx.space
    return [
        (memory, copies, ctx.tensor_in_memory_var(t, memory, copies))
        for memory in space.candidate_memories(t)
        for copies in sorted({choice.count(memory) for choice in space.tensor_choices[t]} - {0})
    ]


def _min_resident_bits(space: DecisionSpace, t: Tensor, tensor_size: int) -> int:
    """The least this tensor can keep resident under any reuse-stop and buffering choice."""
    return min(
        ceil(space.resident_tiles(t, stop, space.may_single_buffer(t, stop)) * tensor_size) for stop in space.stops(t)
    )


def _memory_loads(ctx: FormulationContext, held: dict[tuple[Tensor, Memory], Held]) -> dict[Memory, Any]:
    """Bits each memory holds: every tensor's residency, a copy in the memory of its source (in place, even where the
    transfer also reaches other memories) only what it holds beyond the source."""
    space, ledger = ctx.space, ctx.ledger
    load: dict[Memory, Any] = defaultdict(int)
    copies = {copy: tr.inputs[0] for tr in space.transfer_nodes for copy in tr.outputs}
    for (t, memory), terms in held.items():
        ledger.memory[memory] += terms
        source = held.get((copies[t], memory)) if t in copies else None
        if not source:
            load[memory] = load[memory] + _held_bits(terms)
            continue
        name = f"inplace_{t.name}_{resource_key(memory)}"
        extra = ctx.model.add_var(vtype=SolverVarType.CONTINUOUS, name=name)
        ctx.add_constr(extra >= _held_bits(terms) - _held_bits(source), name=f"{name}_ge", resource=memory)
        load[memory] = load[memory] + extra._raw
        ledger.memory[memory] += [(indicator, -bits, t.name) for indicator, bits, _ in source]
    return load


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
    return dims, str(tile.operand_type)


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
