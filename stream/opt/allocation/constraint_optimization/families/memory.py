"""What each memory holds, and the object-fifo depth and buffer descriptors each core's tensors need."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING, Any, ClassVar

from xdsl.ir.affine import AffineDimExpr

from stream.hardware.architecture.core import Core
from stream.ir.infeasibility import InfeasibleAllocationError
from stream.opt.allocation.constraint_optimization.diagnosis import structural_infeasibility
from stream.opt.allocation.constraint_optimization.families import BUFFERING
from stream.opt.allocation.constraint_optimization.timeslot_allocation import _resource_key
from stream.opt.solver import ObjectiveLevel, SolverModel, SolverVar, SolverVarType
from stream.workload.iterator_type import is_state_operand

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace
    from stream.workload.workload import Tensor, TransferNode

# (indicator, bits when it is 1, tensor): one term of what holding a tensor in a memory takes
Held = list[tuple[SolverVar, int, str]]


def _held_bits(terms: Held) -> Any:
    """Bits a list of residency terms holds."""
    return sum(bits * indicator._raw for indicator, bits, _ in terms)


class MemoryCapacity:
    """What each memory holds fits in its capacity, less what the toolchain reserves."""

    name: ClassVar[str] = "memory_capacity"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        """What each memory holds, keyed by the core owning it, so cores sharing one memory share its capacity."""
        space, ledger = ctx.space, ctx.ledger
        z_stop, z_single = ctx.vars.z_stop, ctx.vars.z_single
        held: dict[tuple[Tensor, Core], Held] = defaultdict(list)
        for node in space.workload.get_iteration_space_nodes():
            # A node's outputs, and the state it keeps resident while it runs there.
            carried = [x for x in node.inputs if is_state_operand(node, x)]
            for t in (*node.outputs, *carried):
                tile = space.workload.get_tensor_single_core(t, node, space.mapping)
                tensor_size = tile.size_bits()
                tile_dims, tile_dtype = _tile_shape(space, node, t, tile)
                sharing: dict[Core, list[Core]] = defaultdict(list)
                for c in space.candidate_cores(t):
                    sharing[space.accelerator.memory_of(c)].append(c)
                # A tile that is the whole tensor is the same data on every core holding it, so a memory
                # those cores share holds it once; a tile of a split tensor is a different part on each.
                whole = tensor_size == t.size_bits()
                holders = [
                    (memory, holder, ctx.tensor_in_memory_var(t, cores if whole else [holder]))
                    for memory, cores in sharing.items()
                    for holder in ([memory] if whole else cores)
                ]
                least_bytes: dict[Core, float] = defaultdict(float)
                for memory, holder, u in holders:
                    min_req: int | None = None
                    for stop in space.stops(t):
                        req_size = ceil(space.resident_tiles(t, stop) * tensor_size)
                        single = (t, stop) in z_single
                        least = ceil(space.resident_tiles(t, stop, single) * tensor_size)
                        min_req = least if min_req is None else min(min_req, least)
                        uz = ctx.binary_product(
                            a=u, b=z_stop[(t, stop)], base_name=f"memload_{t.name}_{_resource_key(holder)}_L{stop}"
                        )
                        held[(t, memory)].append((uz, req_size, t.name))
                        if single:
                            # The second buffer the window would rotate through is not held.
                            us = ctx.binary_product(
                                a=u,
                                b=z_single[(t, stop)],
                                base_name=f"memsingle_{t.name}_{_resource_key(holder)}_L{stop}",
                            )
                            held[(t, memory)].append((us, least - req_size, t.name))
                    if min_req is not None:  # bytes this tensor's tile adds if resident on holder
                        least_bytes[memory] += min_req / 8
                for memory, value in least_bytes.items():
                    ledger.terms[("memory_capacity", memory.id)][t.name] = {
                        "value": value,
                        "dims": tile_dims,
                        "dtype": tile_dtype,
                    }

        load = _memory_loads(ctx, held)

        # The core that writes a handover holds it. A core reading one out of memory it
        # already shares reads it in place; one further away is given a copy of its own.
        for one, other, bits in space.handovers:
            readers = (one,) if space.context.shares_memory(one, other) else (one, other)
            for memory in dict.fromkeys(space.accelerator.memory_of(c) for c in readers):
                load[memory] = load[memory] + bits
                ledger.handover_bits[memory.id] = ledger.handover_bits.get(memory.id, 0) + bits
                terms = ledger.terms[("memory_capacity", memory.id)]
                handed = terms.get("handover", {}).get("value", 0) + bits / 8
                terms["handover"] = {"value": handed, "dims": (), "dtype": ""}

        for memory, expr in load.items():
            cap = space.memory_capacity_bits(memory)
            ledger.bounds[("memory_capacity", memory.id)] = cap / 8  # bytes
            ctx.add_resource_constr(
                expr <= cap, name=f"mem_cap_{_resource_key(memory)}", kind="memory_capacity", resource=memory
            )


def capacity_screen(space: DecisionSpace, model: SolverModel) -> None:
    """Fail before building the model when a memory cannot fit its pinned tensors under any reuse choice."""
    pinned: dict[Core, int] = defaultdict(int)
    for node in space.workload.get_iteration_space_nodes():
        carried = [x for x in node.inputs if is_state_operand(node, x)]
        for t in (*node.outputs, *carried):
            candidates = space.candidate_cores(t)
            if len(candidates) != 1:
                continue
            (c,) = candidates
            tile = space.workload.get_tensor_single_core(t, node, space.mapping)
            pinned[space.accelerator.memory_of(c)] += _min_resident_bits(space, t, tile.size_bits())
    for c, bits in pinned.items():
        cap = space.memory_capacity_bits(c)
        if bits > cap:
            raise InfeasibleAllocationError(
                structural_infeasibility(
                    f"Core {c.id}: tensors pinned to it need at least {bits / 8192:.1f} KB "
                    f"under every reuse choice, but its memory is {cap / 8192:.1f} KB",
                    model,
                )
            )


def _min_resident_bits(space: DecisionSpace, t: Tensor, tensor_size: int) -> int:
    """The least this tensor can keep resident under any reuse-stop and buffering choice."""
    return min(
        ceil(space.resident_tiles(t, stop, space.may_single_buffer(t, stop)) * tensor_size) for stop in space.stops(t)
    )


def _memory_loads(ctx: FormulationContext, held: dict[tuple[Tensor, Core], Held]) -> dict[Core, Any]:
    """Bits each memory holds: every tensor's residency, an in-place copy only what it holds beyond its source."""
    space, ledger = ctx.space, ctx.ledger
    load: dict[Core, Any] = defaultdict(int)
    copies = {copy: tr.inputs[0] for tr in space.transfer_nodes if space.within_one_memory(tr) for copy in tr.outputs}
    for (t, memory), terms in held.items():
        # Keep the indicators + coefficients so the occupancy report can recompute the solved residency.
        ledger.memory[memory.id] += terms
        if t not in copies:
            load[memory] = load[memory] + _held_bits(terms)
            continue
        # A copy within one memory is its source's buffer: it adds only what it holds beyond the source,
        # and is reported as its residency less the source's.
        source = held.get((copies[t], memory), [])
        name = f"inplace_{t.name}_{_resource_key(memory)}"
        extra = ctx.model.add_var(vtype=SolverVarType.CONTINUOUS, name=name)
        ctx.model.add_constr(extra >= _held_bits(terms) - _held_bits(source), name=f"{name}_ge")
        load[memory] = load[memory] + extra._raw
        ledger.memory[memory.id] += [(indicator, -bits, t.name) for indicator, bits, _ in source]
    return load


def _tile_shape(space: DecisionSpace, node: Any, tensor: Any, tile: Any) -> tuple[list[tuple[str, int]], str]:
    """The per-dimension tile sizes of ``tensor`` on one core, each labelled by its loop-dim symbol
    (the same symbols the affine graph view shows), plus the dtype -- so a memory term shows *why* a
    tile is large, not just its total. Best-effort: falls back to bare axis sizes."""
    dtype = str(getattr(tile, "operand_type", "")) if tile is not None else ""
    shape = tuple(getattr(tile, "shape", ()) or ())
    try:
        results = node.get_mapping(tensor).results
        node_dims = space.workload.get_dims(node)
        dims: list[tuple[str, int]] = []
        for i, r in enumerate(results):
            if i >= len(shape):
                break
            if isinstance(r, AffineDimExpr) and r.position < len(node_dims):
                label = str(node_dims[r.position])
            else:
                label = f"axis{i}"
            dims.append((label, int(shape[i])))
        return dims, dtype
    except Exception:  # noqa: BLE001 -- shape labelling is best-effort diagnostics
        return [(f"axis{i}", int(s)) for i, s in enumerate(shape)], dtype


class ObjectFifoDepth:
    """The object-fifo depth each core's tensors need, per core, which a namespace family bounds; the buffering
    depth is the objective level after the off-chip traffic."""

    name: ClassVar[str] = "object_fifo_depth"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("object_fifo_depth",)

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        space, ledger, z_stop = ctx.space, ctx.ledger, ctx.vars.z_stop
        depth: dict[Core, Any] = defaultdict(int)
        for tr in space.transfer_nodes:
            # TODO: Confirm assumption that OF linking causes only single object fifo depth increase
            for t in tr.outputs:
                for c in space.candidate_cores(t):
                    if c.id == space.offchip_core_id:
                        continue
                    assert isinstance(c, Core)
                    u = ctx.tensor_uses_core_var(t, c)
                    min_tiles: int | None = None
                    counted = space.bds_needed_levels if c.type == "memory" else space.tiles_needed_levels
                    for stop in space.stops(t):
                        tiles_needed = counted[(t, stop)]
                        min_tiles = tiles_needed if min_tiles is None else min(min_tiles, tiles_needed)
                        uz = ctx.binary_product(
                            a=u, b=z_stop[(t, stop)], base_name=f"objfifo_{t.name}_{_resource_key(c)}_L{stop}"
                        )
                        depth[c] = depth[c] + tiles_needed * uz._raw
                        ledger.loads[("object_fifo_depth", c.id)].append((uz, tiles_needed))
                    if min_tiles is not None:
                        ledger.terms[("object_fifo_depth", c.id)][t.name] = min_tiles
        ledger.record_capacity_bounds("object_fifo_depth", depth, "aie2_obj_fifo_depth")
        for core, expr in depth.items():
            q.add("object_fifo_depth", expr, index=core)

    def objective(self, ctx: FormulationContext, q: QuantityRegistry) -> list[ObjectiveLevel]:
        space, z_stop = ctx.space, ctx.vars.z_stop
        buffering = ctx.model.quicksum(
            space.tiles_needed_levels[(t, s)] * z_stop[(t, s)]._raw
            for t in space.tensors_to_optimize_reuse_for
            for s in space.stops(t)
        )
        return [ObjectiveLevel(expr=buffering._raw, priority=BUFFERING, name="buffering")]


class BufferDescriptors:
    """The buffer descriptors each core's transfers need, per core, which a namespace family bounds: a compute tile
    needs one per tile it holds, a memory tile one per tile it repeats while its reader does not reuse it."""

    name: ClassVar[str] = "buffer_descriptors"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("buffer_descriptor_depth",)

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        space, ledger = ctx.space, ctx.ledger
        bd_depth: dict[Core, Any] = defaultdict(int)
        for tr in space.transfer_nodes:
            for t in tr.tensors:
                for c in space.candidate_cores(t):
                    if c.id == space.offchip_core_id:
                        continue
                    u = ctx.tensor_uses_core_var(t, c)
                    assert isinstance(c, Core)
                    terms = _descriptor_terms(ctx, tr, t, c, u)
                    for indicator, bds_needed in terms:
                        bd_depth[c] = bd_depth[c] + bds_needed * indicator._raw
                        ledger.loads[("buffer_descriptors", c.id)].append((indicator, bds_needed))
                    if terms:
                        ledger.terms[("buffer_descriptors", c.id)][t.name] = min(n for _, n in terms)
        ledger.record_capacity_bounds("buffer_descriptors", bd_depth, "aie2_bd_depth")
        for core, expr in bd_depth.items():
            q.add("buffer_descriptor_depth", expr, index=core)


def _descriptor_terms(
    ctx: FormulationContext, tr: TransferNode, t: Tensor, c: Core, u: SolverVar
) -> list[tuple[SolverVar, int]]:
    """Per reuse stop of ``t`` on ``c``: the binary that holds there, and the descriptors it then takes."""
    space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
    terms: list[tuple[SolverVar, int]] = []
    if c.type == "compute":
        for stop in space.stops(t):
            uz = ctx.binary_product(a=u, b=z_stop[(t, stop)], base_name=f"bddepth_{t.name}_{_resource_key(c)}_L{stop}")
            terms.append((uz, space.tiles_needed_levels[(t, stop)]))
        return terms
    # If the core is a memory core, we add bd usage only if the eq. tensor on compute
    # is not being reused (zStop[t, stop] == 0 at that reuse level)
    # This means we create a new 'active' helper variable for the eq. tensor
    # TODO: Shouldn't just be exactly that compute tensor reuse level
    if t in tr.outputs:
        assert len(tr.inputs) == 1
        compute_tensor = tr.inputs[0]
    elif t in tr.inputs:
        # TODO: Check that for multiple outputs the reuse levels are equivalent,
        # otherswise we may need to create separate active variables for each output tensor.
        compute_tensor = tr.outputs[0]
    else:
        raise NotImplementedError("Expected tensor to be either input or output of the transfer.")
    compute_levels = len(space.ssis[compute_tensor].get_applicable_temporal_variables())
    for stop in space.stops(t):
        src_tensor_reuse = z_stop[(compute_tensor, stop)] if stop < compute_levels else 0
        gate_var = model.add_var(
            vtype=SolverVarType.BINARY, name=f"active_{compute_tensor.name}_{_resource_key(c)}_L{stop}"
        )
        model.add_constr(
            gate_var == 1 - src_tensor_reuse, name=f"active_gate_{compute_tensor.name}_{_resource_key(c)}_L{stop}"
        )
        uz = ctx.binary_product(a=u, b=z_stop[(t, stop)], base_name=f"bddepth_{t.name}_{_resource_key(c)}_L{stop}")
        uzgate = ctx.binary_product(a=uz, b=gate_var, base_name=f"bddepth_active_{t.name}_{_resource_key(c)}_L{stop}")
        terms.append((uzgate, space.bds_needed_levels[(t, stop)]))
    return terms
