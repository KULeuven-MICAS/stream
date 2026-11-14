"""How long each tensor stays resident: its reuse stop, what that stop makes a transfer serve, and the stops the
target forces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from stream.hardware.architecture.memory import Memory
from stream.opt.allocation.constraint_optimization.diagnosis import StructuralRule
from stream.opt.solver import SolverVarType
from stream.workload.node import TransferType
from stream.workload.steady_state.iteration_space import IterationVariableType, LoopEffect

if TYPE_CHECKING:
    from stream.datatypes import LayerDim
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.solver import LinExpr, SolverVar
    from stream.workload.workload import Tensor, TransferNode


@dataclass(frozen=True)
class MemoryReuseEntry:
    """One staged tensor's residency on a memory tile, held against its reader's."""

    name: str
    memory: Memory
    mem_level: LinExpr
    compute_level: LinExpr
    unexpressible: tuple[tuple[SolverVar, SolverVar], ...]


def replay_unexpressible_levels(relevancies: list[bool], read_levels: int) -> list[tuple[int, int]]:
    """(memory stop, reader stop) level pairs no single whole-object replay realises."""
    pairs: list[tuple[int, int]] = []
    for s_m in range(len(relevancies)):
        top_relevant = max((i for i in range(s_m + 1) if relevancies[i]), default=-1)
        for s_c in range(-1, min(s_m, read_levels)):
            inside = any(not relevancies[i] and i < top_relevant for i in range(s_c + 1, s_m + 1))
            if inside or s_m != len(relevancies) - 1:
                pairs.append((s_m, s_c))
    return pairs


class ReuseRates:
    """The reuse factor of each copy a transfer lands, how many iterations one of its firings serves, from the copy's
    reuse stop; a transfer's own is its first copy's."""

    name: ClassVar[str] = "reuse_rates"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("reuse_factor", "copy_reuse_factor")

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
        for tr in space.transfer_nodes:
            assert len(tr.inputs) == 1, (
                f"Only single-input transfers are supported for fire rate constraints, "
                f"but {tr.name} has inputs {tr.inputs}."
            )
            for k, t in enumerate(tr.outputs):
                name = tr.name if k == 0 else f"{tr.name}_{t.name}"
                reuse_factor = model.add_var(vtype=SolverVarType.INTEGER, name=f"reuse_factor_{name}")
                model.add_constr(
                    reuse_factor
                    == model.quicksum(space.reuse_levels[(t, s)] * z_stop[(t, s)]._raw for s in space.stops(t)),
                    name=f"reuse_factor_def_{name}",
                )
                q.add("copy_reuse_factor", reuse_factor._raw, index=t)
                if k == 0:
                    q.add("reuse_factor", reuse_factor._raw, index=tr)


FUSED_INTERMEDIATE = StructuralRule(
    "Fused intermediate must stay resident",
    "A fused group's intermediate is pinned on-chip and re-read rather than spilled to HBM (the AIE "
    "code-gen cannot stream partial results out and back), which fixes its reuse level across the fused loop.",
)
RESIDENT_OUTPUT = StructuralRule(
    "Output must stay resident",
    "An operator's output is pinned resident and reused in place (no partial spill to HBM), which fixes its reuse "
    "level.",
)


def _held_from(
    ctx: FormulationContext, t: Tensor, first: int, levels: int, name: str, rule: StructuralRule | None = None
) -> None:
    """Hold ``t`` at least up to level ``first`` of its ``levels``."""
    ctx.add_constr(
        ctx.model.quicksum(ctx.vars.z_stop[(t, s)]._raw for s in range(first, levels)) >= 1,
        name=name,
        rule=rule,
    )


class ReuseLevels:
    """A tensor sent from core to core is held up to its outermost irrelevant loop.
    A route with a local copy needs the larger copy kept (:func:`_local_copy_sources_held`)."""

    name: ClassVar[str] = "reuse_levels"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext) -> None:
        for tr in ctx.space.transfer_nodes:
            _local_copy_sources_held(ctx, tr)
        for tr in ctx.space.transfer_nodes:
            if tr.transfer_type not in (TransferType.COMPUTE_TO_COMPUTE):
                continue
            relevancies = ctx.space.ssis[tr].get_applicable_temporal_relevancies()
            last_irrelevant = -1
            for t in tr.outputs:
                for i, r in enumerate(relevancies):
                    if r is False:
                        last_irrelevant = i
                if last_irrelevant >= 0:
                    name = f"force_intermediate_reuse_{tr.name}"
                    _held_from(ctx, t, last_irrelevant, len(relevancies), name, FUSED_INTERMEDIATE)


def _local_copy_sources_held(ctx: FormulationContext, tr: TransferNode) -> None:
    """A route of ``tr`` with a local copy is chosen only if the larger copy is kept at least from the outermost loop
    its reader skips and the smaller copy's reader runs."""
    space, z_stop = ctx.space, ctx.vars.z_stop
    for k, choice in enumerate(space.path_choices[tr]):
        local_copies = {(item.copied_from, item.copies[0]) for item in choice.demand if item.copied_from is not None}
        for larger, smaller in sorted(local_copies):
            held = tr.outputs[larger]
            skipped = _skipped_loops(ctx, tr, larger, smaller)
            loops = space.ssis[held].get_applicable_temporal_variables()
            if (level := max((i for i, v in enumerate(loops) if v.dimension in skipped), default=-1)) < 0:
                continue
            ctx.add_constr(
                ctx.vars.y[(tr, choice)]
                <= ctx.model.quicksum(z_stop[(held, s)]._raw for s in range(level, len(loops))),
                name=f"local_copy_source_held_{tr.name}_{k}_{larger}_{smaller}",
            )


def _skipped_loops(ctx: FormulationContext, tr: TransferNode, larger: int, smaller: int) -> set[LayerDim]:
    """The temporal loops the reader of copy ``larger`` skips while that of copy ``smaller`` runs on them."""
    space = ctx.space
    ssis = [
        space.ssis.get(next(n for n in space.workload.successors(tr) if tr.outputs[c] in n.inputs))
        for c in (larger, smaller)
    ]
    if ssis[0] is None or ssis[1] is None:
        return set()
    absent = {v.dimension for v in ssis[0].get_temporal_variables() if v.effect is LoopEffect.ABSENT}
    return {v.dimension for v in ssis[1].get_temporal_variables() if v.effect is not LoopEffect.ABSENT} & absent


class OutputReuse:
    """A final output is held up to its outermost irrelevant loop."""

    name: ClassVar[str] = "output_reuse"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext) -> None:
        for tr in ctx.space.transfer_nodes:
            if tr.transfer_type not in (TransferType.COMPUTE_TO_MEM,):
                continue
            for t in tr.inputs:
                relevancies = ctx.space.ssis[t].get_applicable_temporal_relevancies()
                last_irrelevant = -1
                for i, r in enumerate(relevancies):
                    if r is False:
                        last_irrelevant = i
                if last_irrelevant >= 0:
                    name = f"force_output_reuse_{tr.name}"
                    _held_from(ctx, t, last_irrelevant, len(relevancies), name, RESIDENT_OUTPUT)


class ReuseCompatibility:
    """The reuse levels on either side of a transfer between a memory and a compute tile agree: on the way in the
    memory tile holds the tensor at least as long as its reader, which a namespace family checks it can replay; on
    the way out they are equal, as a partial output cannot go to a memory tile and come back."""

    name: ClassVar[str] = "reuse_compatibility"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("memory_reuse",)

    def build(self, ctx: FormulationContext) -> None:
        q = ctx.quantities
        space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
        memory_reuse: list[MemoryReuseEntry] = []
        namespace_memories = list({m.namespace: m for m in space.mem_memories}.values())
        for tr in space.transfer_nodes:
            inputs = tr.inputs
            outputs = tr.outputs
            if tr.transfer_type in (TransferType.COMPUTE_TO_MEM,):
                assert len(outputs) == 1, "Expected exactly one output tensor for COMPUTE_TO_MEM transfer."
                output_tensor = outputs[0]
                for input_tensor in inputs:
                    out_levels = len(space.ssis[output_tensor].get_applicable_temporal_sizes())
                    in_levels = len(space.ssis[input_tensor].get_applicable_temporal_sizes())
                    shared = min(out_levels, in_levels)
                    for s in range(-1, shared - 1):
                        model.add_constr(
                            z_stop[(output_tensor, s)] == z_stop[(input_tensor, s)],
                            name=f"reuse_eq_input_{tr.name}_L{s}",
                        )
                    model.add_constr(
                        model.quicksum(z_stop[(output_tensor, s)]._raw for s in range(shared - 1, out_levels))
                        == model.quicksum(z_stop[(input_tensor, s)]._raw for s in range(shared - 1, in_levels)),
                        name=f"reuse_eq_input_{tr.name}_L{shared - 1}",
                    )
            elif tr.transfer_type in (TransferType.MEM_TO_COMPUTE,):
                assert len(inputs) == 1, "Expected exactly one input tensor for MEM_TO_COMPUTE transfer."
                input_tensor = inputs[0]
                for output_tensor in outputs:
                    model.add_constr(
                        _reuse_level_expr(ctx, input_tensor) >= _reuse_level_expr(ctx, output_tensor),
                        name=f"reuse_ge_output_{tr.name}",
                    )
                    unexpressible = _replay_unexpressible_pairs(ctx, input_tensor, output_tensor)
                    memory_reuse.extend(
                        MemoryReuseEntry(
                            tr.name,
                            memory,
                            _reuse_level_expr(ctx, input_tensor),
                            _reuse_level_expr(ctx, output_tensor),
                            unexpressible,
                        )
                        for memory in namespace_memories
                    )
        q.add("memory_reuse", tuple(memory_reuse))


def _reuse_level_expr(ctx: FormulationContext, t: Tensor) -> Any:
    """The chosen reuse level of ``t`` as a linear expression."""
    z_stop = ctx.vars.z_stop
    return ctx.model.quicksum(s * z_stop[(t, s)]._raw for s in ctx.space.stops(t))


def _replay_unexpressible_pairs(
    ctx: FormulationContext, staged: Tensor, read: Tensor
) -> tuple[tuple[SolverVar, SolverVar], ...]:
    """The unexpressible (staged, read) stop pairs, as their z_stop variables."""
    ssis, z_stop = ctx.space.ssis, ctx.vars.z_stop
    relevancies = [v.relevant for v in ssis[staged].get_applicable_temporal_variables()]
    read_levels = len(ssis[read].get_applicable_temporal_variables())
    levels = replay_unexpressible_levels(relevancies, read_levels)
    return tuple((z_stop[(staged, s_m)], z_stop[(read, s_c)]) for s_m, s_c in levels)


class SpatialReuse:
    """Reuse covers every temporal loop inside (or at) a tensor's outermost spatial loop; the temporal loops outside
    it need no buffering for spatial coverage."""

    name: ClassVar[str] = "spatial_reuse"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext) -> None:
        for t in ctx.space.tensors_to_optimize_reuse_for:
            variables = ctx.space.ssis[t].variables
            applicable_temporal = ctx.space.ssis[t].get_applicable_temporal_variables()
            outermost_spatial_pos = -1
            for pos, var in enumerate(variables):
                if var.type == IterationVariableType.SPATIAL:
                    outermost_spatial_pos = pos
            if outermost_spatial_pos < 0:
                continue
            min_reuse_level = -1
            for i, tv in enumerate(applicable_temporal):
                pos = next(p for p, v in enumerate(variables) if v is tv)
                if pos <= outermost_spatial_pos:
                    min_reuse_level = i
                else:
                    break
            if min_reuse_level < 0:
                continue
            _held_from(ctx, t, min_reuse_level, len(applicable_temporal), f"force_reuse_past_spatial_{t.name}")
