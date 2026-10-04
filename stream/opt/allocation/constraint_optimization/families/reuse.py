"""How long each tensor stays resident: its reuse stop, what that stop makes a transfer serve, and the stops the
target forces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from stream.opt.allocation.constraint_optimization.context import MemoryReuseEntry
from stream.opt.allocation.constraint_optimization.diagnosis import StructuralRule
from stream.opt.solver import SolverVarType
from stream.workload.node import TransferType
from stream.workload.steady_state.iteration_space import IterationVariableType

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
    from stream.opt.solver import SolverVar
    from stream.workload.workload import Tensor


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
    """The reuse factor of each transfer: how many iterations one firing serves, from its tensor's reuse stop."""

    name: ClassVar[str] = "reuse_rates"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("reuse_factor",)

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
        for tr in space.transfer_nodes:
            assert len(tr.inputs) == 1, (
                f"Only single-input transfers are supported for fire rate constraints, "
                f"but {tr.name} has inputs {tr.inputs}."
            )
            t = tr.outputs[0]
            reuse_factor = model.add_var(vtype=SolverVarType.INTEGER, name=f"reuse_factor_{tr.name}")
            model.add_constr(
                reuse_factor
                == model.quicksum(space.reuse_levels[(t, s)] * z_stop[(t, s)]._raw for s in space.stops(t)),
                name=f"reuse_factor_def_{tr.name}",
            )
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
    """A tensor handed from core to core is held up to its outermost irrelevant loop."""

    name: ClassVar[str] = "reuse_levels"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
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


class OutputReuse:
    """A final output is held up to its outermost irrelevant loop."""

    name: ClassVar[str] = "output_reuse"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
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
    """The reuse levels on either side of a transfer between a memory and a compute tile agree; the residency a
    memory tile keeps beyond its reader is what a namespace family checks it can replay.

    On the way in the memory tile only has to hold the tensor for at least as long as the compute tile reads it, so
    its level bounds the compute level from above: equating them would let the compute tile's capacity decide how
    long the memory tile keeps a tensor, sending the shim offchip for data already on chip. On the way out the
    levels are equal: a partial output cannot be sent to a memory tile and brought back, so the compute tile owns
    it until it is complete and the memory tile inherits exactly that residency."""

    name: ClassVar[str] = "reuse_compatibility"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ("memory_reuse",)

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
        space, model, z_stop = ctx.space, ctx.model, ctx.vars.z_stop
        memory_reuse: list[MemoryReuseEntry] = []
        namespace_cores = list({c.namespace: c for c in space.mem_cores}.values())
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
                    # Way-in reuse: the memory tile need only hold the tensor AT LEAST as long as the compute
                    # reads it (>=, not ==), so a deeper-loop-invariant operand isn't evicted and re-streamed.
                    model.add_constr(
                        _reuse_level_expr(ctx, input_tensor) >= _reuse_level_expr(ctx, output_tensor),
                        name=f"reuse_ge_output_{tr.name}",
                    )
                    # Whether that extra residency is realisable is a target property.
                    # One core per namespace is enough; the rest repeat the constraint.
                    unexpressible = _replay_unexpressible_pairs(ctx, input_tensor, output_tensor)
                    memory_reuse.extend(
                        MemoryReuseEntry(
                            tr.name,
                            core,
                            _reuse_level_expr(ctx, input_tensor),
                            _reuse_level_expr(ctx, output_tensor),
                            unexpressible,
                        )
                        for core in namespace_cores
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

    def build(self, ctx: FormulationContext, q: QuantityRegistry) -> None:
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
