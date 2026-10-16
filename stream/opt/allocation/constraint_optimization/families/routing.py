"""Where each tensor lives and which route each transfer takes."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from stream.opt.allocation.constraint_optimization.diagnosis import ResourceKind
from stream.opt.allocation.constraint_optimization.families import ROUTE_HOPS
from stream.opt.allocation.constraint_optimization.utils import resource_key
from stream.opt.solver import ObjectiveLevel, SolverVar, SolverVarType
from stream.workload.workload import Tensor

if TYPE_CHECKING:
    from stream.cost_model.communication_manager import MulticastPathPlan
    from stream.hardware.architecture.core import Core
    from stream.hardware.architecture.noc.communication_link import CommunicationLink
    from stream.opt.allocation.constraint_optimization.formulation import FormulationContext
    from stream.workload.workload import TransferNode


class Placement:
    """Each movable tensor takes exactly one of its placements."""

    name: ClassVar[str] = "placement"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext) -> None:
        model, x = ctx.model, ctx.vars.x
        for t in ctx.space.tensor_var:
            model.add_constr(
                model.quicksum(x[(t, choice)]._raw for choice in ctx.space.tensor_choices[t]) == 1,
                name=f"place_{t.name}",
            )


class PathChoice:
    """Each transfer takes one route, whose ends hold the tensors it moves; the route length is the last objective
    level."""

    name: ClassVar[str] = "path_choice"
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext) -> None:
        colocated: dict[tuple[Tensor, Tensor, Core], SolverVar] = {}
        for tr in ctx.space.transfer_nodes:
            choices = ctx.space.path_choices[tr]
            ctx.model.add_constr(
                ctx.model.quicksum(ctx.vars.y[(tr, choice)]._raw for choice in choices) == 1,
                name=f"one_path_{tr.name}",
            )
            _source_coherence(ctx, tr, choices)
            destination_coherence(ctx, tr, choices)
            _empty_path_coherence(ctx, tr, choices, colocated)

    def objective(self, ctx: FormulationContext) -> list[ObjectiveLevel]:
        y, links = ctx.vars.y, ctx.space.links_in_choice
        hops = ctx.model.quicksum(
            len(links[(tr, choice)]) * y[(tr, choice)]._raw
            for tr in ctx.space.transfer_nodes
            for choice in ctx.space.path_choices[tr]
        )
        return [ObjectiveLevel(expr=hops._raw, priority=ROUTE_HOPS, name="route_hops")]


def _source_coherence(ctx: FormulationContext, tr: TransferNode, choices: tuple[MulticastPathPlan, ...]) -> None:
    src_tensors = tr.inputs
    assert all(isinstance(t, Tensor) for t in src_tensors), f"Transfer {tr.name} has non-tensor input(s): {src_tensors}"
    for src_tensor in src_tensors:
        for i, choice in enumerate(choices):
            y = ctx.vars.y[(tr, choice)]
            for src_core in ctx.space.choice_src_cores[(tr, choice)]:
                ctx.add_constr(
                    y <= ctx.tensor_on_core_expr(src_tensor, src_core),
                    name=f"path_src_match_{tr.name}_{src_tensor.name}_{resource_key(src_core)}_choice_{i}",
                    resource=src_core,
                )


def destination_coherence(ctx: FormulationContext, tr: TransferNode, choices: tuple[MulticastPathPlan, ...]) -> None:
    """A chosen route's targets hold the copies it moves there (each copy lives with the node it reaches), and a
    tensor sits only where its chosen route delivers it."""
    space, model = ctx.space, ctx.model
    for dst_tensor in tr.outputs:
        assert isinstance(dst_tensor, Tensor), f"Expected {dst_tensor} to be a Tensor."
        for i, choice in enumerate(choices):
            y = ctx.vars.y[(tr, choice)]
            for dst_core in space.choice_dst_cores[(tr, choice)] & space.candidate_cores(dst_tensor):
                ctx.add_constr(
                    y <= ctx.tensor_on_core_expr(dst_tensor, dst_core),
                    name=f"path_dst_match_{tr.name}_{dst_tensor.name}_{resource_key(dst_core)}_choice_{i}",
                    resource=dst_core,
                )
        if space.is_fixed(dst_tensor):
            continue
        for core in space.candidate_cores(dst_tensor):
            delivering = [
                ctx.vars.y[(tr, choice)]._raw
                for choice in choices
                if core in space.choice_dst_cores[(tr, choice)] or space.choice_has_empty_path[(tr, choice)]
            ]
            ctx.add_constr(
                ctx.tensor_on_core_expr(dst_tensor, core) <= model.quicksum(delivering),
                name=f"dst_delivered_{tr.name}_{dst_tensor.name}_{resource_key(core)}",
                resource=core,
            )


def _empty_path_coherence(
    ctx: FormulationContext,
    tr: TransferNode,
    choices: tuple[MulticastPathPlan, ...],
    colocated: dict[tuple[Tensor, Tensor, Core], SolverVar],
) -> None:
    """A route of only empty paths needs its source and destination tensors on a common candidate core."""
    if len(tr.inputs) != 1 or len(tr.outputs) == 0:
        return
    space, model = ctx.space, ctx.model
    src_tensor = tr.inputs[0]
    assert isinstance(src_tensor, Tensor)
    for i, choice in enumerate(choices):
        if not space.choice_has_empty_path[(tr, choice)]:
            continue
        if len(choice.links_used) != 0:
            raise ValueError("Something went wrong in empty path determination")
        y = ctx.vars.y[(tr, choice)]
        for dst_tensor in tr.outputs:
            assert isinstance(dst_tensor, Tensor)
            common_cores = space.candidate_cores(src_tensor) & space.candidate_cores(dst_tensor)
            if not common_cores:
                model.add_constr(y == 0, name=f"empty_path_infeasible_{tr.name}_{dst_tensor.name}_choice_{i}")
                continue
            coloc_terms = [_same_core_var(ctx, src_tensor, dst_tensor, core, colocated) for core in common_cores]
            model.add_constr(
                y <= model.quicksum(t._raw for t in coloc_terms),
                name=f"empty_path_match_{tr.name}_{dst_tensor.name}_choice_{i}",
            )


def _same_core_var(
    ctx: FormulationContext,
    src_tensor: Tensor,
    dst_tensor: Tensor,
    core: Core,
    colocated: dict[tuple[Tensor, Tensor, Core], SolverVar],
) -> SolverVar:
    """A binary that is 1 only where both tensors sit on ``core``."""
    key = (src_tensor, dst_tensor, core)
    if key in colocated:
        return colocated[key]
    model = ctx.model
    suffix = f"{src_tensor.name}_{dst_tensor.name}_{resource_key(core)}"
    v = colocated[key] = model.add_var(vtype=SolverVarType.BINARY, name=f"same_{suffix}")
    src_occ = ctx.tensor_on_core_expr(src_tensor, core)
    dst_occ = ctx.tensor_on_core_expr(dst_tensor, core)
    ctx.add_constr(v <= src_occ, name=f"same_src_ub_{suffix}", resource=core)
    ctx.add_constr(v <= dst_occ, name=f"same_dst_ub_{suffix}", resource=core)
    ctx.add_constr(v >= src_occ + dst_occ - 1, name=f"same_lb_{suffix}", resource=core)
    return v


LINK_CONTENTION = ResourceKind("link_contention", "communication link over-subscribed")


class LinkContention:
    """The transfers of a slot share each link: the slot lasts at least as long as they keep any link busy together,
    each for the part of its latency its share of the data keeps that link busy. On circuit-switched links a transfer
    holds the link alone, so a link carries at most one transfer per slot."""

    name: ClassVar[str] = "link_contention"
    requires: ClassVar[tuple[str, ...]] = ("transfer_latency",)
    provides: ClassVar[tuple[str, ...]] = ()

    def build(self, ctx: FormulationContext) -> None:
        space, model = ctx.space, ctx.model
        held: dict[tuple[CommunicationLink, int], list[SolverVar]] = defaultdict(list)
        busy: dict[tuple[CommunicationLink, int], list] = defaultdict(list)
        for (tr, choice), y in ctx.vars.y.items():
            s = space.slot_of[tr]
            latency = ctx.quantities.get("transfer_latency", (tr, choice)).expr
            for link in space.links_in_choice[(tr, choice)]:
                if space.hardware.circuit_switched(link.cores):
                    held[(link, s)].append(y)
            for link, share in space.link_load(tr, choice).items():
                if not space.hardware.circuit_switched(link.cores):
                    busy[(link, s)].append(share * latency)
        for (link, s), vars_ in held.items():
            ctx.add_constr(
                model.quicksum(v._raw for v in vars_) <= 1,
                name=f"link_usage_{resource_key(link)}_{s}",
                resource=link,
                kind=LINK_CONTENTION,
            )
        for (link, s), terms in busy.items():
            if len(terms) > 1:
                ctx.add_constr(
                    ctx.vars.slot_latency[s] >= model.quicksum(terms),
                    name=f"link_busy_{resource_key(link)}_{s}",
                    resource=link,
                    kind=LINK_CONTENTION,
                )
