"""A tensor a transfer produces is held only on the cores the transfer's chosen path writes to."""

from __future__ import annotations

from xdsl.dialects.builtin import bf16

from stream.opt.allocation.constraint_optimization.families.routing import destination_coherence
from stream.opt.allocation.constraint_optimization.formulation import DecisionVariables, FormulationContext
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.opt.solver import SolverBackend, SolverParams, SolverVarType, create_solver
from stream.workload.workload import Tensor


class _Node:
    def __init__(self, name: str, outputs=()):
        self.name, self.outputs = name, list(outputs)


class _Tile:
    def __init__(self, core_id: int, name: str):
        self.id, self.name = core_id, name


def _placed(one_tile: bool) -> tuple[str, ...]:
    """Where the key lands when the shim's path writes one memory tile, or two."""
    model = create_solver(SolverBackend.ORTOOLS_GSCIP, "delivered")
    model.set_param(SolverParams.VERBOSITY, 0)
    key = Tensor.create("key", bf16, (64, 64))
    transfer = _Node("Transfer(key)", [key])
    tile_a, tile_b = _Tile(1, "a"), _Tile(7, "b")
    placements = ((tile_a,), (tile_a, tile_b))
    space = object.__new__(DecisionSpace)
    space.tensor_choices = {key: placements}
    space._fixed, space._candidates = frozenset(), {}
    x = {(key, p): model.add_var(vtype=SolverVarType.BINARY, name=f"x{len(p)}") for p in placements}
    model.add_constr(model.quicksum(v._raw for v in x.values()) == 1)
    path = "one" if one_tile else "two"
    y = {(transfer, path): model.add_var(vtype=SolverVarType.BINARY, name="y")}
    model.add_constr(y[(transfer, path)] == 1)
    space.choice_dst_cores = {(transfer, path): {tile_a} if one_tile else {tile_a, tile_b}}
    space.choice_has_empty_path = {(transfer, path): False}
    ctx = FormulationContext(space, DecisionVariables(x, y, {}, {}, {}), model, QuantityRegistry())
    destination_coherence(ctx, transfer, (path,))
    # Spreading the key would let the next transfer read it from more tiles at once.
    model.set_objective(-x[(key, placements[1])]._raw)
    model.optimize()
    chosen = next(p for p in placements if x[(key, p)].X > 0.5)
    return tuple(t.name for t in chosen)


def test_a_tensor_is_not_held_where_its_transfer_never_writes():
    assert _placed(one_tile=True) == ("a",)


def test_a_tensor_may_be_held_on_every_tile_its_transfer_writes():
    assert _placed(one_tile=False) == ("a", "b")
