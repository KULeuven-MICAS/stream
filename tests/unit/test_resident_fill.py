"""The cycles a run waits for the off-chip tensors it holds in a single buffer, which fill before the first
iteration reads them and overlap no iteration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stream.opt.allocation.constraint_optimization.families import overlap
from stream.opt.allocation.constraint_optimization.formulation import (
    DecisionVariables,
    FormulationContext,
    ResourceLedger,
)
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.opt.solver import SolverBackend, SolverParams, SolverVarType, create_solver

TILES = 32
MOVES = 3
# Per tile: (cycles on its own path, cycles of the shared off-chip bandwidth), a strided key
# and a contiguous value as the attention trace moved them.
KEY, VALUE = (1225, 1225), (1024, 442)


class _Node:
    """A tensor or transfer: a name, its outputs, and identity for a key."""

    def __init__(self, name: str, outputs=()):
        self.name, self.outputs = name, list(outputs)


def _fill(*, rotating: bool, single: bool = False) -> float:
    """The solved fill for a key and a value held at reuse level 0, rotated or not, and if rotated
    in one buffer or two. An outer loop of ``MOVES`` moves a rotated window on."""
    model = create_solver(SolverBackend.ORTOOLS_GSCIP, "fill")
    model.set_param(SolverParams.VERBOSITY, 0)
    space = object.__new__(DecisionSpace)
    space.shared_bandwidth = {0: SimpleNamespace(ceiling=1.0)}
    cycles = {}
    y, z_stop, z_single = {}, {}, {}
    space.tensors_to_optimize_reuse_for, space.ssis = [], {}
    space.rotation_levels, space.tiles_needed_levels = {}, {}
    for name, cost in (("key", KEY), ("value", VALUE)):
        tensor = _Node(name)
        transfer = _Node(f"Transfer({name})", [tensor])
        cycles[transfer] = cost
        y[(transfer, "path")] = chosen = model.add_var(vtype=SolverVarType.BINARY, name=f"y_{name}")
        z = model.add_var(vtype=SolverVarType.BINARY, name=f"z_{name}")
        model.add_constr(chosen == 1)
        model.add_constr(z == 1)
        z_stop[(tensor, 0)] = z
        z_stop[(tensor, 1)] = outer = model.add_var(vtype=SolverVarType.BINARY, name=f"z1_{name}")
        model.add_constr(outer == 0)
        space.tensors_to_optimize_reuse_for.append(tensor)
        space.ssis[tensor] = SimpleNamespace(
            get_applicable_temporal_variables=lambda: [None, None],
            get_applicable_temporal_sizes=lambda: [TILES, MOVES],
        )
        space.rotation_levels[(tensor, 0)] = rotating
        space.rotation_levels[(tensor, 1)] = False
        space.tiles_needed_levels[(tensor, 0)] = 1 if single else TILES
        space.tiles_needed_levels[(tensor, 1)] = MOVES * space.tiles_needed_levels[(tensor, 0)]
        if single:
            z_single[(tensor, 0)] = z
    space.is_const_i = lambda transfer: True  # type: ignore[method-assign]
    space.transfer_latency_for_path = lambda transfer, path: cycles[transfer][0]  # type: ignore[method-assign]
    space.shared_cycles = lambda core, transfer, path, rate: cycles[transfer][1]  # type: ignore[method-assign]
    q = QuantityRegistry()
    ctx = FormulationContext(space, DecisionVariables({}, y, z_stop, z_single, {}), model, q, ResourceLedger())
    overlap._resident_fill(ctx, q)
    fill = q.get("fill").expr
    model.set_objective(fill)
    model.optimize()
    return model.value(fill)


def test_a_whole_window_held_in_one_buffer_fills_before_the_run():
    """Each fill takes its own path's time and all share the off-chip bandwidth, so the run
    waits for the bandwidth: both tensors' shares end to end, beyond either fill alone."""
    assert _fill(rotating=False) == pytest.approx(TILES * (KEY[1] + VALUE[1]))


def test_a_rotating_window_is_prefetched_and_costs_no_wait():
    assert _fill(rotating=True) == 0


def test_a_rotating_window_held_in_one_buffer_waits_each_time_it_moves_on():
    """A head's key held once on its core: every next head waits for its fill, on its own path
    and through the shared bandwidth, so the run waits for the bandwidth once a head."""
    assert _fill(rotating=True, single=True) == pytest.approx(MOVES * (KEY[1] + VALUE[1]))
