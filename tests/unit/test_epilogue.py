"""The epilogue: a copy kept along a loop in which its producer still computes fires after the producer's last call
there, so the run's last firing, and every call after it, comes after the last iteration and is not spread over the
iterations."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from stream.opt.allocation.constraint_optimization.families import SLOT_PRESSURE, overlap
from stream.opt.allocation.constraint_optimization.formulation import (
    DecisionVariables,
    FormulationContext,
    ResourceLedger,
)
from stream.opt.allocation.constraint_optimization.quantities import QuantityRegistry
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.opt.solver import SolverBackend, SolverParams, SolverVarType, create_solver
from stream.workload.steady_state.iteration_space import IterationVariable, LoopEffect

VARY, INV, ABS = LoopEffect.VARYING, LoopEffect.INVARIANT, LoopEffect.ABSENT


@dataclass(eq=False)
class _Node:
    """A computation node or a transfer: its whole call or firing, its effect along each loop (innermost first), its
    readers, for a transfer the reuse stop its copy takes, and the cycles its busiest port takes per call or firing. A
    transfer stands for its own copy."""

    name: str
    call: int
    effects: tuple[LoopEffect, ...]
    readers: tuple[_Node, ...] = ()
    stop: int | None = None
    port: int = 0

    @property
    def outputs(self) -> list[_Node]:
        return [self]


def _space(nodes: list[_Node], sizes: tuple[int, ...]) -> DecisionSpace:
    """A decision space over ``nodes`` in dataflow order, each in its own slot, each transfer on one route; the first
    node's producer is a group input."""
    space = object.__new__(DecisionSpace)
    space.ssc_nodes = tuple(n for n in nodes if n.stop is None)
    space.transfer_nodes = tuple(n for n in nodes if n.stop is not None)
    space.iterations = prod(sizes)
    space.ssis = {
        n: SimpleNamespace(
            get_temporal_variables=lambda n=n: [
                IterationVariable(f"L{k}", size, effect)
                for k, (size, effect) in enumerate(zip(sizes, n.effects, strict=True))
            ]
        )
        for n in nodes
    }
    edge = object()
    space.workload = SimpleNamespace(
        successors=lambda n: list(n.readers),
        predecessors=lambda n: [p for p in nodes if n in p.readers] or [edge],
        topological_sort=lambda: list(nodes),
    )
    space.path_choices = {n: ("route",) for n in space.transfer_nodes}
    applicable = {n: [v for v in space.ssis[n].get_temporal_variables() if v.applicable] for n in nodes}
    space.reuse_levels = {
        (n, s): prod(v.size for v in applicable[n][: s + 1] if not v.relevant)
        for n in space.transfer_nodes
        for s in range(-1, len(applicable[n]))
    }
    space.stops = lambda t: range(-1, len(applicable[t]))  # type: ignore[method-assign]
    space.slot_of = {n: k for k, n in enumerate(nodes)}
    space.runtime = lambda n: n.call  # type: ignore[method-assign]
    space.transfer_latency_for_path = lambda n, path: n.call  # type: ignore[method-assign]
    space.choice_shares_memory = lambda n, path: False  # type: ignore[method-assign]
    space.shared_bandwidth = {}
    return space


def _epilogue(nodes: list[_Node], sizes: tuple[int, ...], slots: list[float], drained: dict[str, int]) -> float | None:
    """The solved epilogue of ``nodes`` over loops of ``sizes`` (innermost first), each slot lasting as ``slots`` gives
    and each node drained as ``drained`` gives by name; None where nothing is kept along a loop in which its producer
    still computes."""
    model = create_solver(SolverBackend.ORTOOLS_GSCIP, "epilogue")
    model.set_param(SolverParams.VERBOSITY, 0)
    space = _space(nodes, sizes)
    slot_latency = {}
    for s, cycles in enumerate(slots):
        slot_latency[s] = model.add_var(vtype=SolverVarType.CONTINUOUS, name=f"slot_{s}")
        model.add_constr(slot_latency[s] == cycles)
    y, z_stop = {}, {}
    for n in space.transfer_nodes:
        y[(n, "route")] = route = model.add_var(vtype=SolverVarType.BINARY, name=f"y_{n.name}")
        model.add_constr(route == 1)
        for s in space.stops(n):
            z_stop[(n, s)] = stop = model.add_var(vtype=SolverVarType.BINARY, name=f"z_{n.name}_{s}")
            model.add_constr(stop == (s == n.stop))
    q = QuantityRegistry()
    q.add("iteration", model.quicksum(v._raw for v in slot_latency.values()))
    q.add(SLOT_PRESSURE, 0, index="slots", upper_bound=max(slots, default=0))
    ctx = FormulationContext(space, DecisionVariables({}, y, z_stop, {}, slot_latency), model, q, ResourceLedger())
    with patch.multiple(
        overlap,
        firing_port_cycles=lambda ctx, n, path: float(n.port),
        call_port_cycles=lambda ctx: {n: float(n.port) for n in space.ssc_nodes},
    ):
        epilogue = overlap._epilogue(ctx, {n: drained.get(n.name, 0) for n in nodes})
    if epilogue is None:
        return None
    model.set_objective(epilogue)
    model.optimize()
    return model.value(epilogue)


def _reduction(firing: int = 40, stop: int = 0, after: tuple[_Node, ...] = ()) -> list[_Node]:
    """Gemm1 varies along K (4 trips), 100 a call; Transfer(H1), firing ``firing``, moves the sum it accumulates,
    held at ``stop``: at 0 over the four trips, at -1 not at all."""
    h1 = _Node("Transfer(H1)", firing, (INV,), after, stop=stop)
    return [_Node("Gemm1", 100, (VARY,), (h1,)), h1, *after]


def test_a_call_its_port_holds_longer_than_it_computes_takes_the_port_time():
    """Gemm2 computes in 200, but its busiest port takes 260 for its call's bits: it ends at 100 + 10 + 40 + 260 + 5
    = 415, against 100 + 15 an epilogue of 300."""
    gemm2 = _Node("Gemm2", 200, (ABS,), port=260)
    drained = {"Gemm1": 10, "Transfer(H1)": 10, "Gemm2": 15}
    assert _epilogue(_reduction(after=(gemm2,)), (4,), [100, 0, 0], drained) == pytest.approx(300)


def test_a_node_run_once_after_waits_whatever_the_stop_before_it():
    """Held at -1, H1 fires in every iteration, its slot holding its whole 40: it ends at 140 in the last iteration,
    150 drained. Gemm2, absent from K, still runs once in the run on the whole sum, so its slot holds none of its 200
    and the epilogue all of it: 150 + 200 = 350, against 140 + 10 an epilogue of 200."""
    gemm2 = _Node("Gemm2", 200, (ABS,))
    nodes = _reduction(stop=-1, after=(gemm2,))
    drained = {"Gemm1": 10, "Transfer(H1)": 10, "Gemm2": 10}
    assert _epilogue(nodes, (4,), [100, 40, 0], drained) == pytest.approx(200)


def test_partial_sums_moved_every_trip_end_within_the_last_iteration():
    """Held at -1, H1 fires in every iteration, its slot holding its whole 40: its last firing ends with the last
    iteration, 100 + 40, and adds no epilogue."""
    assert _epilogue(_reduction(stop=-1), (4,), [100, 40], {"Gemm1": 10}) == 0


def test_the_last_of_a_firing_every_few_iterations_is_in_the_epilogue():
    """K inner (2 trips), N outer (3): H1 is held along K and moves on with N, firing every 2 iterations, 3 times; the
    last fires after the last iteration, while its slot spreads the two before over the iterations, 40 / 2 = 20 each:
    100 + 40 - (100 + 20) = 20 after."""
    h1 = _Node("Transfer(H1)", 40, (INV, VARY), stop=1)
    nodes = [_Node("Gemm1", 100, (VARY, VARY), (h1,)), h1]
    assert _epilogue(nodes, (2, 3), [100, 20], {}) == pytest.approx(20)
    assert _space(nodes, (2, 3)).once_per_run_stops(h1, h1) == frozenset()


def test_a_group_input_kept_over_the_iterations_has_no_epilogue():
    """W comes from outside the group and is held along K: it is fetched before the iterations that read it."""
    gemm = _Node("Gemm1", 100, (VARY,))
    w = _Node("Transfer(W)", 40, (INV,), (gemm,), stop=0)
    assert _space([w, gemm], (4,)).epilogue_stops(w, w) == frozenset()
    assert _epilogue([w, gemm], (4,), [0, 100], {}) is None


def test_a_producer_absent_from_the_loop_holds_its_result_from_the_first_trip():
    """Gemm1 is absent from K and runs once; H1 holds its result along K, made before the first trip, so it waits for
    no last trip."""
    h1 = _Node("Transfer(H1)", 40, (INV,), stop=0)
    nodes = [_Node("Gemm1", 100, (ABS,), (h1,)), h1]
    assert _space(nodes, (4,)).epilogue_stops(h1, h1) == frozenset()
