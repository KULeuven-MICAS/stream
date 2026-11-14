"""The cycles a run waits for a node absent from a fused loop, on hand-built chains of nodes."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
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
from stream.workload.steady_state.iteration_space import IterationVariable, LoopEffect

VARY, INV, ABS = LoopEffect.VARYING, LoopEffect.INVARIANT, LoopEffect.ABSENT


@dataclass(eq=False)
class _Node:
    """A computation node or a transfer: its whole call or firing, its calls in the run, its effect along each loop
    (innermost first) and its readers."""

    name: str
    call: int
    calls: int
    effects: tuple[LoopEffect, ...]
    readers: tuple[_Node, ...] = ()
    transfer: bool = False


def _space(nodes: list[_Node], sizes: tuple[int, ...]) -> DecisionSpace:
    """A decision space over ``nodes`` in dataflow order, its loops of ``sizes`` innermost first."""
    iterations = prod(sizes)
    space = object.__new__(DecisionSpace)
    space.ssc_nodes = tuple(n for n in nodes if not n.transfer)
    space.transfer_nodes = tuple(n for n in nodes if n.transfer)
    space.iterations = iterations
    space.ssis = {
        n: SimpleNamespace(
            get_temporal_variables=lambda n=n: [
                IterationVariable(f"L{k}", size, effect)
                for k, (size, effect) in enumerate(zip(sizes, n.effects, strict=True))
            ]
        )
        for n in nodes
    }
    space.workload = SimpleNamespace(
        successors=lambda n: list(n.readers),
        predecessors=lambda n: [p for p in nodes if n in p.readers],
        topological_sort=lambda: list(nodes),
    )
    space.waiting_absent_nodes = frozenset(n for n in space.ssc_nodes if ABS in n.effects)
    space.runtime = lambda n: n.call  # type: ignore[method-assign]
    space.active_runtime = lambda n: n.call * n.calls / iterations  # type: ignore[method-assign]
    space.transfer_latency_for_path = lambda n, path: n.call * (1 if path == "fast" else 10)  # type: ignore[method-assign]
    return space


def _wait(nodes: list[_Node], sizes: tuple[int, ...], iteration: int, overlap_cycles: int, fill: int = 0, **drained):
    """The solved absent wait of ``nodes``, given one iteration's latency, the overlap of two, the ``fill`` and the
    ``drained`` cycles by node name. Each transfer takes the first of two routes, the second ten times slower."""
    model = create_solver(SolverBackend.ORTOOLS_GSCIP, "absent_wait")
    model.set_param(SolverParams.VERBOSITY, 0)
    space = _space(nodes, sizes)
    y, q = {}, QuantityRegistry()
    for n in space.transfer_nodes:
        for route in ("fast", "slow"):
            y[(n, route)] = chosen = model.add_var(vtype=SolverVarType.BINARY, name=f"y_{n.name}_{route}")
            model.add_constr(chosen == (route == "fast"))
            per_iteration = space.transfer_latency_for_path(n, route) * n.calls / space.iterations
            q.add("transfer_latency", per_iteration * chosen._raw, index=(n, route))
    q.add("iteration", iteration)
    q.add("overlap", overlap_cycles)
    ctx = FormulationContext(space, DecisionVariables({}, y, {}, {}, {}), model, q, ResourceLedger())
    wait = overlap._absent_wait(ctx, fill, {n: drained.get(n.name, 0) for n in nodes})
    model.set_objective(wait)
    model.optimize()
    return model.value(wait)


def test_the_fill_counts_once_and_each_node_waits_for_its_own_drain():
    """Gemm1 varies along a K loop of 4 trips, 100 a call; one firing of 40 moves the whole H1; Gemm2, absent from K,
    runs once in 1000 and drains 10: its chain ends at 4 x 100 + 40 + 1000 + 10 = 1450. One iteration is 360 and two
    overlap by 110, so the steady schedule ends at 4 x 360 - 3 x 110 = 1110, plus a fill of 250: a wait of 90. The side
    node ends at 40 + 200, well inside the fill."""
    gemm2 = _Node("Gemm2", 1000, 1, (ABS,))
    h1 = _Node("Transfer(H1)", 40, 1, (INV,), (gemm2,), transfer=True)
    nodes = [_Node("Gemm1", 100, 4, (VARY,), (h1,)), h1, gemm2, _Node("Side", 10, 4, (VARY,))]
    assert _wait(nodes, (4,), 360, 110, fill=250, Gemm2=10, Side=200) == pytest.approx(90)


def test_a_reader_ends_after_its_producers_last_call():
    """X free, Gemm0 makes a tile in 100 a trip, Gemm1 sums the tiles in 10 a trip on another core and Gemm2, absent
    on the loop, takes the sum once in 40. Gemm1's last call ends after Gemm0's last, at 410, so Gemm2 ends at 450,
    though its first call may start after Gemm1's four calls, at 140. One iteration is 100 + 10 + 10 = 120 with an
    overlap of 20: the steady schedule ends at 4 * 120 - 3 * 20 = 420, 30 earlier."""
    gemm2 = _Node("Gemm2", 40, 1, (ABS,))
    gemm1 = _Node("Gemm1", 10, 4, (VARY,), (gemm2,))
    gemm0 = _Node("Gemm0", 100, 4, (VARY,), (gemm1,))
    x = _Node("Transfer(X)", 0, 4, (VARY,), (gemm0,), transfer=True)
    assert _wait([x, gemm0, gemm1, gemm2], (4,), 120, 20) == pytest.approx(30)


def test_a_transfer_reused_along_an_inner_loop_fires_once_over_it():
    """H1 is moved once per K trip (outer, 4) and reused along tokens (inner, 2): the reader absent from K waits for
    H1's firings over the K trips, 1 + 3 = 4, not one per token as well, 1 + 3 x 2 = 7."""
    reader = _Node("Gemm2", 1000, 2, (VARY, ABS))
    h1 = _Node("Transfer(H1)", 10, 4, (INV, VARY), (reader,), transfer=True)
    assert _space([h1, reader], (2, 4)).calls_before_first_read(h1, reader) == 4
