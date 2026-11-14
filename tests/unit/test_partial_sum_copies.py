"""Partial sums carried at their own format through each copy set a route delivers to."""

from __future__ import annotations

from itertools import count
from types import SimpleNamespace

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.workload.demand import DemandItem

FIRST = ((0, 15), (0, 127))
SECOND = ((16, 31), (0, 127))
"""Two 16 x 128 boxes of a bf16 output: 2048 elements, 32768 bits each."""


_IDS = count()


class _Link:
    """A link, hashed by identity as the route's hops key it."""

    bandwidth = 64


class _Transfer:
    """A transfer of a bf16 tensor, hashed by identity."""

    inputs = (SimpleNamespace(operand_type=SimpleNamespace(bitwidth=16)),)


def _memory(name: str, offchip: bool = False) -> Memory:
    """A memory viewed by one core of its own."""
    core = Core.__new__(Core)
    core.id = next(_IDS)
    memory = Memory.__new__(Memory)
    memory.name, memory.offchip, memory.views = name, offchip, ((core, "memory"),)
    return memory


def _route(target: Memory) -> tuple[MulticastPathPlan, _Link]:
    """Two cores' memories m0 and m1 send ``target`` the partial sums of the first box, for copy 0, and m1 sends it
    the finished second box, for copy 1, over one link."""
    m0, m1, link = _memory("m0"), _memory("m1"), _Link()
    items = (
        DemandItem(target=0, sources=(0, 1), box=FIRST, reduce=True, copies=(0,)),
        DemandItem(target=0, sources=(1,), box=SECOND, copies=(1,)),
    )
    plan = MulticastPathPlan(
        sources=(m0, m1),
        targets=(target,),
        total_hops_objective=1,
        links_used=(link,),  # type: ignore[arg-type]
        hops=((m1, target, link),),  # type: ignore[arg-type]
        hop_demand=(items,),
        delivered_items=items,
        demand=items,
    )
    return plan, link


def _space(tr: _Transfer) -> DecisionSpace:
    """A decision space holding the transfer's partial sums at fp32, twice the bf16 it delivers."""
    space = object.__new__(DecisionSpace)
    space.partial_sum_ratio = {tr: 2.0}  # type: ignore[dict-item]
    return space


def test_a_copy_set_carries_its_partial_sums_over_a_link_at_their_format():
    """Copy 0's 2048 partial sums cross the link at fp32, 2048 x 32 = 65536 bits, twice the 32768 bits it is
    delivered; copy 1's finished box crosses at bf16, 2048 x 16 = 32768 bits."""
    tr = _Transfer()
    space = _space(tr)
    plan, link = _route(_memory("m2"))
    assert space.copy_set_link_bits(tr, plan, (0,)) == {link: 2048 * 32}  # type: ignore[arg-type]
    assert space.copy_set_link_bits(tr, plan, (1,)) == {link: 2048 * 16}  # type: ignore[arg-type]
    assert space.link_shares(tr, plan, (0,)) == ((link, 2.0),)  # type: ignore[arg-type]


def test_a_memory_receives_each_copy_set_at_its_own_width():
    """The target receives copy 0's partial sums at twice the bits delivered and copy 1's at once; over both sets,
    (2048 x 2 + 2048) / 4096 = 1.5 times."""
    tr = _Transfer()
    space, target = _space(tr), _memory("m2")
    plan, _ = _route(target)
    shares = [space.memory_share(tr, plan, target, False, copies) for copies in ((0,), (1,), None)]  # type: ignore[arg-type]
    assert shares == [2.0, 1.0, 1.5]
