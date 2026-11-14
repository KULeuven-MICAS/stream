"""A shared-bandwidth memory bounds the step by its transfers together at its ceiling and by each direction's at its
one-way rate, which is lower: a read-only step takes as long as its reads at the one-way rate."""

from __future__ import annotations

from types import SimpleNamespace

from stream.opt.allocation.constraint_optimization.families import overlap


class _Var(float):
    @property
    def _raw(self) -> float:
        return float(self)


class _Quantities:
    def add(self, *args, **kwargs) -> None:
        pass


def test_reads_alone_are_bounded_at_the_one_way_rate(monkeypatch):
    dram = type("Memory", (), {"name": "dram"})()
    added = []
    bits = {"x": 252.5 * 1000, "w": 252.5 * 2000}
    monkeypatch.setattr(
        overlap,
        "_active_shared_latency",
        lambda ctx, memory, tr, choice, y, rate=None: SimpleNamespace(_raw=bits[tr] / (rate or 336.6)),
    )
    space = SimpleNamespace(
        shared_bandwidth={dram: SimpleNamespace(ceiling=336.6, contiguous=252.5)},
        direction=lambda memory, choice: "read",
        choice_shares_memory=lambda tr, choice: False,
    )
    model = SimpleNamespace(
        add_var=lambda **kwargs: _Var(0.0),
        add_constr=lambda constraint, name: added.append((name, constraint)),
        quicksum=sum,
    )
    ctx = SimpleNamespace(
        space=space, model=model, quantities=_Quantities(), vars=SimpleNamespace(y={("x", 0): 1, ("w", 1): 1})
    )
    overlap._shared_bandwidth_bounds(ctx, overlap=0.0, iteration=2500.0)
    assert dict(added)["shared_read_bound_dram"] is False
    assert "shared_write_bound_dram" not in dict(added)


def test_a_firing_read_in_adjacent_runs_takes_the_adjacent_rate():
    """A 1 KB row read as 128 B runs by eight links as adjacent runs gets their rate, not the
    contiguous rate of the row."""
    from stream.cost_model.bandwidth import BandwidthModel
    from stream.opt.allocation.constraint_optimization.space import DecisionSpace

    dram = type("Memory", (), {"name": "dram"})()
    model = BandwidthModel.from_description(
        {"ceiling": 336.6, "contiguous": 252.5, "strided": {"read": {512: 138.6}}, "adjacent": {"read": {128: 126.25}}}
    )
    space = SimpleNamespace(
        direction=lambda memory, path: "read",
        layouts=SimpleNamespace(
            runs=lambda tr, path: (1024.0, 1024.0), adjacent_runs=lambda tr, path, memory, d: 128.0
        ),
        moved_bits=lambda tr, path, copies=None: 252500.0,
        memory_share=lambda tr, path, memory, leaving, copies=None: 1.0,
        shared_bandwidth={dram: model},
    )
    assert DecisionSpace.shared_cycles(space, dram, "w", "route", 252.5) == 2000
