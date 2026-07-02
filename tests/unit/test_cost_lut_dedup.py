"""CoreCostLUT: O(1) equality-aware lookup, node-level dedup scaling, and version invalidation."""

from __future__ import annotations

import pickle
from pathlib import Path

from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.cost_model.core_cost import CoreCostEntry
from stream.cost_model.core_cost_lut import COST_MODEL_VERSION, CoreCostLUT
from stream.workload.node import ComputationNode
from stream.workload.tensor import Tensor


def _gemm(name: str, m: int, k: int, n: int, op_type: str = "Gemm") -> ComputationNode:
    a = Tensor.create("A", bf16, (m, k))
    b = Tensor.create("B", bf16, (k, n))
    o = Tensor.create("O", bf16, (m, n))
    maps = (
        AffineMap.from_callable(lambda m, k, n: (m, k)),
        AffineMap.from_callable(lambda m, k, n: (k, n)),
        AffineMap.from_callable(lambda m, k, n: (m, n)),
    )
    return ComputationNode(type=op_type, name=name, inputs=(a, b), outputs=(o,), operand_mapping=maps)


class _FakeCore:
    def __init__(self, core_id: int):
        self.id = core_id

    def has_same_performance(self, other) -> bool:
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, _FakeCore) and self.id == other.id


def _entry() -> CoreCostEntry:
    return CoreCostEntry(energy_total=1.0, latency_total=1.0, ideal_cycle=1, ideal_temporal_cycle=1)


def test_equal_node_lookup_is_key_based():
    lut = CoreCostLUT()
    core = _FakeCore(0)
    lut.add_cost(_gemm("first", 8, 4, 16), core, _entry())
    # a structurally identical node (different name) is found; a different one is not
    assert lut.get_equal_node(_gemm("second", 8, 4, 16)) is not None
    assert lut.get_equal_node(_gemm("other", 8, 8, 16)) is None
    # the audited collision: same shapes, different op type must miss
    assert lut.get_equal_node(_gemm("conv", 8, 4, 16, op_type="Conv")) is None


def test_dedup_scales_with_unique_nodes():
    """N identical nodes require one evaluation; distinct shapes require their own."""
    lut = CoreCostLUT()
    core = _FakeCore(0)
    nodes = [_gemm(f"block{i}", 8, 4, 16) for i in range(5)] + [_gemm("unique", 8, 8, 16)]
    evaluated = 0
    for node in nodes:
        if lut.get_equal_node(node) is None:
            lut.add_cost(node, core, _entry())
            evaluated += 1
    assert evaluated == 2  # one for the repeated block, one for the unique node


def test_version_mismatch_invalidates_disk_cache(tmp_path: Path):
    cache = tmp_path / "lut.pickle"
    lut = CoreCostLUT(cache_path=str(cache))
    lut.add_cost(_gemm("g", 8, 4, 16), _FakeCore(0), _entry())
    lut.save()
    assert CoreCostLUT(cache_path=str(cache)).get_nodes()  # loads back with the matching version

    with open(cache, "wb") as fp:
        pickle.dump({"version": COST_MODEL_VERSION + 1, "lut": lut.lut}, fp)
    reloaded = CoreCostLUT(cache_path=str(cache))
    assert reloaded.get_nodes() == []  # stale version ignored
    assert not cache.exists()  # and the stale file is removed
