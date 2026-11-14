"""A dim tiled at nested sizes stages its larger tile in a memory tile: the copy there holds that block, its transfer
off chip reads it in runs as wide as the block, and the tile search offers the block wide enough to read at the full
rate."""

from __future__ import annotations

from types import SimpleNamespace

from stream.cost_model.bandwidth import BandwidthModel
from stream.datatypes import LayerDim
from stream.hardware.ports import READ
from stream.opt.allocation.constraint_optimization.space import DecisionSpace
from stream.workload.steady_state.iteration_space import IterationVariable, LoopEffect

K, M, N = LayerDim("d0"), LayerDim("d1"), LayerDim("d2")
COPY = "a_memtile"


def _space(kind: str) -> SimpleNamespace:
    """A copy of an activation looped over 8 k tiles within a 512-wide block, 16 such blocks, n and m."""
    loops = [IterationVariable(d, n, LoopEffect.VARYING) for d, n in ((K, 8), (K, 16), (N, 4), (M, 8))]
    space = SimpleNamespace(
        ssis={COPY: SimpleNamespace(get_applicable_temporal_variables=lambda: loops)},
        candidate_memories=lambda t: [SimpleNamespace(kind=kind)],
    )
    space.staged_whole = lambda t: DecisionSpace.staged_whole(space, t)
    return space


def test_a_memory_tile_stages_the_k_tiles_of_a_block():
    assert DecisionSpace.staging_block(_space("memory"), COPY) == (0, {K: 8})


def test_a_core_stages_nothing():
    assert DecisionSpace.staging_block(_space("compute"), COPY) == (-1, {})


def test_the_full_rate_starts_at_twice_the_longest_measured_span():
    """Strix's DRAM reads reach the contiguous rate at 1 KB runs, twice its longest measured 512 B."""
    model = BandwidthModel(ceiling=260.1, contiguous=177.7, strided={READ: {128: 56.4, 512: 135.2}, "write": {}})
    assert model.full_rate_span(READ) == 1024


def test_the_tile_search_offers_a_block_read_at_the_full_rate():
    """A bf16 activation (m, k) contiguous along k, cut in 64-wide k tiles: 128 B rows, so a block of 8 tiles reaches
    the 1 KB the DRAM reads at its full rate, nested inside the 64-wide loop."""
    from stream.mapping.mapping import FusedGroup
    from stream.stages.generation.tile_search import TileSearchStage

    activation = SimpleNamespace(operand_type=SimpleNamespace(bitwidth=16))
    gemm = SimpleNamespace(name="Gemm", inputs=[activation])
    model = BandwidthModel(ceiling=260.1, contiguous=177.7, strided={READ: {128: 56.4, 512: 135.2}, "write": {}})
    stage = SimpleNamespace(
        workload=SimpleNamespace(
            get_computation_nodes=lambda: [gemm],
            strides_for_tensor=lambda t: {K: (0, 1), M: (1, 0)},
            get_dimension_size=lambda d: 8192,
        ),
        ctx=SimpleNamespace(get=lambda key: SimpleNamespace(bandwidth={"dram": model})),
    )
    group = FusedGroup("g", ("Gemm",), ((K, 64), (N, 64), (M, 64)))
    staged = TileSearchStage._staged(stage, group)
    assert staged is not None and staged.intra_core_tiling == ((K, 64), (K, 512), (N, 64), (M, 64))


def test_a_block_of_k_tiles_is_read_across_them():
    """The memory tile holds 8 k tiles of a 64x64 activation tile: the read off chip takes each row across all 8 (512
    elements, 1 KB) where k is the off-chip tensor's contiguous axis and the read lands in a memory tile."""
    import pytest

    convert = pytest.importorskip("stream.compiler.transforms.aie_convert_ofs", reason="the AIE dialects")
    from xdsl.dialects.builtin import StringAttr, bf16

    from stream.compiler.dialects.stream import StrensorSpace, StrensorType, StrensorVar, StrensorVarType

    k_in = StrensorVar(StrensorVarType.TEMPORAL, 8, K)
    rows, columns = StrensorVar(StrensorVarType.KERNEL, 64, M), StrensorVar(StrensorVarType.KERNEL, 64, K)
    space = StrensorSpace((StrensorVar(StrensorVarType.TEMPORAL, 16, K), k_in, rows, columns))
    memory = StrensorType(bf16, space, [StringAttr("tile_0_1")], reuse_index=3)
    pairs = [(v, v) for v in reversed(space.vars)]
    assert convert._widened_run(memory, pairs, {K: 1, M: 8192}, staged=True) is k_in
    assert convert._widened_run(memory, pairs, {K: 8192, M: 1}, staged=True) is None
    assert convert._widened_run(memory, pairs, {K: 1, M: 8192}, staged=False) is None
