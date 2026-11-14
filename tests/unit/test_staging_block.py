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


def test_the_tile_search_offers_the_other_operand_held_across_the_inner_loop():
    """A GEMM tiled k, then n, then m holds the activation across n and reads the weight once per m block; swapping n
    and m holds the weight across m and streams the activation, at either k tiling."""
    from stream.mapping.mapping import FusedGroup
    from stream.stages.generation.tile_search import _reordered

    assert _reordered(FusedGroup("g", ("Gemm",), ((K, 64), (N, 64), (M, 64)))).intra_core_tiling == (
        (K, 64),
        (M, 64),
        (N, 64),
    )
    staged = FusedGroup("g", ("Gemm",), ((K, 64), (K, 512), (N, 64), (M, 64)))
    assert _reordered(staged).intra_core_tiling == ((K, 64), (K, 512), (M, 64), (N, 64))
    assert _reordered(FusedGroup("g", ("Silu",), ((N, 64), (M, 1)))) is None


def test_a_block_split_between_cores_is_not_read_across_its_tiles():
    """Where the cores split the contiguous axis too, each one's tiles of the block interleave with the others', so the
    read takes each tile on its own."""
    import pytest

    convert = pytest.importorskip("stream.compiler.transforms.aie_convert_ofs", reason="the AIE dialects")
    from xdsl.dialects.builtin import StringAttr, bf16

    from stream.compiler.dialects.stream import StrensorSpace, StrensorType, StrensorVar, StrensorVarType

    n_in = StrensorVar(StrensorVarType.TEMPORAL, 4, N)
    rows, columns = StrensorVar(StrensorVarType.KERNEL, 64, K), StrensorVar(StrensorVarType.KERNEL, 64, N)
    cores = StrensorVar(StrensorVarType.SPATIAL, 8, N)
    space = StrensorSpace((n_in, cores, rows, columns))
    memory = StrensorType(bf16, space, [StringAttr("tile_0_1")], reuse_index=4)
    pairs = [(v, v) for v in reversed(space.vars)]
    assert convert._widened_run(memory, pairs, {N: 1, K: 8192}, staged=True) is None


def test_the_tile_search_offers_more_output_tiles_held_across_the_reduction():
    """A GEMM tiled k, n, m at 32-row calls also loops m innermost over four calls: each core holds four output tiles
    across the k loop and reuses each weight tile it fetches for all four."""
    from stream.mapping.mapping import FusedGroup
    from stream.stages.generation.tile_search import _rows_held

    group = FusedGroup("g", ("Gemm",), ((K, 64), (N, 64), (M, 32)))
    extent = {K: 2048, N: 2048, M: 512}.__getitem__
    assert _rows_held(group, 4, extent).intra_core_tiling == ((M, 32), (K, 64), (N, 64), (M, 128))
    assert _rows_held(FusedGroup("g", ("Gemm",), ((K, 64), (M, 32), (M, 64))), 2, extent) is None
    assert _rows_held(FusedGroup("g", ("Gemm",), ((K, 64), (M, 1))), 2, extent) is None
    assert _rows_held(FusedGroup("g", ("Gemm",), ((K, 64), (M, 192))), 2, extent) is None


def _search_stage(bounds=None):
    """A tile search over one GEMM group tiled k, n, m at 64, its call bounds by mapping where given."""
    from stream.mapping.mapping import FusedGroup

    group = FusedGroup("g", ("Gemm",), ((K, 64), (N, 64), (M, 64)))

    class _Mapping:
        def __init__(self, groups):
            self.fused_groups = groups

        def with_fused_groups(self, groups):
            return _Mapping(list(groups))

    compute = SimpleNamespace(kind="compute")
    accelerator = SimpleNamespace(
        network=SimpleNamespace(edges=lambda data: [(None, compute, {"cl": SimpleNamespace(bandwidth=64)})]),
        memories={"core": compute},
    )
    return SimpleNamespace(
        workload=SimpleNamespace(get_dimension_size={K: 2048, N: 2048, M: 2048}.__getitem__),
        mapping=_Mapping([group]),
        ctx=SimpleNamespace(get=lambda key: accelerator),
        _staged=lambda g: None,
    )


def test_the_tile_search_offers_the_order_and_the_rows_held_of_a_group():
    from stream.stages.generation.tile_search import TileSearchStage

    stage = _search_stage()
    found = TileSearchStage._alternatives(stage, 0, None, stage.mapping)
    assert [key for key, _ in found] == [
        (0, None, "order"),
        (0, None, "rows"),
        (0, None, "rows"),
        (0, None, "rows order"),
        (0, None, "rows order"),
    ]
    assert [m.fused_groups[0].intra_core_tiling for _, m in found] == [
        ((K, 64), (M, 64), (N, 64)),
        ((M, 64), (K, 64), (N, 64), (M, 128)),
        ((M, 64), (K, 64), (N, 64), (M, 256)),
        ((M, 64), (K, 64), (M, 128), (N, 64)),
        ((M, 64), (K, 64), (M, 256), (N, 64)),
    ]


def test_with_output_tiles_held_the_weight_is_also_held_across_the_rest_of_m():
    """Holding four 32-row output tiles leaves an m loop outermost, so the weight is read once per trip of it; swapping
    that loop with n, past the held rows and the k loops, holds the weight across all of m."""
    from stream.mapping.mapping import FusedGroup
    from stream.stages.generation.tile_search import _reordered

    held = FusedGroup("g", ("Gemm",), ((M, 32), (K, 64), (K, 512), (N, 64), (M, 128)))
    assert _reordered(held, 1).intra_core_tiling == ((M, 32), (K, 64), (K, 512), (M, 128), (N, 64))
    assert _reordered(held) is None


def test_the_tile_search_keeps_the_calls_bound_fastest_and_their_ties(monkeypatch):
    """Of five calls, the three bound fastest and one tied with the third; not one bound slower than the seed's call,
    nor one the cores cannot hold."""
    from stream.stages.generation import tile_search
    from stream.stages.generation.tile_search import TileSearchStage

    stage = _search_stage()
    called = [(name, SimpleNamespace(fused_groups=[name])) for name in "abcdefg"]
    bounds = {"seed": 0.010, "a": 0.004, "b": 0.006, "c": 0.008, "d": 0.008, "e": 0.009, "f": 0.020, "g": float("inf")}
    monkeypatch.setattr(tile_search, "call_bound", lambda w, m, group, width, capacity=0: bounds[group])
    monkeypatch.setattr(tile_search, "build_hardware_facts", lambda a, c: SimpleNamespace(usable_bits=lambda m: 1))
    stage.mapping = SimpleNamespace(fused_groups=["seed"])
    kept = TileSearchStage._fastest_calls(stage, 0, called)
    assert [key for key, _ in kept] == ["a", "b", "c", "d"]


def _visited(dims):
    """The addresses a walk, outer first, visits in order."""
    from itertools import product

    return [
        sum(i * stride for i, (_, stride) in zip(index, dims, strict=True))
        for index in product(*(range(s) for s, _ in dims))
    ]


def _scattered(outer: tuple, object_shape: list[int], local: list | None = None):
    """A memory tile holding 16 k tiles of a 64x32 activation tile within the ``outer`` loops, the widened run off chip
    scattered into its object: the module, its device, its fifo and the convert pass. ``local`` lists the held loops
    innermost first, None for the k tiles; by default the k tiles, then ``outer`` inside out."""
    import pytest

    convert = pytest.importorskip("stream.compiler.transforms.aie_convert_ofs", reason="the AIE dialects")
    from xdsl.dialects.builtin import ModuleOp, StringAttr, bf16
    from xdsl.ir import Block, Region
    from xdsl_aie.dialects.aie import DeviceOp, ObjectFifoOp, RuntimeSequenceOp, TileOp

    from stream.compiler.dialects.stream import StrensorSpace, StrensorType, StrensorVar, StrensorVarType

    k_in = StrensorVar(StrensorVarType.TEMPORAL, 16, K)
    rows, columns = StrensorVar(StrensorVarType.KERNEL, 64, M), StrensorVar(StrensorVarType.KERNEL, 32, K)
    space = StrensorSpace((*outer, k_in, rows, columns))
    memory = StrensorType(bf16, space, [StringAttr("tile_0_1")], reuse_index=len(space.vars))
    shim, mem = TileOp(0, 0), TileOp(0, 1)
    fifo = ObjectFifoOp.from_referenced_type(shim, [mem], "staged", 1, bf16, object_shape)
    sequence = RuntimeSequenceOp(Region(Block()))
    device = DeviceOp(0, [shim, mem, fifo, sequence])
    module = ModuleOp([device])
    held = [k_in if var is None else var for var in (local or [None, *reversed(outer)])]
    convert._scatter_into_tiles(sequence, [StringAttr("staged")], memory, k_in, held)
    return module, device, fifo, convert


def _walk(fifo) -> list[tuple[int, int]]:
    return [tuple(dim) for dim in fifo.dimensionsFromStreamPerConsumer.data[0]]


def test_a_block_held_with_two_more_loops_is_written_by_a_walk_a_memory_tile_steps():
    """Within 2 k blocks and 2 row blocks, the five-dim walk that scatters the widened run into the tiles folds the two
    outer blocks into one dim, visiting the same addresses in the same order."""
    from stream.compiler.dialects.stream import StrensorVar, StrensorVarType

    k_blocks, m_blocks = StrensorVar(StrensorVarType.TEMPORAL, 2, K), StrensorVar(StrensorVarType.TEMPORAL, 2, M)
    module, _, fifo, convert = _scattered((m_blocks, k_blocks), [2, 2, 16, 64, 32])
    scattered = _walk(fifo)
    assert scattered == [(2, 65536), (2, 32768), (64, 32), (16, 2048), (32, 1)]
    convert.fit_memory_tile_walks(module)
    assert len(_walk(fifo)) <= convert.MEMORY_TILE_DIMS
    assert _visited(_walk(fifo)) == _visited(scattered)


def test_a_distributed_block_keeps_its_slot_dim_for_the_link_to_take_off():
    """Sent to 4 cores, 2 row blocks each: the walk steps from slot to slot on its outer dim, which the link takes off
    before the walk is fitted, so the 4 dims left within a slot stand unfolded."""
    from xdsl.dialects.builtin import StringAttr

    from stream.compiler.dialects.stream import StrensorVar, StrensorVarType

    cores, m_blocks = StrensorVar(StrensorVarType.TEMPORAL, 4, M), StrensorVar(StrensorVarType.TEMPORAL, 2, M)
    module, device, fifo, convert = _scattered((cores, m_blocks), [4, 2, 16, 64, 32], [None, m_blocks])
    slot = 2 * 16 * 64 * 32
    assert _walk(fifo)[0] == (4, slot)
    within = _walk(fifo)[1:]
    convert.RealizeLinks.fill_slot_by_slot(device, StringAttr("staged"), 4)
    convert.fit_memory_tile_walks(module)
    assert _walk(fifo) == within


def test_a_walk_no_fold_brings_to_four_dims_is_refused():
    """A memory tile written by five dims of which none continues the one inside it is refused, not handed to
    mlir-aie."""
    import pytest

    convert = pytest.importorskip("stream.compiler.transforms.aie_convert_ofs", reason="the AIE dialects")
    from xdsl.dialects.builtin import ModuleOp, bf16
    from xdsl_aie.dialects.aie import (
        BDDimLayout,
        BDDimLayoutArray,
        BDDimLayoutArrayArray,
        BDDimLayoutArrayArrayAttr,
        DeviceOp,
        ObjectFifoOp,
        TileOp,
    )

    walk = ((2, 3), (3, 7), (5, 23), (7, 131), (11, 1))
    shim, mem = TileOp(0, 0), TileOp(0, 1)
    fifo = ObjectFifoOp.from_referenced_type(shim, [mem], "staged", 1, bf16, [2 * 3 * 5 * 7 * 11])
    fifo.dimensionsFromStreamPerConsumer = BDDimLayoutArrayArrayAttr(
        BDDimLayoutArrayArray((BDDimLayoutArray(tuple(map(BDDimLayout, walk))),))
    )
    module = ModuleOp([DeviceOp(0, [shim, mem, fifo])])
    with pytest.raises(NotImplementedError, match="walk of 5 dimensions"):
        convert.fit_memory_tile_walks(module)
