"""Flash attention: ``mha.cc``'s online softmax and the value accumulation behind it.

Taken online, the key axis of an attention head is a *linear* reduction: the context
accumulates over key blocks exactly the way a GEMM's output accumulates over its
contraction, and the nonlinearity -- a running row maximum and sum -- lives inside the
kernel. So to the tiling machinery these are an ordinary elementwise node and an
ordinary GEMM node, and the key is freely blockable with nothing relaxed.

The running state is ``mha.cc``'s float scale buffer, four ``B_q``-long rows holding
``[m_{i-1} | m_i | l_i | exp2(m_{i-1} - m_i)]``. ``partial_softmax`` writes it and
``matmul_PV``/``rescale_O`` read it, and the two cannot share a core: the probability
block leaves ``partial_softmax`` row major and reaches ``matmul_PV`` in the MAC tiling,
and only a DMA re-lays it out. So the scale crosses one core boundary, as an object fifo
between neighbouring tiles -- buffers and locks in the memory they already share, no DMA
channel. The state itself stays on the softmax core, where it has to: it is read and
written across key blocks, and a fifo hands out a different buffer each time. What
crosses is a copy of it, taken every key block, which is what lets the fifo be two deep
and the two cores run a block apart instead of in lockstep.

Neither the scale buffer nor the block-index buffer is a workload tensor. Both are
kernel artifacts the binding creates, the way :class:`AIEKernelWithZeroing` creates the
zeroing call that belongs to a GEMM rather than to the graph.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, prod
from typing import ClassVar, cast

from snaxc.ir.tsl import TiledStridedLayout
from xdsl.dialects.arith import AddiOp, CmpiOp, ConstantOp, ExtUIOp, IndexCastOp, MuliOp, RemUIOp, TruncFOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    f32,
    i32,
)
from xdsl.dialects.func import CallOp
from xdsl.dialects.memref import StoreOp
from xdsl.dialects.scf import IfOp, IndexSwitchOp, YieldOp
from xdsl.ir import Block, Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl_aie.dialects.aie import (
    BufferOp,
    CoreOp,
    DeviceOp,
    ObjectFifoAcquireOp,
    ObjectFifoOp,
    ObjectFifoPortEnum,
    ObjectFIFOReleaseOp,
    TileOp,
)

from stream.compiler.dialects.stream import (
    ComputationNodeOp,
    StrensorType,
    StrensorVarType,
)
from stream.compiler.kernels.aie_kernel import (
    AIEKernel,
    StateOperand,
    device_of,
    induction_variable,
    tiled_layout,
)
from stream.compiler.kernels.binding import Bindings
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.kernels.softmax import SoftmaxKernel

SCALE_ROWS = 4
INDEX_BITS = 3 * 32
"""The block-index buffer of every core running a flash kernel (:func:`_index_buffer`)."""

# The running scale, carried over the key block ``n`` and indexed by the query block ``m``.
STATE_SCALE = StateOperand("flash_state", SCALE_ROWS, carried_over="n", indexed_by="m", handover=2, element_type=f32)
"""Rows of ``B_q`` the scale buffer holds: m_{i-1}, m_i, l_i and exp2(m_{i-1} - m_i)."""

SNAPSHOT, SNAPSHOT_OBJECT = "passThroughLine", "mha_passThrough.o"
"""The vectorized copy that takes the scale off the softmax core, and its object."""

LOG2E = 1.4453125
"""bf16 log2(e), the factor softmax.cc scales by before exp2.

It is the *whole* factor: a design reaching these kernels hands in a query already
scaled by 1/sqrt(d_head), so scaling here as well would square it.
"""


def _tile(op: Operation) -> TileOp:
    parent = op.parent_op()
    while parent is not None and not isinstance(parent, CoreOp):
        parent = parent.parent_op()
    assert isinstance(parent, CoreOp)
    assert isinstance(parent.tile, OpResult) and isinstance(parent.tile.op, TileOp)
    return parent.tile.op


def _position(tile: TileOp) -> tuple[int, int]:
    return tile.col.value.data, tile.row.value.data


# The score side of a step, whichever way it was mapped: on its own core or fused with
# the GEMM before it. ``matmul_PV`` looks for its partner under either name.
SCORE_SIDE = ("partial_softmax", "partial_softmax_mode", "matmul_softmax")
# mha.cc's bitmask for which side of the step is in a GEMM's tiling.
TILED_IN, TILED_OUT = 1, 2

# Stamped on a core so the relation survives its node being rewritten: per dimension the
# node's output varies or loops along, ``[dim, point, extent]``, where the core sits along it and over how many cores
# the dimension is split (1, point 0, where it is not).
FLASH_POINTS, FLASH_KERNEL = "flash_points", "flash_kernel"


def _spatial_point(op: ComputationNodeOp, dim) -> int | None:
    """Where this instance sits along ``dim``, combining the parts its space splits it into.

    A layer wider than its neighbour carries the split as spatial by temporal, and the
    spatial part runs fastest, so the two recombine the way the runtime descriptor lays
    them out.
    """
    if op.spatial_index is None:
        return None
    index = {var.dim: var.size for var in op.spatial_index.data.vars}
    if dim not in index:
        return None
    return index[dim]


def _split_dim(op: ComputationNodeOp):
    """The dimension this step is handed out over, which is the one both halves share."""
    spatial = [v for v in cast(StrensorType, op.output.type).ssis.data.vars if v.type is StrensorVarType.SPATIAL]
    return spatial[-1].dim if spatial else None


def _spatial_extent(op: ComputationNodeOp, dim) -> int:
    space = cast(StrensorType, op.output.type).ssis.data
    return prod(v.size for v in space.vars if v.type is StrensorVarType.SPATIAL and v.dim == dim) or 1


def _stamp_points(device: DeviceOp) -> None:
    """Record on every core which blocks of each dimension it holds, before any node is rewritten.

    A node carries its spatial index; a call does not, and the two halves of a step are
    rewritten in whatever order the walker reaches them. Stamping once, on the first
    rewrite, leaves the relation readable from either side afterwards.
    """
    cores = [core for core in device.walk() if isinstance(core, CoreOp)]
    if any(FLASH_POINTS in core.attributes for core in cores):
        return
    for core in cores:
        node = next((n for n in core.walk() if isinstance(n, ComputationNodeOp)), None)
        if node is None or node.spatial_index is None:
            continue
        space = cast(StrensorType, node.output.type).ssis.data
        dims = dict.fromkeys(v.dim for v in space.vars if v.type is not StrensorVarType.ABSENT)
        points = [(dim, _spatial_point(node, dim) or 0, _spatial_extent(node, dim)) for dim in dims]
        core.attributes[FLASH_POINTS] = ArrayAttr(
            [ArrayAttr([IntAttr(dim.position), IntAttr(point), IntAttr(extent)]) for dim, point, extent in points]
        )
        core.attributes[FLASH_KERNEL] = StringAttr(node.kernel.data)


def _points(core: CoreOp) -> dict[int, tuple[int, int]]:
    """A stamped core's point and extent along each dimension of its node's output."""
    stamp = cast(ArrayAttr, core.attributes[FLASH_POINTS])
    return {
        dim.data: (point.data, extent.data)
        for dim, point, extent in (cast(ArrayAttr, entry).data for entry in stamp.data)
    }


def _partners(device: DeviceOp, op: ComputationNodeOp, function: str | tuple[str, ...]) -> list[list[TileOp]]:
    """The tiles running the other half of this step, grouped by the blocks they hold, in the order met.

    A step's two halves share the running scale, so they are the cores holding the same
    query blocks -- not the cores that happen to sit next to each other. The blocks are
    compared along the dimensions this half's output has: a value core that also splits the
    head, which the softmax does not have, holds the same query blocks as its partner in the
    other half of the head, so the two are one group and both take every block's scale.
    Where the halves are the same width that is one group each, and neighbours share the
    memory the scale crosses through; where one half is wider it is several, taken in turn.
    """
    _stamp_points(device)
    names = (function,) if isinstance(function, str) else function
    others = [
        (_points(core), core.tile.op)
        for core in device.walk()
        if isinstance(core, CoreOp)
        and FLASH_KERNEL in core.attributes
        and isinstance(core.tile, OpResult)
        and isinstance(core.tile.op, TileOp)
        and any(core.attributes[FLASH_KERNEL].data.startswith(name) for name in names)
    ]
    mine = _tile(op)
    core = next(c for c in device.walk() if isinstance(c, CoreOp) and c.tile.op is mine)
    if not others or FLASH_POINTS not in core.attributes:
        raise ValueError(f"no core runs {function} to share a scale buffer with")
    own = _points(core)
    shared = sorted(dim for dim in own if all(dim in points for points, _ in others))
    narrow = {dim: min([own[dim][1], *(points[dim][1] for points, _ in others)]) for dim in shared}
    groups: dict[tuple[int, ...], list[TileOp]] = {}
    for points, tile in others:
        if all(points[dim][0] % narrow[dim] == own[dim][0] % narrow[dim] for dim in shared):
            groups.setdefault(tuple(points[dim][0] for dim in shared), []).append(tile)
    if not groups:
        raise ValueError(
            f"no core running {function} holds the query blocks of the one at "
            f"{_position(mine)}, so they cannot share a scale buffer"
        )
    return [sorted(groups[key], key=_position) for key in sorted(groups)]


def _source(groups: list[list[TileOp]], tile: TileOp) -> list[TileOp]:
    """The one score core of each group, which a value core takes its scale from."""
    if any(len(group) != 1 for group in groups):
        raise ValueError(f"the value core at {_position(tile)} holds query blocks of more than one score core at once")
    return [group[0] for group in groups]


def _named(device: DeviceOp, kind: type, name: str):
    for candidate in device.walk():
        if isinstance(candidate, kind) and candidate.sym_name.data == name:
            return candidate
    return None


def _scale_name(producer: TileOp, consumer: TileOp) -> str:
    """One fifo per pair, since either half of a step may span several of the other.

    Naming it after the producer alone was enough while a core handed to exactly one
    other; a softmax core feeding two value cores needs one fifo apiece.
    """
    col, row = _position(producer)
    other_col, other_row = _position(consumer)
    return f"flash_scale_{col}_{row}_{other_col}_{other_row}"


def _core_buffer(
    device: DeviceOp, tile: TileOp, kind: str, element_type, size: int, rewriter: PatternRewriter | None = None
) -> SSAValue:
    """A core's own buffer of one kind, made once and found again by name."""
    col, row = _position(tile)
    buffer = _named(device, BufferOp, name := f"flash_{kind}_{col}_{row}")
    if buffer is None:
        assert rewriter is not None
        buffer = BufferOp(tile.result, element_type, ArrayAttr([IntAttr(size)]), StringAttr(name))
        rewriter.insert_op(buffer, InsertPoint.after(tile))
    return buffer.buffer


def _index_buffer(device: DeviceOp, tile: TileOp, rewriter: PatternRewriter | None = None) -> SSAValue:
    """``[kv_block, q_block, row_offset]``, which every mha.cc entry point takes as a pointer."""
    return _core_buffer(device, tile, "index", i32, 3, rewriter)


def _scale_fifo(
    device: DeviceOp,
    rewriter: PatternRewriter,
    producer: TileOp,
    consumer: TileOp,
    element_type,
    size: int,
) -> None:
    """The buffers and locks between the two cores an online-softmax step spans.

    Spelled without a repeat count, which is what keeps it in the shared memory the two
    tiles already have between them rather than on a DMA channel the core cannot spare.
    """
    if _named(device, ObjectFifoOp, _scale_name(producer, consumer)) is not None:
        return
    fifo = ObjectFifoOp.from_referenced_type(
        producer.result,
        [consumer.result],
        _scale_name(producer, consumer),
        STATE_SCALE.handover,
        element_type,
        (size,),
        repeat_count=None,
    )
    block = producer.parent_block()
    assert block is not None
    last = max((producer, consumer), key=block.get_operation_index)
    rewriter.insert_op(fifo, InsertPoint.after(last))


def _turn_var(op: ComputationNodeOp):
    """The variable saying which of the other half's cores this iteration belongs to.

    A dimension split between cores and loops has more than one temporal part, and the turn
    is the innermost of them -- the same one the object fifo switch alternates on, which
    reads the last temporal variable of the space. Taking the outermost instead only looks
    right while the two are the same size, and then picks the wrong loop.
    """
    dim = _split_dim(op)
    space = cast(StrensorType, op.output.type).ssis.data
    return next(v for v in reversed(space.vars) if v.type is StrensorVarType.TEMPORAL and v.dim == dim)


def _hand_scale(op: ComputationNodeOp, groups: Sequence[Sequence[TileOp]], state, width, element_type, rows: int):
    """Snapshot the running scale into the fifo of each value core that owns this block.

    A value side wider than this one holds this core's query blocks on several cores, and
    they come round in the order the blocks do, so the block loop's innermost turn says
    which group this iteration owes; every core of that group, one per share of the head,
    takes the scale. With one group there is nothing to choose between.
    """
    tile = _tile(op)

    def hand(target: TileOp) -> list[Operation]:
        acquire = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Produce.get_int(), 32),
            1,
            _scale_name(tile, target),
            (rows,),
            element_type,
        )
        return [
            acquire,
            # Unconditional: a block this core skipped still owes the one behind it the
            # scale it last wrote, which is what the closing rescale divides by.
            CallOp(SNAPSHOT, [state, acquire.results[0], width], []),
            ObjectFIFOReleaseOp(
                IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Produce.get_int(), 32),
                IntegerAttr.from_int_and_width(1, 32),
                _scale_name(tile, target),
            ),
        ]

    def hand_all(group: Sequence[TileOp]) -> list[Operation]:
        return [operation for target in group for operation in hand(target)]

    if len(groups) == 1:
        return hand_all(groups[0])
    turn = induction_variable(op, _turn_var(op))
    count = ConstantOp(IntegerAttr.from_index_int_value(len(groups)), IndexType())
    which = RemUIOp(turn, count)
    return [
        count,
        which,
        IndexSwitchOp(
            arg=which.result,
            cases=DenseArrayBase.from_list(IntegerType(64), list(range(len(groups)))),
            default_region=Region(Block([*hand_all(groups[0]), YieldOp()])),
            case_regions=[Region(Block([*hand_all(group), YieldOp()])) for group in groups],
            result_types=[],
        ),
    ]


def _block_index(op: ComputationNodeOp, dim) -> tuple[list[Operation], SSAValue, int]:
    """This block's global index along ``dim``, and how many blocks that dimension holds.

    The blocks are handed out over the cores and over the temporal loops, exactly as
    :meth:`SoftmaxKernel.row_offset` hands out rows; counting in blocks rather than in
    elements only means leaving the kernel variable out of the running stride.
    """
    space = cast(StrensorType, op.output.type).ssis.data
    position = {var.dim: var.size for var in (op.spatial_index.data.vars if op.spatial_index else ())}
    # The cores this dimension is spread over run fastest within it, then the loops, however
    # the space happens to list them. That is the order the runtime descriptor lays a split
    # out in, and two layers that split the same dimension by different amounts only agree on
    # which block belongs to which core if both count it this way.
    parts = [v for v in reversed(space.vars) if v.dim == dim and v.type is not StrensorVarType.KERNEL]
    parts.sort(key=lambda v: v.type is not StrensorVarType.SPATIAL)
    ops: list[Operation] = []
    terms: list[SSAValue] = []
    seen: dict[tuple, int] = {}
    constant, stride = 0, 1
    for var in parts:
        if var.type is StrensorVarType.SPATIAL:
            constant += position.get(dim, 0) * stride
        elif var.type is StrensorVarType.TEMPORAL:
            # Equal parts name their loops alike, so they are read innermost first.
            key = (var.type, var.size, var.dim)
            seen[key] = (nth := seen.get(key, 0)) + 1
            ops += [
                index := IndexCastOp(induction_variable(op, var, nth), i32),
                size := ConstantOp.from_int_and_width(stride, i32),
                term := MuliOp(index, size),
            ]
            terms.append(term.result)
        stride *= var.size
    ops.append(offset := ConstantOp.from_int_and_width(constant, i32))
    result = offset.result
    for term in terms:
        ops.append(total := AddiOp(result, term))
        result = total.result
    return ops, result, stride


def _store_index(buffer: SSAValue, key: SSAValue, query: SSAValue) -> list[Operation]:
    """The block indices, the query block starting at its first row."""
    return [
        first := ConstantOp.from_int_and_width(0, IndexType()),
        second := ConstantOp.from_int_and_width(1, IndexType()),
        third := ConstantOp.from_int_and_width(2, IndexType()),
        offset := ConstantOp.from_int_and_width(0, i32),
        StoreOp.get(key, buffer, [first.result]),
        StoreOp.get(query, buffer, [second.result]),
        StoreOp.get(offset.result, buffer, [third.result]),
    ]


@dataclass(kw_only=True)
class CausalGemmKernel(GemmKernel):
    """A GEMM over the score matrix that leaves out the blocks a causal mask would zero.

    Both kernels behind it return before doing anything when the key block's first key sits
    past the query block's last row, so the scores of such a block are written and never
    read: the score GEMM is the one stage of the step that a mask inside the kernels cannot
    skip. The same test around the call skips it here, in positions, since the two blocks
    need not be the same size, while the zeroing before it stays unconditional, which is
    what leaves the block defined whichever way the test goes.
    """

    @property
    def unique_name(self) -> str:
        return f"{super().unique_name}_causal"

    def work_share(self, index: int, width: int, steps: int) -> float:
        blocks = range(index, width * steps, width)
        attended = sum(ceil((block + 1) * self.m / self.n) for block in blocks)
        key_blocks = ceil(width * steps * self.m / self.n)
        return attended / (width * steps * key_blocks)

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        query, key = self.output_axes(op)
        key_ops, key_block, _ = _block_index(op, key)
        query_ops, query_block, _ = _block_index(op, query)
        return [
            *key_ops,
            *query_ops,
            keys := ConstantOp.from_int_and_width(self.n, i32),
            first_key := MuliOp(key_block, keys),
            rows := ConstantOp.from_int_and_width(self.m, i32),
            first_row := MuliOp(query_block, rows),
            span := ConstantOp.from_int_and_width(self.m - 1, i32),
            last_row := AddiOp(first_row, span),
            attends := CmpiOp(first_key, last_row, "sle"),
            IfOp(attends, [], Region(Block([*GemmKernel.function_call(self, op), YieldOp()]))),
        ]


@dataclass(kw_only=True)
class PartialSoftmaxKernel(SoftmaxKernel):
    """One online-softmax step over an m x n block of the score matrix.

    Same row-wise shape as the plain softmax -- rows contiguous, the block's own width
    reduced -- but the row is now a slice of the key rather than the whole of it, so the
    running maximum and sum carry across the blocks in the scale buffer instead of
    finishing inside one call. Masking is the kernel's own business: it reads the block
    indices and drops everything a query may not attend, which is why this kernel takes
    no separate causal entry point: ``causal`` says which mask the kernel is built with.
    """

    causal: bool = True

    # Which side of the step arrives or leaves in the tiling a GEMM works in. Neither is
    # the plain kernel; either one takes the mode-selectable entry point.
    tiled_in: bool = False
    tiled_out: bool = False
    FALLBACK: ClassVar[Mapping[str, tuple[str, str | None]]] = {SNAPSHOT: (SNAPSHOT, SNAPSHOT_OBJECT)}
    SCRATCH_BITS: ClassVar[int] = INDEX_BITS

    @property
    def function_name(self) -> str:
        return "partial_softmax_mode" if self.tiled_in or self.tiled_out else "partial_softmax"

    @property
    def unique_name(self) -> str:
        return f"{self.function_name}_{int(self.tiled_in)}{int(self.tiled_out)}_{self.m}_{self.n}"

    def binding_options(self) -> dict[str, bool]:
        return {"causal": self.causal}

    @property
    def mode(self) -> int:
        return (TILED_IN if self.tiled_in else 0) | (TILED_OUT if self.tiled_out else 0)

    def operand_layouts(self) -> Sequence[TiledStridedLayout]:
        """Row major in; out either row major or in the value accumulation's own tiling.

        Leaving the block MAC tiled lets it cross straight to the core that accumulates it,
        with no memory tile to re-lay it out -- which is what a softmax layer wider than the
        layer it feeds needs, because that handover is a join between cores. It costs a
        scatter per group of rows on the way out, so a layer that can afford the memory tile
        writes row major instead.
        """
        mac = tiled_layout(self.m, self.n, self.mac["m"], self.mac["n"])
        return [mac if self.tiled_in else self._row_major(), mac if self.tiled_out else self._row_major()]

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter, bindings: Bindings) -> None:
        device, tile = device_of(op), _tile(op)
        _index_buffer(device, tile, rewriter)
        self._state_buffer(device, tile, rewriter)
        for group in _partners(device, op, "matmul_PV"):
            for target in group:
                _scale_fifo(device, rewriter, tile, target, f32, SCALE_ROWS * self.m)
        AIEKernel.rewrite(self, op, rewriter, bindings)

    def _state_buffer(self, device: DeviceOp, tile: TileOp, rewriter: PatternRewriter | None = None) -> SSAValue:
        return _core_buffer(device, tile, "state", f32, SCALE_ROWS * self.m, rewriter)

    def state_operands(self) -> Sequence[StateOperand]:
        """The online softmax's running maximum and sum, carried from one key block to the next.

        The index buffer beside it is not a recurrence: both entries are written on every
        call and read within it, so nothing crosses an iteration.
        """
        return [STATE_SCALE]

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        device, tile = device_of(op), _tile(op)
        query, key = self.output_axes(op)
        key_ops, key_block, key_blocks = _block_index(op, key)
        query_ops, query_block, query_blocks = _block_index(op, query)
        state = self._state_buffer(device, tile)
        ops: list[Operation] = [*key_ops, *query_ops]
        ops += _store_index(index := _index_buffer(device, tile), key_block, query_block)
        ops += [
            rows := ConstantOp.from_int_and_width(self.m, i32),
            zero := ConstantOp.from_int_and_width(0, i32),
            opening := CmpiOp(key_block, zero, "eq"),
            IfOp(
                opening,
                [],
                Region(Block([CallOp("init_scale_buffer", [state, rows.result], []), YieldOp()])),
            ),
            # xDSL has no printer for a bf16 literal, so it is narrowed on the core.
            log2e := ConstantOp(FloatAttr(LOG2E, f32)),
            scaling := TruncFOp(log2e, self.element_type),
            columns := ConstantOp.from_int_and_width(self.n, i32),
            queries := ConstantOp.from_int_and_width(query_blocks * self.m, i32),
            keys := ConstantOp.from_int_and_width(key_blocks * self.n, i32),
            tiling := ConstantOp.from_int_and_width(self.mode, i32),
            CallOp(
                self.function_name,
                [
                    op.inputs[0],
                    op.inputs[1],
                    state,
                    index,
                    scaling.result,
                    rows.result,
                    columns.result,
                    queries.result,
                    keys.result,
                    *([tiling.result] if self.mode else []),
                ],
                [],
            ),
            width := ConstantOp.from_int_and_width(SCALE_ROWS * self.m, i32),
        ]
        ops += _hand_scale(
            op,
            _partners(device, op, "matmul_PV"),
            state,
            width.result,
            f32,
            SCALE_ROWS * self.m,
        )
        return ops


@dataclass(kw_only=True)
class FusedScoreSoftmaxKernel(GemmKernel):
    """A step's score side whole: the GEMM and the online softmax on one core.

    The softmax reads and writes the block a row at a time out of the GEMM's own MAC
    tiling, so nothing re-lays it out: the two share a core, and the block reaches
    ``matmul_PV`` as its GEMM operand without a memory tile in between. That frees a row
    of the array and the ports the swizzle used to cost. The running state, and the scale
    that crosses to ``matmul_PV``, are the same as when the softmax stands on its own.
    """

    FALLBACK: ClassVar[Mapping[str, tuple[str, str | None]]] = {
        **GemmKernel.FALLBACK,
        SNAPSHOT: (SNAPSHOT, SNAPSHOT_OBJECT),
    }
    SCRATCH_BITS: ClassVar[int] = INDEX_BITS

    @property
    def function_name(self) -> str:
        return "matmul_softmax"

    def _state_buffer(self, device: DeviceOp, tile: TileOp, rewriter: PatternRewriter | None = None) -> SSAValue:
        return _core_buffer(device, tile, "state", f32, SCALE_ROWS * self.m, rewriter)

    def state_operands(self) -> Sequence[StateOperand]:
        """The online softmax's running maximum and sum, carried from one key block to the next.

        The index buffer beside it is not a recurrence: both entries are written on every
        call and read within it, so nothing crosses an iteration.
        """
        return [STATE_SCALE]

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter, bindings: Bindings) -> None:
        device, tile = device_of(op), _tile(op)
        _index_buffer(device, tile, rewriter)
        self._state_buffer(device, tile, rewriter)
        for group in _partners(device, op, "matmul_PV"):
            for target in group:
                _scale_fifo(device, rewriter, tile, target, f32, SCALE_ROWS * self.m)
        GemmKernel.rewrite(self, op, rewriter, bindings)

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        device, tile = device_of(op), _tile(op)
        query, key = self.output_axes(op)
        key_ops, key_block, key_blocks = _block_index(op, key)
        query_ops, query_block, query_blocks = _block_index(op, query)
        state = self._state_buffer(device, tile)
        ops: list[Operation] = [*key_ops, *query_ops]
        ops += _store_index(index := _index_buffer(device, tile), key_block, query_block)
        ops += [
            rows := ConstantOp.from_int_and_width(self.m, i32),
            zero := ConstantOp.from_int_and_width(0, i32),
            opening := CmpiOp(key_block, zero, "eq"),
            IfOp(
                opening,
                [],
                Region(Block([CallOp("init_scale_buffer", [state, rows.result], []), YieldOp()])),
            ),
            log2e := ConstantOp(FloatAttr(LOG2E, f32)),
            scaling := TruncFOp(log2e, self.element_type),
            columns := ConstantOp.from_int_and_width(self.n, i32),
            queries := ConstantOp.from_int_and_width(query_blocks * self.m, i32),
            keys := ConstantOp.from_int_and_width(key_blocks * self.n, i32),
            CallOp(
                self.function_name,
                [
                    op.inputs[0],
                    op.inputs[1],
                    op.inputs[2],
                    state,
                    index,
                    scaling.result,
                    rows.result,
                    columns.result,
                    queries.result,
                    keys.result,
                ],
                [],
            ),
            width := ConstantOp.from_int_and_width(SCALE_ROWS * self.m, i32),
        ]
        ops += _hand_scale(
            op,
            _partners(device, op, "matmul_PV"),
            state,
            width.result,
            f32,
            SCALE_ROWS * self.m,
        )
        return ops


@dataclass(kw_only=True)
class FlashKernel(GemmKernel):
    """The value half of an online-softmax step: ``O += P V``, rescaled as the row max moves.

    A GEMM over the key with the probability block as its A operand, plus the two things
    the running state buys: the context is scaled by ``exp2(m_{i-1} - m_i)`` before every
    block after the first, and divided by the final row sum after the last one. The key
    is the contraction, so the context stays on the compute tile across the key loop by
    the same reuse the backend already gives a GEMM's output. It accumulates in a float
    buffer of the core's own, which the closing rescale writes into the output. ``causal``
    is the mask the kernel is built with, which skips the blocks the softmax beside it skips.
    """

    ACCUMULATOR: ClassVar[str | None] = "fp32"
    SCRATCH_BITS: ClassVar[int] = INDEX_BITS
    causal: bool = True
    precision: str = "high"

    def __post_init__(self) -> None:
        if self.precision != "high":
            raise ValueError(f"matmul_PV multiplies in bf16 (precision 'high'), not {self.precision!r}")

    @property
    def library_name(self) -> str:
        return self.function_name

    def binding_options(self) -> dict[str, bool]:
        return {"causal": self.causal}

    @property
    def function_name(self) -> str:
        return "matmul_PV"

    def initialize(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> list[CallOp]:
        """Nothing: ``matmul_PV`` zeroes the accumulator on the first key block, and ``rescale_O`` writes every row."""
        return []

    def _accumulator(self, device: DeviceOp, tile: TileOp, rewriter: PatternRewriter | None = None) -> SSAValue:
        return _core_buffer(device, tile, "accumulator", f32, self.m * self.n, rewriter)

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter, bindings: Bindings) -> None:
        device, tile = device_of(op), _tile(op)
        _index_buffer(device, tile, rewriter)
        self._accumulator(device, tile, rewriter)
        for source in _source(_partners(device, op, SCORE_SIDE), tile):
            _scale_fifo(device, rewriter, source, tile, f32, SCALE_ROWS * self.m)
        GemmKernel.rewrite(self, op, rewriter, bindings)

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        self.check_operands(op)
        device, tile = device_of(op), _tile(op)
        sources = _source(_partners(device, op, SCORE_SIDE), tile)
        kernel_dims = self.output_axes(op)
        space = cast(StrensorType, op.output.type).ssis.data
        indexed = {var.dim for var in space.get_kernel_variables()}
        reduced = {var.dim for var in space.vars if var.type is not StrensorVarType.KERNEL} - indexed
        if len(reduced) > 1:
            raise ValueError(f"kernel {self.function_name} accumulates over one dimension, not {sorted(reduced)}")
        key_ops, key_block, key_blocks = _block_index(op, next(iter(reduced), None))
        query_ops, query_block, _ = _block_index(op, kernel_dims[0])
        index = _index_buffer(device, tile)
        ops: list[Operation] = [*key_ops, *query_ops]
        ops += _store_index(index, key_block, query_block)
        if len(sources) == 1:
            return [*ops, *self._step(op, sources[0], key_block, key_blocks, index)]

        # A score side wider than this one holds each of this core's query blocks on a
        # different core, so which scale to take turns with the innermost part of the split
        # -- the same part the block index above counts with, so the turn and the block it
        # is working on stay in step.
        selector = induction_variable(op, _turn_var(op))
        cases = [
            Region(Block([*self._step(op, source, key_block, key_blocks, index), YieldOp()])) for source in sources
        ]
        ops.append(
            IndexSwitchOp(
                arg=selector,
                cases=DenseArrayBase.from_list(IntegerType(64), list(range(len(cases)))),
                default_region=Region(Block([*self._step(op, sources[0], key_block, key_blocks, index), YieldOp()])),
                case_regions=cases,
                result_types=[],
            )
        )
        return ops

    def _step(self, op: ComputationNodeOp, source: TileOp, key_block, key_blocks, index) -> list[Operation]:
        """One value accumulation against the scale the given score core wrote."""
        acquire = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Consume.get_int(), 32),
            1,
            _scale_name(source, _tile(op)),
            (SCALE_ROWS * self.m,),
            f32,
        )
        scale = acquire.results[0]
        accumulator = self._accumulator(device_of(op), _tile(op))
        return [
            acquire,
            rows := ConstantOp.from_int_and_width(self.m, i32),
            zero := ConstantOp.from_int_and_width(0, i32),
            # Block zero is never causally skipped, so testing it at run time is the same
            # as the peeled first iteration the kernel was written for.
            opened := CmpiOp(key_block, zero, "ne"),
            carried := ExtUIOp(opened, i32),
            # The keys that exist: the kernel zeroes the value rows of a block past them.
            keys := ConstantOp.from_int_and_width(key_blocks * self.k, i32),
            CallOp(
                self.function_name,
                [op.inputs[0], op.inputs[1], accumulator, scale, rows.result, carried.result, index, keys.result],
                [],
            ),
            last := ConstantOp.from_int_and_width(key_blocks - 1, i32),
            closing := CmpiOp(key_block, last, "eq"),
            IfOp(
                closing,
                [],
                Region(
                    Block([CallOp("rescale_O", [accumulator, op.inputs[2], scale, rows.result, index], []), YieldOp()])
                ),
            ),
            ObjectFIFOReleaseOp(
                IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Consume.get_int(), 32),
                IntegerAttr.from_int_and_width(1, 32),
                _scale_name(source, _tile(op)),
            ),
        ]
