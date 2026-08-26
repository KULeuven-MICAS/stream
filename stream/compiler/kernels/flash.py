"""Flash attention: ``mha.cc``'s online softmax and the value accumulation behind it.

Taken online, the key axis of an attention head is a *linear* reduction: the context
accumulates over key blocks exactly the way a GEMM's output accumulates over its
contraction, and the nonlinearity -- a running row maximum and sum -- lives inside the
kernel. So to the tiling machinery these are an ordinary elementwise node and an
ordinary GEMM node, and the key is freely blockable with nothing relaxed.

The running state is ``mha.cc``'s scale buffer, four ``B_q``-long rows holding
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

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from typing import cast

from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout
from xdsl.dialects.arith import AddiOp, CmpiOp, ConstantOp, ExtUIOp, IndexCastOp, MuliOp, TruncFOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    FloatAttr,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    f32,
    i32,
)
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.memref import StoreOp
from xdsl.dialects.scf import IfOp, IndexSwitchOp, YieldOp
from xdsl.ir import Block, Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl_aie.dialects.aie import (
    BufferOp,
    CoreOp,
    DeviceOp,
    ObjectFifoAcquireOp,
    ObjectFifoOp,
    ObjectFifoPortEnum,
    ObjectFIFOReleaseOp,
    ObjectFIFOSubviewAccessOp,
    TileOp,
)

from stream.compiler.dialects.stream import (
    ComputationNodeOp,
    StrensorType,
    StrensorVarType,
)
from stream.compiler.kernels.aie_kernel import MAC_ROWS_BFP16, AIEKernel, R, T, induction_variable
from stream.compiler.kernels.gemm import GemmKernel
from stream.compiler.kernels.softmax import SoftmaxKernel

FLASH_TILE = 64
"""The one block shape mha.cc is written for: B_q, B_kv and d_head all 64.

``matmul_PV`` reuses the query GEMM's compile-time ``DIM_M``/``DIM_K``/``DIM_N`` for the
probability block, and both it and ``rescale_O`` walk the context block with 64- and
512-element strides spelled out in the source.
"""

SCALE_ROWS = 4
"""Rows of ``B_q`` the scale buffer holds: m_{i-1}, m_i, l_i and exp2(m_{i-1} - m_i)."""

SNAPSHOT, SNAPSHOT_OBJECT = "passThroughLine", "mha_passThrough.o"
"""The vectorized copy that takes the scale off the softmax core, and its object."""

LOG2E = 1.4453125
"""bf16 log2(e), the factor softmax.cc scales by before exp2.

It is the *whole* factor: a design reaching these kernels hands in a query already
scaled by 1/sqrt(d_head), so scaling here as well would square it.
"""


def _device(op: Operation) -> DeviceOp:
    parent = op.parent_op()
    while parent is not None and not isinstance(parent, DeviceOp):
        parent = parent.parent_op()
    assert isinstance(parent, DeviceOp)
    return parent


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

# Stamped on a core so the relation survives its node being rewritten.
FLASH_POINT, FLASH_EXTENT, FLASH_KERNEL = "flash_point", "flash_extent", "flash_kernel"


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
    """Record on every core which query blocks it holds, before any node is rewritten.

    A node carries its spatial index; a call does not, and the two halves of a step are
    rewritten in whatever order the walker reaches them. Stamping once, on the first
    rewrite, leaves the relation readable from either side afterwards.
    """
    cores = [core for core in device.walk() if isinstance(core, CoreOp)]
    if any(FLASH_POINT in core.attributes for core in cores):
        return
    for core in cores:
        node = next((n for n in core.walk() if isinstance(n, ComputationNodeOp)), None)
        if node is None or node.spatial_index is None:
            continue
        dim = _split_dim(node)
        point = _spatial_point(node, dim)
        if dim is None or point is None:
            continue
        core.attributes[FLASH_POINT] = IntegerAttr.from_int_and_width(point, 32)
        core.attributes[FLASH_EXTENT] = IntegerAttr.from_int_and_width(_spatial_extent(node, dim), 32)
        core.attributes[FLASH_KERNEL] = StringAttr(node.kernel.data)


def _partners(device: DeviceOp, op: ComputationNodeOp, function: str | tuple[str, ...]) -> list[TileOp]:
    """The tiles running the other half of this step, in the order this half meets them.

    A step's two halves share the running scale, so they are the cores holding the same
    query blocks -- not the cores that happen to sit next to each other. Where the halves
    are the same width that is one tile each, and neighbours share the memory the scale
    crosses through; where one half is wider it is several, and the scale takes a stream.
    """
    _stamp_points(device)
    names = (function,) if isinstance(function, str) else function
    others = [
        (
            core.attributes[FLASH_POINT].value.data,
            core.attributes[FLASH_EXTENT].value.data,
            core.tile.op,
        )
        for core in device.walk()
        if isinstance(core, CoreOp)
        and FLASH_KERNEL in core.attributes
        and isinstance(core.tile, OpResult)
        and isinstance(core.tile.op, TileOp)
        and any(core.attributes[FLASH_KERNEL].data.startswith(name) for name in names)
    ]
    mine = _tile(op)
    core = next(c for c in device.walk() if isinstance(c, CoreOp) and c.tile.op is mine)
    if not others or FLASH_POINT not in core.attributes:
        raise ValueError(f"no core runs {function} to share a scale buffer with")
    point = core.attributes[FLASH_POINT].value.data
    narrow = min([core.attributes[FLASH_EXTENT].value.data, *(extent for _, extent, _ in others)])
    found = sorted((p, tile) for p, _, tile in others if p % narrow == point % narrow)
    if not found:
        raise ValueError(
            f"no core running {function} holds the query blocks of the one at "
            f"{_position(mine)}, so they cannot share a scale buffer"
        )
    return [tile for _, tile in found]


def _partner(device: DeviceOp, op: ComputationNodeOp, function: str | tuple[str, ...]) -> TileOp:
    """The single tile this step's other half runs on."""
    found = _partners(device, op, function)
    if len(found) != 1:
        raise ValueError(
            f"the core at {_position(_tile(op))} shares its scale buffer with "
            f"{len(found)} cores running {function}, and this half expects one"
        )
    return found[0]


def _named(device: DeviceOp, kind: type, name: str):
    for candidate in device.walk():
        if isinstance(candidate, kind) and candidate.sym_name.data == name:
            return candidate
    return None


def _scale_name(tile: TileOp) -> str:
    col, row = _position(tile)
    return f"flash_scale_{col}_{row}"


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
    """``[kv_block, q_block]``, which every mha.cc entry point takes as a pointer."""
    return _core_buffer(device, tile, "index", i32, 2, rewriter)


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
    if _named(device, ObjectFifoOp, _scale_name(producer)) is not None:
        return
    fifo = ObjectFifoOp.from_referenced_type(
        producer.result,
        [consumer.result],
        _scale_name(producer),
        2,
        element_type,
        (size,),
        repeat_count=None,
    )
    block = producer.parent_block()
    assert block is not None
    last = max((producer, consumer), key=block.get_operation_index)
    rewriter.insert_op(fifo, InsertPoint.after(last))


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


def _kernel_dims(op: ComputationNodeOp) -> list:
    return [var.dim for var in cast(StrensorType, op.output.type).ssis.data.get_kernel_variables()]


def _store_index(buffer: SSAValue, key: SSAValue, query: SSAValue) -> list[Operation]:
    return [
        first := ConstantOp.from_int_and_width(0, IndexType()),
        second := ConstantOp.from_int_and_width(1, IndexType()),
        StoreOp.get(key, buffer, [first.result]),
        StoreOp.get(query, buffer, [second.result]),
    ]


@dataclass
class CausalGemmKernel(GemmKernel):
    """A GEMM over the score matrix that leaves out the blocks a causal mask would zero.

    Both kernels behind it return before doing anything when the key block sits past the
    query block, so the scores of such a block are written and never read: the score GEMM
    is the one stage of the step that a mask inside the kernels cannot skip. The same test
    around the call skips it here, while the zeroing before it stays unconditional, which
    is what leaves the block defined whichever way the test goes.
    """

    @property
    def unique_name(self) -> str:
        return f"{super().unique_name}_causal"

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        query, key = _kernel_dims(op)
        key_ops, key_block, _ = _block_index(op, key)
        query_ops, query_block, _ = _block_index(op, query)
        return [
            *key_ops,
            *query_ops,
            attends := CmpiOp(key_block, query_block, "sle"),
            IfOp(attends, [], Region(Block([*GemmKernel.function_call(self, op), YieldOp()]))),
        ]


@dataclass
class PartialSoftmaxKernel(SoftmaxKernel):
    """One online-softmax step over an m x n block of the score matrix.

    Same row-wise shape as the plain softmax -- rows contiguous, the block's own width
    reduced -- but the row is now a slice of the key rather than the whole of it, so the
    running maximum and sum carry across the blocks in the scale buffer instead of
    finishing inside one call. Masking is the kernel's own business: it reads the block
    indices and drops everything a query may not attend, which is why this kernel takes
    no separate causal entry point.
    """

    # Which side of the step arrives or leaves in the tiling a GEMM works in. Neither is
    # the plain kernel; either one takes the mode-selectable entry point.
    tiled_in: bool = False
    tiled_out: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.m, self.n) != (FLASH_TILE, FLASH_TILE):
            raise ValueError(f"mha.cc is written for a {FLASH_TILE}x{FLASH_TILE} block, not {self.m}x{self.n}")

    @property
    def linkwith_name(self) -> str:
        return "mha.o"

    @property
    def function_name(self) -> str:
        return "partial_softmax_mode" if self.tiled_in or self.tiled_out else "partial_softmax"

    @property
    def unique_name(self) -> str:
        return f"{self.function_name}_{int(self.tiled_in)}{int(self.tiled_out)}_{self.m}_{self.n}"

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
        rows = MAC_ROWS_BFP16 if self.bfp16_mmul else R
        mt, nt = self.m // rows, self.n // T
        mac = TiledStridedLayout(
            [
                TiledStride([Stride(rows * T * nt, mt), Stride(T, rows)]),
                TiledStride([Stride(rows * T, nt), Stride(1, T)]),
            ]
        )
        return [mac if self.tiled_in else self._row_major(), mac if self.tiled_out else self._row_major()]

    def _scale_type(self) -> MemRefType:
        return MemRefType(self.element_type, (SCALE_ROWS * self.m,))

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        return FunctionType.from_lists(
            inputs=[
                op.inputs[0].type,
                op.inputs[1].type,
                self._scale_type(),
                MemRefType(i32, (2,)),
                self.element_type,
                i32,
                i32,
                i32,
                i32,
                *([i32] if self.mode else []),
            ],
            outputs=[],
        )

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        device, tile = _device(op), _tile(op)
        _index_buffer(device, tile, rewriter)
        self._state_buffer(device, tile, rewriter)
        _scale_fifo(device, rewriter, tile, _partner(device, op, "matmul_PV"), self.element_type, SCALE_ROWS * self.m)
        SymbolTable.insert_or_update(
            device,
            FuncOp(
                "init_scale_buffer",
                FunctionType.from_lists([self._scale_type(), i32], []),
                Region(),
                "private",
            ),
        )
        snapshot = FuncOp(
            SNAPSHOT, FunctionType.from_lists([self._scale_type(), self._scale_type(), i32], []), Region(), "private"
        )
        snapshot.attributes["link_with"] = StringAttr(SNAPSHOT_OBJECT)
        SymbolTable.insert_or_update(device, snapshot)
        AIEKernel.rewrite(self, op, rewriter)

    def _state_buffer(self, device: DeviceOp, tile: TileOp, rewriter: PatternRewriter | None = None) -> SSAValue:
        return _core_buffer(device, tile, "state", self.element_type, SCALE_ROWS * self.m, rewriter)

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        device, tile = _device(op), _tile(op)
        query, key = _kernel_dims(op)
        key_ops, key_block, key_blocks = _block_index(op, key)
        query_ops, query_block, query_blocks = _block_index(op, query)
        acquire = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Produce.get_int(), 32),
            IntegerAttr.from_int_and_width(1, 32),
            _scale_name(tile),
            (SCALE_ROWS * self.m,),
            self.element_type,
        )
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
            acquire,
            scale := ObjectFIFOSubviewAccessOp(IntegerAttr(0, i32), acquire),
            width := ConstantOp.from_int_and_width(SCALE_ROWS * self.m, i32),
            # Unconditional: a block this core skipped still owes the one behind it the
            # scale it last wrote, which is what the closing rescale divides by.
            CallOp(SNAPSHOT, [state, scale.output, width.result], []),
            ObjectFIFOReleaseOp(
                IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Produce.get_int(), 32),
                IntegerAttr.from_int_and_width(1, 32),
                _scale_name(tile),
            ),
        ]
        return ops


@dataclass
class FusedScoreSoftmaxKernel(GemmKernel):
    """A step's score side whole: the GEMM and the online softmax on one core.

    The softmax reads and writes the block a row at a time out of the GEMM's own MAC
    tiling, so nothing re-lays it out: the two share a core, and the block reaches
    ``matmul_PV`` as its GEMM operand without a memory tile in between. That frees a row
    of the array and the ports the swizzle used to cost. The running state, and the scale
    that crosses to ``matmul_PV``, are the same as when the softmax stands on its own.
    """

    def __post_init__(self) -> None:
        if (self.m, self.k, self.n) != (FLASH_TILE,) * 3:
            raise ValueError(
                f"mha.cc is written for a {FLASH_TILE} query, key and head block, not {self.m}x{self.k}x{self.n}"
            )

    @property
    def unique_name(self) -> str:
        return f"{self.function_name}_{self.m}_{self.k}_{self.n}"

    @property
    def linkwith_name(self) -> str:
        return "mha.o"

    @property
    def function_name(self) -> str:
        return "matmul_softmax"

    @property
    def zero_name(self) -> str:
        return "zero_bf16"

    def _scale_type(self) -> MemRefType:
        return MemRefType(self.element_type, (SCALE_ROWS * self.m,))

    def _state_buffer(self, device: DeviceOp, tile: TileOp, rewriter: PatternRewriter | None = None) -> SSAValue:
        return _core_buffer(device, tile, "state", self.element_type, SCALE_ROWS * self.m, rewriter)

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        return FunctionType.from_lists(
            inputs=[
                op.inputs[0].type,
                op.inputs[1].type,
                op.inputs[2].type,
                self._scale_type(),
                MemRefType(i32, (2,)),
                self.element_type,
                i32,
                i32,
                i32,
                i32,
            ],
            outputs=[],
        )

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        device, tile = _device(op), _tile(op)
        _index_buffer(device, tile, rewriter)
        self._state_buffer(device, tile, rewriter)
        _scale_fifo(device, rewriter, tile, _partner(device, op, "matmul_PV"), self.element_type, SCALE_ROWS * self.m)
        SymbolTable.insert_or_update(
            device,
            FuncOp("init_scale_buffer", FunctionType.from_lists([self._scale_type(), i32], []), Region(), "private"),
        )
        snapshot = FuncOp(
            SNAPSHOT, FunctionType.from_lists([self._scale_type(), self._scale_type(), i32], []), Region(), "private"
        )
        snapshot.attributes["link_with"] = StringAttr(SNAPSHOT_OBJECT)
        SymbolTable.insert_or_update(device, snapshot)
        GemmKernel.rewrite(self, op, rewriter)

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        device, tile = _device(op), _tile(op)
        query, key = _kernel_dims(op)
        key_ops, key_block, key_blocks = _block_index(op, key)
        query_ops, query_block, query_blocks = _block_index(op, query)
        acquire = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Produce.get_int(), 32),
            IntegerAttr.from_int_and_width(1, 32),
            _scale_name(tile),
            (SCALE_ROWS * self.m,),
            self.element_type,
        )
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
            acquire,
            scale := ObjectFIFOSubviewAccessOp(IntegerAttr(0, i32), acquire),
            width := ConstantOp.from_int_and_width(SCALE_ROWS * self.m, i32),
            CallOp(SNAPSHOT, [state, scale.output, width.result], []),
            ObjectFIFOReleaseOp(
                IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Produce.get_int(), 32),
                IntegerAttr.from_int_and_width(1, 32),
                _scale_name(tile),
            ),
        ]
        return ops


@dataclass
class FlashKernel(GemmKernel):
    """The value half of an online-softmax step: ``O += P V``, rescaled as the row max moves.

    A GEMM over the key with the probability block as its A operand, plus the two things
    the running state buys: the context is scaled by ``exp2(m_{i-1} - m_i)`` before every
    block after the first, and divided by the final row sum after the last one. The key
    is the contraction, so the context stays on the compute tile across the key loop by
    the same reuse the backend already gives a GEMM's output.
    """

    def __post_init__(self) -> None:
        if (self.m, self.k, self.n) != (FLASH_TILE,) * 3:
            raise ValueError(
                f"mha.cc is written for a {FLASH_TILE} query, key and head block, not {self.m}x{self.k}x{self.n}"
            )

    @property
    def unique_name(self) -> str:
        return f"{self.function_name}_{self.m}_{self.k}_{self.n}"

    @property
    def linkwith_name(self) -> str:
        return "mha.o"

    @property
    def function_name(self) -> str:
        return "matmul_PV"

    @property
    def zero_name(self) -> str:
        return "zero_bf16"

    def _scale_type(self) -> MemRefType:
        return MemRefType(self.element_type, (SCALE_ROWS * self.m,))

    def function_type(self, op: ComputationNodeOp) -> FunctionType:
        return FunctionType.from_lists(
            inputs=[
                op.inputs[0].type,
                op.inputs[1].type,
                op.inputs[2].type,
                self._scale_type(),
                i32,
                i32,
                MemRefType(i32, (2,)),
            ],
            outputs=[],
        )

    def _rescale_type(self, op: ComputationNodeOp) -> FunctionType:
        return FunctionType.from_lists(
            inputs=[op.inputs[2].type, self._scale_type(), i32, MemRefType(i32, (2,))],
            outputs=[],
        )

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        device, tile = _device(op), _tile(op)
        _index_buffer(device, tile, rewriter)
        for source in _partners(device, op, SCORE_SIDE):
            _scale_fifo(device, rewriter, source, tile, self.element_type, SCALE_ROWS * self.m)
        SymbolTable.insert_or_update(device, FuncOp("rescale_O", self._rescale_type(op), Region(), "private"))
        GemmKernel.rewrite(self, op, rewriter)

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        self.check_operands(op)
        device, tile = _device(op), _tile(op)
        sources = _partners(device, op, SCORE_SIDE)
        kernel_dims = _kernel_dims(op)
        space = cast(StrensorType, op.output.type).ssis.data
        reduced = {var.dim for var in space.vars if var.type is not StrensorVarType.KERNEL} - set(kernel_dims)
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
        selector = induction_variable(op, self._turn_var(op))
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

    @staticmethod
    def _turn_var(op: ComputationNodeOp):
        """The variable saying which of its sources this core is taking from.

        A dimension split between cores and loops has more than one temporal part, and the
        turn is the innermost of them -- the same one the object fifo switch alternates on,
        which reads the last temporal variable of the space. Taking the outermost instead
        only looks right while the two are the same size, and then picks the wrong loop.
        """
        dim = _split_dim(op)
        space = cast(StrensorType, op.output.type).ssis.data
        return next(v for v in reversed(space.vars) if v.type is StrensorVarType.TEMPORAL and v.dim == dim)

    def _step(self, op: ComputationNodeOp, source: TileOp, key_block, key_blocks, index) -> list[Operation]:
        """One value accumulation against the scale the given score core wrote."""
        acquire = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Consume.get_int(), 32),
            IntegerAttr.from_int_and_width(1, 32),
            _scale_name(source),
            (SCALE_ROWS * self.m,),
            self.element_type,
        )
        return [
            acquire,
            scale := ObjectFIFOSubviewAccessOp(IntegerAttr(0, i32), acquire),
            rows := ConstantOp.from_int_and_width(self.m, i32),
            zero := ConstantOp.from_int_and_width(0, i32),
            # Block zero is never causally skipped, so testing it at run time is the same
            # as the peeled first iteration the kernel was written for.
            opened := CmpiOp(key_block, zero, "ne"),
            carried := ExtUIOp(opened, i32),
            CallOp(
                self.function_name,
                [op.inputs[0], op.inputs[1], op.inputs[2], scale.output, rows.result, carried.result, index],
                [],
            ),
            last := ConstantOp.from_int_and_width(key_blocks - 1, i32),
            closing := CmpiOp(key_block, last, "eq"),
            IfOp(
                closing,
                [],
                Region(Block([CallOp("rescale_O", [op.inputs[2], scale.output, rows.result, index], []), YieldOp()])),
            ),
            ObjectFIFOReleaseOp(
                IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Consume.get_int(), 32),
                IntegerAttr.from_int_and_width(1, 32),
                _scale_name(source),
            ),
        ]
