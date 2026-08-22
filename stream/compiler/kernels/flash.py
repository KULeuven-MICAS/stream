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
from typing import cast

from xdsl.dialects.arith import AddiOp, CmpiOp, ConstantOp, ExtUIOp, IndexCastOp, MuliOp, TruncFOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerAttr,
    MemRefType,
    StringAttr,
    f32,
    i32,
)
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.memref import StoreOp
from xdsl.dialects.scf import IfOp, YieldOp
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
from stream.compiler.kernels.aie_kernel import AIEKernel, induction_variable
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


def _runs(core: CoreOp, function: str) -> bool:
    """Whether this core runs ``function``, before or after its node was rewritten."""
    return any(
        (isinstance(op, ComputationNodeOp) and op.kernel.data.startswith(function))
        or (isinstance(op, CallOp) and op.callee.root_reference.data == function)
        for op in core.walk()
    )


def _partner(device: DeviceOp, tile: TileOp, function: str) -> TileOp:
    """The tile running the other half of this step, which shares the scale buffer.

    The two halves sit above one another in one column, and only neighbouring tiles
    share memory, which is what lets the scale cross without a DMA channel.
    """
    col, row = _position(tile)
    found = [
        core.tile.op
        for core in device.walk()
        if isinstance(core, CoreOp)
        and isinstance(core.tile, OpResult)
        and isinstance(core.tile.op, TileOp)
        and _position(core.tile.op) in ((col, row - 1), (col, row + 1))
        and _runs(core, function)
    ]
    if len(found) != 1:
        raise ValueError(
            f"the core on tile ({col}, {row}) shares its scale buffer with the one "
            f"running {function}, which has to be its neighbour in the same column; "
            f"{len(found)} of the two tiles beside it run it"
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
    ops: list[Operation] = []
    terms: list[SSAValue] = []
    constant, stride = 0, 1
    for var in reversed(space.vars):
        if var.dim != dim or var.type is StrensorVarType.KERNEL:
            continue
        if var.type is StrensorVarType.SPATIAL:
            constant += position.get(dim, 0) * stride
        elif var.type is StrensorVarType.TEMPORAL:
            ops += [
                index := IndexCastOp(induction_variable(op, var), i32),
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

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.m, self.n) != (FLASH_TILE, FLASH_TILE):
            raise ValueError(f"mha.cc is written for a {FLASH_TILE}x{FLASH_TILE} block, not {self.m}x{self.n}")

    @property
    def unique_name(self) -> str:
        return f"{self.function_name}_{self.m}_{self.n}"

    @property
    def linkwith_name(self) -> str:
        return "mha.o"

    @property
    def function_name(self) -> str:
        return "partial_softmax"

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
            ],
            outputs=[],
        )

    def rewrite(self, op: ComputationNodeOp, rewriter: PatternRewriter) -> None:
        device, tile = _device(op), _tile(op)
        _index_buffer(device, tile, rewriter)
        self._state_buffer(device, tile, rewriter)
        _scale_fifo(device, rewriter, tile, _partner(device, tile, "matmul_PV"), self.element_type, SCALE_ROWS * self.m)
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
        _scale_fifo(
            device, rewriter, _partner(device, tile, "partial_softmax"), tile, self.element_type, SCALE_ROWS * self.m
        )
        SymbolTable.insert_or_update(device, FuncOp("rescale_O", self._rescale_type(op), Region(), "private"))
        GemmKernel.rewrite(self, op, rewriter)

    def function_call(self, op: ComputationNodeOp) -> Sequence[Operation]:
        self.check_operands(op)
        device, tile = _device(op), _tile(op)
        source = _partner(device, tile, "partial_softmax")
        kernel_dims = _kernel_dims(op)
        space = cast(StrensorType, op.output.type).ssis.data
        reduced = {var.dim for var in space.vars if var.type is not StrensorVarType.KERNEL} - set(kernel_dims)
        if len(reduced) > 1:
            raise ValueError(f"kernel {self.function_name} accumulates over one dimension, not {sorted(reduced)}")
        key_ops, key_block, key_blocks = _block_index(op, next(iter(reduced), None))
        query_ops, query_block, _ = _block_index(op, kernel_dims[0])
        acquire = ObjectFifoAcquireOp(
            IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Consume.get_int(), 32),
            IntegerAttr.from_int_and_width(1, 32),
            _scale_name(source),
            (SCALE_ROWS * self.m,),
            self.element_type,
        )
        index = _index_buffer(device, tile)
        ops: list[Operation] = [*key_ops, *query_ops]
        ops += _store_index(index, key_block, query_block)
        ops += [
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
        return ops
