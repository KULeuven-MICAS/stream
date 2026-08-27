"""The guards the flash bindings emit around mha.cc's entry points.

Each is silently wrong when it changes: the score GEMM's causal test decides which blocks
exist at all, the snapshot is what the rescale behind it divides by, the running state is
reset once per query block rather than once per key block, and at depth one the scale fifo
would run the two cores in lockstep. Nothing but hardware sees any of it, so the ops the
bindings emit are asserted on here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")

TILE = 64
BLOCKS = 8
COLUMNS = 4


@pytest.fixture(scope="module")
def column():
    """One column of the fused flash design, rewritten the way codegen rewrites it.

    Score GEMM, online softmax and value GEMM on three neighbouring tiles, each in the
    query loop over the key loop the blocked design nests them in.
    """
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp, bf16
    from xdsl.dialects.scf import ForOp, YieldOp
    from xdsl.ir import Block, Region
    from xdsl.pattern_rewriter import PatternRewriteWalker
    from xdsl_aie.dialects.aie import (
        AIEDeviceEnum,
        CoreOp,
        DeviceOp,
        EndOp,
        ObjectFifoAcquireOp,
        ObjectFifoPortEnum,
        ObjectFIFOSubviewAccessOp,
        TileOp,
    )

    from stream.compiler.dialects.stream import (
        ComputationNodeOp,
        StrensorSpace,
        StrensorSpaceAttr,
        StrensorType,
        StrensorVar,
        StrensorVarAttr,
        StrensorVarType,
    )
    from stream.compiler.kernels.flash import CausalGemmKernel, FlashKernel, PartialSoftmaxKernel
    from stream.compiler.transforms.convert_aie_kernels import ConvertAIEKernels
    from stream.datatypes import LayerDim

    query, key, head = LayerDim(0), LayerDim(2), LayerDim(3)
    query_loop = StrensorVar(StrensorVarType.TEMPORAL, BLOCKS // COLUMNS, query)
    key_loop = StrensorVar(StrensorVarType.TEMPORAL, BLOCKS, key)
    outer = (query_loop, StrensorVar(StrensorVarType.SPATIAL, COLUMNS, query), key_loop)
    # A point variable carries its position, so this is the first column of the four.
    position = StrensorSpaceAttr(StrensorSpace((StrensorVar(StrensorVarType.POINT, 0, query),)))

    def node(kernel_name, operands, kernel_dims):
        """One computation node, its operands taken from fifos the way codegen leaves them."""
        acquires = [
            ObjectFifoAcquireOp(
                IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Consume.get_int(), 32),
                IntegerAttr.from_int_and_width(1, 32),
                "operand",
                (TILE, TILE),
                bf16,
            )
            for _ in range(operands)
        ]
        accesses = [ObjectFIFOSubviewAccessOp(IntegerAttr(0, 32), acquire) for acquire in acquires]
        kernel_vars = tuple(StrensorVar(StrensorVarType.KERNEL, TILE, dim) for dim in kernel_dims)
        output = StrensorType(bf16, StrensorSpace(outer + kernel_vars))
        node = ComputationNodeOp([access.output for access in accesses], (output,), kernel_name, position)
        return [op for pair in zip(acquires, accesses, strict=True) for op in pair] + [node]

    def core(row, body):
        """The node on its own tile, wrapped in the key loop and then the query loop."""
        ops = body
        for var in (key_loop, query_loop):
            bounds = [ConstantOp.from_int_and_width(bound, IndexType()) for bound in (0, var.size, 1)]
            loop = ForOp(*bounds, [], Region(Block([*ops, YieldOp()], arg_types=[IndexType()])))
            loop.attributes["layer_dim"] = StrensorVarAttr(var)
            ops = [*bounds, loop]
        tile = TileOp(0, row)
        return [tile, CoreOp(None, tile, Region(Block([*ops, EndOp()])))]

    scores = CausalGemmKernel(61.8, bf16, TILE, TILE, TILE, "default", True)
    softmax = PartialSoftmaxKernel(50.0, bf16, TILE, TILE, "contiguous", True)
    context = FlashKernel(61.8, bf16, TILE, TILE, TILE, "default", True)
    device = DeviceOp(
        IntegerAttr.from_int_and_width(AIEDeviceEnum.npu2.get_int(), 32),
        Region(
            Block(
                [
                    *core(0, node(scores.unique_name, 3, (query, key))),
                    *core(1, node(softmax.unique_name, 2, (query, key))),
                    *core(2, node(context.unique_name, 3, (query, head))),
                    EndOp(),
                ]
            )
        ),
    )
    kernels = {kernel.unique_name: kernel for kernel in (scores, softmax, context)}
    PatternRewriteWalker(ConvertAIEKernels(kernels)).rewrite_module(ModuleOp([device]))
    return device


def _call(device, name):
    """The one call to ``name`` in the rewritten column."""
    from xdsl.dialects.func import CallOp

    calls = [op for op in device.walk() if isinstance(op, CallOp) and op.callee.root_reference.data == name]
    assert len(calls) == 1, f"{len(calls)} calls to {name}"
    return calls[0]


def _guard(call):
    """The ``scf.if`` condition the call sits under, or None where it runs unconditionally."""
    from xdsl.dialects.scf import IfOp

    parent = call.parent_op()
    while parent is not None:
        if isinstance(parent, IfOp):
            return parent.cond.op
        parent = parent.parent_op()
    return None


def _driven_by(value, dim):
    """Whether ``value`` is computed from the induction variable of the loop over ``dim``."""
    from xdsl.dialects.scf import ForOp
    from xdsl.ir import BlockArgument, OpResult

    if isinstance(value, BlockArgument):
        loop = value.block.parent_op()
        return isinstance(loop, ForOp) and loop.attributes["layer_dim"].data.dim == dim
    return isinstance(value, OpResult) and any(_driven_by(operand, dim) for operand in value.op.operands)


def test_the_score_gemm_skips_only_the_blocks_past_the_diagonal(column):
    """``sle``: the diagonal block is the one a query both attends and is masked inside."""
    from xdsl.dialects.arith import CmpiOp

    from stream.datatypes import LayerDim

    condition = _guard(_call(column, "matmul_bf16_bf16_64_64_64"))
    assert isinstance(condition, CmpiOp) and "arith.cmpi sle," in str(condition)
    assert _driven_by(condition.lhs, LayerDim(2)) and _driven_by(condition.rhs, LayerDim(0))


def test_the_scale_snapshot_is_unconditional(column):
    """A block the softmax skipped still owes the core behind it the scale it last wrote."""
    assert _guard(_call(column, "passThroughLine")) is None


def test_the_state_is_reset_once_per_query_block(column):
    """The reset is the key loop's first block, not every block and not the query's."""
    from xdsl.dialects.arith import CmpiOp, ConstantOp

    from stream.datatypes import LayerDim

    condition = _guard(_call(column, "init_scale_buffer"))
    assert isinstance(condition, CmpiOp) and "arith.cmpi eq," in str(condition)
    assert _driven_by(condition.lhs, LayerDim(2))
    assert isinstance(zero := condition.rhs.op, ConstantOp) and zero.value.value.data == 0


def test_the_scale_fifo_is_two_deep(column):
    """At depth one the softmax could not enter a block before the GEMM had left the one before.

    Named after both ends, since a softmax core handing to two value cores holds one fifo
    apiece and the producer alone no longer tells them apart.
    """
    from xdsl_aie.dialects.aie import ObjectFifoOp

    fifos = [op for op in column.walk() if isinstance(op, ObjectFifoOp)]
    assert [fifo.sym_name.data for fifo in fifos] == ["flash_scale_0_1_0_2"]
    assert fifos[0].elemNumber.value.data == 2


def test_the_fused_score_kernel_stays_in_a_gemms_tilings():
    """Fusing the score GEMM into the softmax only pays off if neither side is re-laid out.

    Nothing but IRON_FUSED_KERNEL reaches this kernel, so no other test would notice its
    layouts drifting away from the GEMM it replaces.
    """
    from stream.compiler.kernels import AIEKernels

    shape = {"utilization": 50.0, "m": TILE, "k": TILE, "n": TILE, "layout": "default"}
    fused = AIEKernels["matmul_softmax"](**shape)
    assert fused.function_name == "matmul_softmax"
    assert fused.operand_layouts() == AIEKernels["gemm"](**shape).operand_layouts()


def test_only_the_kernels_that_carry_a_running_scale_declare_state():
    """What a kernel keeps between iterations, declared where the size is decided.

    The value GEMM keeps an index buffer too, but both its entries are written on every call
    and read within it, so nothing crosses an iteration and it is not a recurrence. Declaring
    it as one would make the value core a carrier in the recurrence bound and forbid the two
    halves of a step from overlapping.
    """
    from stream.compiler.kernels import AIEKernels

    declaring = {
        name: [s.name for s in kernel.state_operands()]
        for name, kernel in (
            ("partial_softmax", AIEKernels["partial_softmax"](utilization=50.0, n=TILE, layout="contiguous", m=TILE)),
            (
                "matmul_softmax",
                AIEKernels["matmul_softmax"](utilization=50.0, m=TILE, k=TILE, n=TILE, layout="default"),
            ),
            ("gemm", AIEKernels["gemm"](utilization=61.8, m=TILE, k=TILE, n=TILE, layout="default", flash=True)),
            ("softmax", AIEKernels["softmax"](utilization=50.0, n=TILE, layout="contiguous")),
        )
    }
    assert declaring == {
        "partial_softmax": ["flash_state"],
        "matmul_softmax": ["flash_state"],
        "gemm": [],
        "softmax": [],
    }


def test_the_declared_state_is_the_size_the_core_buffer_holds():
    """The declaration and the allocation are two statements of one fact, so they are checked
    against each other rather than both against a number written twice."""
    from stream.compiler.kernels import AIEKernels
    from stream.compiler.kernels.flash import SCALE_ROWS

    kernel = AIEKernels["partial_softmax"](utilization=50.0, n=TILE, layout="contiguous", m=TILE)
    (state,) = kernel.state_operands()
    # _core_buffer allocates SCALE_ROWS * m elements; the declaration says rows per step, and
    # the node supplies the query extent that a split then divides.
    assert state.rows * kernel.m == SCALE_ROWS * kernel.m
    assert (state.carried_over, state.indexed_by) == (1, 0)
