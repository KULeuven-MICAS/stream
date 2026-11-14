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
pytest.importorskip("aie.iron.kernels", reason="the library binds mha.cc through mlir-aie")

from pathlib import Path  # noqa: E402

from stream.compiler.kernels.library import KernelLibrary  # noqa: E402

TILE = 64
BLOCKS = 8
COLUMNS = 4
LIBRARY = KernelLibrary.load(Path(__file__).parents[2] / "stream/inputs/aie/kernels/aie2p.toml")


@pytest.fixture(scope="module")
def column():
    """One column of the fused flash design, rewritten the way codegen rewrites it.

    Score GEMM, online softmax and value GEMM on three neighbouring tiles, each in the
    query loop over the key loop the blocked design nests them in.
    """
    return _column(split=True)[0]


@pytest.fixture(scope="module")
def unsplit():
    """The same column where one query block is all there is, so no core splits it."""
    return _column(split=False)[0]


@pytest.fixture(scope="module")
def full():
    """The column of an attention without a mask, and the bindings its calls resolved to."""
    return _column(split=True, causal=False)


def _column(split, causal=True, rows=TILE, keys=TILE):
    """The column at ``rows`` queries by ``keys`` keys a block, its head ``TILE`` wide."""
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
    from stream.compiler.kernels.binding import Bindings
    from stream.compiler.kernels.registry import AIE_KERNELS
    from stream.compiler.transforms.convert_aie_kernels import ConvertAIEKernels
    from stream.datatypes import LayerDim

    query, key, head = LayerDim(0), LayerDim(2), LayerDim(3)
    query_loop = StrensorVar(StrensorVarType.TEMPORAL, BLOCKS // COLUMNS, query)
    key_loop = StrensorVar(StrensorVarType.TEMPORAL, BLOCKS, key)
    outer = (query_loop, StrensorVar(StrensorVarType.SPATIAL, COLUMNS, query), key_loop) if split else (key_loop,)
    # A point variable carries its position, so this is the first column of the four.
    points = (StrensorVar(StrensorVarType.POINT, 0, query),) if split else ()
    position = StrensorSpaceAttr(StrensorSpace(points))

    def node(kernel_name, shapes, kernel_dims):
        """One computation node, its operands taken from fifos the way codegen leaves them."""
        acquires = [
            ObjectFifoAcquireOp(
                IntegerAttr.from_int_and_width(ObjectFifoPortEnum.Consume.get_int(), 32),
                1,
                "operand",
                shape,
                bf16,
            )
            for shape in shapes
        ]
        kernel_vars = tuple(StrensorVar(StrensorVarType.KERNEL, size, dim) for dim, size in kernel_dims)
        output = StrensorType(bf16, StrensorSpace(outer + kernel_vars))
        node = ComputationNodeOp([a.results[0] for a in acquires], (output,), kernel_name, position)
        return [*acquires, node]

    def core(row, body):
        """The node on its own tile, wrapped in the key loop and then the query loop."""
        ops = body
        for var in (key_loop, query_loop) if split else (key_loop,):
            bounds = [ConstantOp.from_int_and_width(bound, IndexType()) for bound in (0, var.size, 1)]
            loop = ForOp(*bounds, [], Region(Block([*ops, YieldOp()], arg_types=[IndexType()])))
            loop.attributes["layer_dim"] = StrensorVarAttr(var)
            ops = [*bounds, loop]
        tile = TileOp(0, row)
        return [tile, CoreOp(None, tile, Region(Block([*ops, EndOp()])))]

    flags = {"layout": "default", "causal": causal, "library": LIBRARY}
    scores = AIE_KERNELS["gemm"](m=rows, k=TILE, n=keys, **flags)
    softmax = AIE_KERNELS["partial_softmax"](m=rows, n=keys, layout="contiguous", causal=causal, library=LIBRARY)
    context = AIE_KERNELS["gemm"](m=rows, k=keys, n=TILE, flash=True, **flags)
    score_side, value_side = ((query, rows), (key, keys)), ((query, rows), (head, TILE))
    nodes = [
        node(scores.unique_name, [(rows, TILE), (TILE, keys), (rows, keys)], score_side),
        node(softmax.unique_name, [(rows, keys)] * 2, score_side),
        node(context.unique_name, [(rows, keys), (keys, TILE), (rows, TILE)], value_side),
    ]
    tiles = [op for row, body in enumerate(nodes) for op in core(row, body)]
    device = DeviceOp(
        IntegerAttr.from_int_and_width(AIEDeviceEnum.npu2.get_int(), 32), Region(Block([*tiles, EndOp()]))
    )
    kernels = {kernel.unique_name: kernel for kernel in (scores, softmax, context)}
    bindings = Bindings("npu2")
    PatternRewriteWalker(ConvertAIEKernels(kernels, bindings)).rewrite_module(ModuleOp([device]))
    return device, bindings


def _call(device, name):
    """The one call to ``name``, under the prefix its binding gives it, in the rewritten column."""
    from xdsl.dialects.func import CallOp

    calls = [op for op in device.walk() if isinstance(op, CallOp) and op.callee.root_reference.data.endswith(name)]
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

    condition = _guard(_call(column, "matmul_bf16_bf16"))
    assert isinstance(condition, CmpiOp) and "arith.cmpi sle," in str(condition)
    assert _driven_by(condition.lhs, LayerDim(2)) and _driven_by(condition.rhs, LayerDim(0))


def _evaluate(value, loops):
    """``value`` as an integer, with each loop's induction variable at ``loops[dim]``."""
    from xdsl.dialects.arith import AddiOp, CmpiOp, ConstantOp, IndexCastOp, MuliOp
    from xdsl.ir import BlockArgument

    if isinstance(value, BlockArgument):
        return loops[value.block.parent_op().attributes["layer_dim"].data.dim]
    op = value.op
    if isinstance(op, ConstantOp):
        return op.value.value.data
    if isinstance(op, IndexCastOp):
        return _evaluate(op.input, loops)
    lhs, rhs = _evaluate(op.lhs, loops), _evaluate(op.rhs, loops)
    if isinstance(op, CmpiOp):
        assert "arith.cmpi sle," in str(op)
        return lhs <= rhs
    return {AddiOp: lhs + rhs, MuliOp: lhs * rhs}[type(op)]


@pytest.mark.parametrize("rows, keys", [(64, 32), (32, 64)])
def test_the_score_gemm_skips_a_block_only_past_the_query_blocks_last_row(rows, keys):
    """Key block ``j`` holds keys from ``j * keys``; the first column's query block at loop index ``i`` is the array's
    block ``COLUMNS * i`` and ends at row ``COLUMNS * i * rows + rows - 1``. At 64 rows over 32 keys, query
    block 0 attends key blocks 0 and 1: comparing block indices would drop block 1."""
    from stream.datatypes import LayerDim

    device, _ = _column(split=True, rows=rows, keys=keys)
    condition = _guard(_call(device, "matmul_bf16_bf16"))
    for i in range(BLOCKS // COLUMNS):
        attended = [j for j in range(BLOCKS) if _evaluate(condition.results[0], {LayerDim(0): i, LayerDim(2): j})]
        assert attended == [j for j in range(BLOCKS) if j * keys <= COLUMNS * i * rows + rows - 1]


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
    from stream.compiler.kernels.registry import AIE_KERNELS

    shape = {"m": TILE, "k": TILE, "n": TILE, "layout": "default", "library": LIBRARY}
    fused = AIE_KERNELS["matmul_softmax"](**shape)
    assert fused.function_name == "matmul_softmax"
    assert fused.operand_layouts() == AIE_KERNELS["gemm"](**shape).operand_layouts()


def test_only_the_kernels_that_carry_a_running_scale_declare_state():
    """What a kernel keeps between iterations, declared where the size is decided.

    The value GEMM keeps an index buffer too, but both its entries are written on every call
    and read within it, so nothing crosses an iteration and it is not a recurrence. Declaring
    it as one would make the value core a carrier in the recurrence bound and forbid the two
    halves of a step from overlapping.
    """
    from stream.compiler.kernels.registry import AIE_KERNELS

    declaring = {
        name: [s.name for s in kernel.state_operands()]
        for name, kernel in (
            ("partial_softmax", AIE_KERNELS["partial_softmax"](n=TILE, layout="contiguous", m=TILE)),
            ("matmul_softmax", AIE_KERNELS["matmul_softmax"](m=TILE, k=TILE, n=TILE, layout="default")),
            ("gemm", AIE_KERNELS["gemm"](m=TILE, k=TILE, n=TILE, layout="default", flash=True)),
            ("softmax", AIE_KERNELS["softmax"](n=TILE, layout="contiguous")),
        )
    }
    assert declaring == {
        "partial_softmax": ["flash_state"],
        "matmul_softmax": ["flash_state"],
        "gemm": [],
        "softmax": [],
    }


def test_the_declared_state_is_the_size_the_core_buffer_holds(column):
    """The declaration and the allocation are two statements of one fact, so they are checked
    against each other rather than both against a number written twice."""
    from xdsl_aie.dialects.aie import BufferOp

    from stream.compiler.kernels.registry import AIE_KERNELS

    kernel = AIE_KERNELS["partial_softmax"](n=TILE, layout="contiguous", m=TILE)
    (state,) = kernel.state_operands()
    (buffer,) = (op for op in column.walk() if isinstance(op, BufferOp) and op.sym_name.data.startswith("flash_state"))
    # The declaration says rows per step, and the node supplies the query extent that a split then divides.
    assert buffer.buffer.type.get_shape() == (state.rows * kernel.m,)
    assert buffer.buffer.type.element_type == state.element_type
    assert (state.carried_over, state.indexed_by) == ("n", "m")


def test_an_unsplit_step_still_hands_its_scale_on(unsplit):
    """With no split to say which cores pair up, the two halves of a step share one column."""
    from xdsl_aie.dialects.aie import ObjectFifoOp

    assert [op.sym_name.data for op in unsplit.walk() if isinstance(op, ObjectFifoOp)] == ["flash_scale_0_1_0_2"]


def _array(size, dtype):
    import numpy as np

    return np.ndarray[(size,), np.dtype[dtype]]


def _flash_entries(m, keys, head, causal=True):
    """Each flash entry point of mha.cc the providers bind, at an m-query block over ``keys`` keys of a ``head``-wide
    value: the softmax side's on the key block, which is all it reads, the value side's at the shape given."""
    from stream.compiler.kernels import mlir_aie

    softmax = mlir_aie.partial_softmax("npu2", m, keys, causal=causal)
    context = mlir_aie.matmul_pv("npu2", m, keys, head, causal=causal)
    return {
        "partial_softmax": softmax,
        "init_scale_buffer": softmax.init_scale_buffer,
        "matmul_PV": context,
        "rescale_O": context.rescale_O,
    }


@pytest.mark.parametrize("m, head", [(32, 64), (64, 64), (32, 128)])
def test_the_flash_entries_take_mlir_aie_1_4_6s_arguments(m, head):
    """mlir-aie 1.4.6 keeps O and the running state in float, and ``rescale_O`` writes O into a bf16 output: a binding
    that still passed bf16 compiled, linked and computed garbage. The value side is ``head`` wide whatever the key
    block: V is 32 x head, O m x head."""
    import numpy as np
    from ml_dtypes import bfloat16

    keys = 32
    index, state = _array(3, np.int32), _array(4 * m, np.float32)
    expected = {
        "partial_softmax": [
            _array(m * keys, bfloat16),
            _array(m * keys, bfloat16),
            state,
            index,
            bfloat16,
            *[np.int32] * 4,
        ],
        "init_scale_buffer": [state, np.int32],
        "matmul_PV": [
            _array(m * keys, bfloat16),
            _array(keys * head, bfloat16),
            _array(m * head, np.float32),
            state,
            np.int32,
            np.int32,
            index,
            np.int32,
        ],
        "rescale_O": [_array(m * head, np.float32), _array(m * head, bfloat16), state, np.int32, index],
    }
    assert {name: entry.arg_types() for name, entry in _flash_entries(m, keys, head).items()} == expected


@pytest.mark.parametrize("head, causal", [(64, True), (128, True), (64, False), (128, False)])
def test_the_flash_entries_take_what_mha_cc_defines(head, causal):
    """Arity and element types against the C definitions in the source the bindings compile; ``score_t`` is bf16, as
    no binding builds ``MHA_F32_SCORES``."""
    import re
    from pathlib import Path
    from typing import get_args

    import numpy as np
    from ml_dtypes import bfloat16

    c_types = {"bfloat16": bfloat16, "score_t": bfloat16, "float": np.float32, "int32_t": np.int32}
    for name, entry in _flash_entries(64, 64, head, causal).items():
        source = Path(entry.source_file).read_text()
        (parameters,) = re.findall(rf"^void {name}\(([^)]*)\)\s*{{", source, re.MULTILINE)
        defined = []
        for parameter in parameters.split(","):
            words = re.sub(r"\bconst\b|\b__restrict\b|\*", " ", parameter).split()
            defined.append(("*" in parameter, np.dtype(c_types[words[0]])))
        declared = [
            (True, np.dtype(get_args(get_args(arg)[1])[0])) if get_args(arg) else (False, np.dtype(arg))
            for arg in entry.arg_types()
        ]
        assert declared == defined, name


@pytest.mark.parametrize("causal", [True, False])
def test_the_flash_entries_are_built_with_the_mask_asked_for(causal):
    """mha.cc is causal unless built with ``MHA_CAUSAL=0``; the value side at head 128 is built at ``DIM_K=128``, the
    softmax side at its key block, 64."""
    entries = _flash_entries(32, 64, 128, causal)
    for name, entry in entries.items():
        assert ("-DMHA_CAUSAL=0" in entry.compile_flags) is not causal, name
    assert "-DDIM_K=128" in entries["matmul_PV"].compile_flags
    assert "-DDIM_K=64" in entries["partial_softmax"].compile_flags


@pytest.mark.parametrize("head, causal", [(64, True), (128, False)])
def test_the_flash_entries_build_mha_cc_as_irons_attention_does(head, causal):
    """IRON's attention builds mha.cc once per pipeline, at (B_q, d, B_kv), with K as stored, no bfp16 multiplies and
    AIE2P's accurate exp2; the value side binds that object, so its flags are those, and the softmax side's differ only
    in the head, which it does not read."""
    from aie.iron.kernels import linalg

    from stream.compiler.kernels import mlir_aie

    mlir_aie._target("npu2")
    expected = linalg.mha(32, head, 32, b_col_maj=True, accurate_exp2=True, causal=causal).compile_flags
    entries = _flash_entries(32, 32, head, causal)
    assert entries["matmul_PV"].compile_flags == expected
    assert entries["partial_softmax"].compile_flags == [f if f != f"-DDIM_K={head}" else "-DDIM_K=32" for f in expected]


def test_an_unmasked_step_links_mha_cc_built_without_its_mask(full):
    """The softmax and value cores call objects built with ``MHA_CAUSAL=0``: a causal object there would drop every
    key past each query, and the value side would rescale by the state of a block the softmax skipped."""
    _, bindings = full
    linked = {binding.name.split("_", 1)[1]: binding for binding in bindings.used.values()}
    for name in ("partial_softmax", "init_scale_buffer", "matmul_PV", "rescale_O"):
        assert "-DMHA_CAUSAL=0" in linked[name].compile_flags, name


def test_the_value_core_accumulates_in_float_and_rescales_into_the_output(column):
    """``matmul_PV`` adds into the core's float buffer, which it zeroes on the first key block itself, and the closing
    ``rescale_O`` writes that buffer into the bf16 output tile."""
    from math import prod

    from xdsl.dialects.builtin import MemRefType, bf16, f32
    from xdsl.dialects.func import CallOp
    from xdsl_aie.dialects.aie import CoreOp

    pv, rescale = _call(column, "matmul_PV"), _call(column, "rescale_O")
    accumulator, output = rescale.arguments[0], rescale.arguments[1]
    assert pv.arguments[2] is accumulator and accumulator.type == MemRefType(f32, (TILE * TILE,))
    assert output.type.element_type == bf16 and prod(output.type.get_shape()) == TILE * TILE
    core = pv.parent_op()
    while not isinstance(core, CoreOp):
        core = core.parent_op()
    calls = [op.callee.root_reference.data for op in core.walk() if isinstance(op, CallOp)]
    assert not any(callee.endswith("zero") for callee in calls)
