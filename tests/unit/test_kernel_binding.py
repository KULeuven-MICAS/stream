from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("snaxc", reason="the AIE kernels are a separate install, via stream-setup-aie")

from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp, bf16, f32, i32  # noqa: E402
from xdsl.dialects.func import CallOp, FuncOp  # noqa: E402
from xdsl.ir import Block, Region  # noqa: E402
from xdsl.ir.affine import AffineMap  # noqa: E402
from xdsl.pattern_rewriter import PatternRewriteWalker  # noqa: E402
from xdsl_aie.dialects import aie  # noqa: E402

from stream.compiler.dialects import stream  # noqa: E402
from stream.compiler.kernels.binding import Bindings, check, load_bindings  # noqa: E402
from stream.compiler.kernels.library import KernelLibrary  # noqa: E402
from stream.compiler.kernels.registry import AIE_KERNELS  # noqa: E402
from stream.compiler.transforms.convert_aie_kernels import ConvertAIEKernels  # noqa: E402
from stream.datatypes import LayerDim  # noqa: E402
from stream.workload.node import ComputationNode  # noqa: E402
from stream.workload.tensor import Tensor  # noqa: E402

LIBRARY = KernelLibrary.from_dict(
    {
        "family": {"matmul": {"ops_per_cycle": 151.0, "mac": {"m": 8, "k": 8, "n": 8}}},
        "kernel": {
            "matmul_bf16_bf16": {
                "family": "matmul",
                "object": "mm.o",
                "dims": [{"name": "k"}, {"name": "n"}, {"name": "m", "blocks": [16, 32]}],
            }
        },
    }
)
SHIPPED = KernelLibrary.load(Path(__file__).parents[2] / "stream/inputs/aie/kernels/aie2p.toml")


def _gemm_node(heads: int, maps):
    lead = (heads,) if heads else ()
    a, b = Tensor.create("a", bf16, (*lead, 32, 64)), Tensor.create("b", bf16, (*lead, 64, 16))
    out = Tensor.create("c", bf16, (*lead, 32, 16))
    return ComputationNode(type="MatMul", name="mm", inputs=(a, b), outputs=(out,), operand_mapping=maps)


@pytest.mark.parametrize(
    "node, expected",
    [
        # Gemm iterates (m, k, n).
        (
            _gemm_node(
                0,
                tuple(
                    AffineMap.from_callable(f)
                    for f in (lambda m, k, n: (m, k), lambda m, k, n: (k, n), lambda m, k, n: (m, n))
                ),
            ),
            [(1, 64, "k"), (2, 16, "n"), (0, 32, "m")],
        ),
        # A batched MatMul iterates (h, m, n, k): the heads lead and the kernel never sees them.
        (
            _gemm_node(
                4,
                tuple(
                    AffineMap.from_callable(f)
                    for f in (lambda h, m, n, k: (h, m, k), lambda h, m, n, k: (h, k, n), lambda h, m, n, k: (h, m, n))
                ),
            ),
            [(3, 64, "k"), (2, 16, "n"), (1, 32, "m")],
        ),
    ],
    ids=["gemm", "batched_matmul"],
)
def test_a_kernel_binds_the_library_dimensions_to_its_node_dimensions(node, expected):
    """By what each dimension does, the output's rows and columns and the contraction, not by
    where a parser happened to put it."""
    gemm = AIE_KERNELS["gemm"](m=32, k=64, n=16, layout="default", library=LIBRARY)
    assert [(position, size, d.name) for position, size, d in gemm.call_tile(node)] == expected
    assert gemm.spec.symbol == "matmul_bf16_bf16"


def test_a_shape_the_library_does_not_compile_is_rejected():
    with pytest.raises(ValueError, match="m=64"):
        AIE_KERNELS["gemm"](m=64, k=64, n=16, layout="default", library=LIBRARY).validate()


def test_a_kernel_without_a_library_says_so():
    with pytest.raises(ValueError, match="needs a kernel library"):
        AIE_KERNELS["gemm"](m=32, k=64, n=16, layout="default").validate()


def _gemms(library, *shapes):
    """A device with one GEMM per tile shape on cores of its own, rewritten the way codegen rewrites it."""
    consume = IntegerAttr.from_int_and_width(aie.ObjectFifoPortEnum.Consume.get_int(), 32)
    produce = IntegerAttr.from_int_and_width(aie.ObjectFifoPortEnum.Produce.get_int(), 32)
    ops, kernels = [], {}
    for column, (m, k, n) in enumerate(shapes):
        gemm = AIE_KERNELS["gemm"](m=m, k=k, n=n, layout="default", library=library)
        acquires = [aie.ObjectFifoAcquireOp(consume, 1, "operand", shape, bf16) for shape in ((m, k), (k, n))]
        tile, memory = aie.TileOp(column, 2), aie.TileOp(column, 1)
        result = aie.ObjectFifoOp.from_referenced_type(tile, [memory], f"result_{column}", 2, bf16, (m, n))
        acquires.append(aie.ObjectFifoAcquireOp(produce, 1, f"result_{column}", (m, n), bf16))
        release = aie.ObjectFIFOReleaseOp(produce, IntegerAttr.from_int_and_width(1, 32), f"result_{column}")
        space = stream.StrensorSpace(
            tuple(stream.StrensorVar(stream.StrensorVarType.KERNEL, size, LayerDim(d)) for d, size in enumerate((m, n)))
        )
        node = stream.ComputationNodeOp(
            [a.results[0] for a in acquires], (stream.StrensorType(bf16, space),), gemm.unique_name
        )
        ops += [tile, memory, result, aie.CoreOp(None, tile, Region(Block([*acquires, node, release, aie.EndOp()])))]
        kernels[gemm.unique_name] = gemm
    device = aie.DeviceOp(
        IntegerAttr.from_int_and_width(aie.AIEDeviceEnum.npu2.get_int(), 32), Region(Block([*ops, aie.EndOp()]))
    )
    bindings = Bindings("npu2")
    PatternRewriteWalker(ConvertAIEKernels(kernels, bindings)).rewrite_module(ModuleOp([device]))
    return device, bindings


def _declarations(device):
    return {op.sym_name.data: op.attributes["link_with"].data for op in device.walk() if isinstance(op, FuncOp)}


@pytest.mark.parametrize(
    "operands, message",
    [
        (["a", "b"], "takes 3 arguments, but stream passes 2"),
        (["a", "a", "c"], r"as argument 1, not memref<4x8xi32>"),
        (["a", "b", "f32"], r"as argument 2, not memref<4x2xf32>"),
    ],
    ids=["arity", "element_count", "dtype"],
)
def test_a_call_the_binding_does_not_declare_is_rejected_by_name(operands, message):
    """mlir-aie added an argument to matmul_PV once and the NPU silently computed garbage, so a call
    has to match its provider's declaration in arity, element count and element type."""
    types = {"a": MemRefType(i32, (4, 8)), "b": MemRefType(i32, (8, 2)), "c": MemRefType(i32, (4, 2))}
    types["f32"] = MemRefType(f32, (4, 2))
    declared = [np.ndarray[(size,), np.dtype[np.int32]] for size in (32, 16, 8)]
    binding = SimpleNamespace(name="x_matmul", arg_types=lambda: declared)
    check("matmul", binding, [types[name] for name in ("a", "b", "c")])
    with pytest.raises(ValueError, match=rf"^kernel matmul \(x_matmul\) .*{message}"):
        check("matmul", binding, [types[name] for name in operands])


def test_a_kernel_no_provider_binds_is_declared_on_the_library_object():
    """The call's declaration names its object, which is what links it, rather than the core."""
    device, _ = _gemms(LIBRARY, (32, 64, 16))
    assert _declarations(device) == {"zero_bf16": "mm.o", "matmul_bf16_bf16": "mm.o"}
    assert not any(isinstance(op, aie.CoreOp) and op.link_with for op in device.walk())


def test_two_gemm_tiles_in_one_design_link_two_objects():
    """Each tile shape compiles an object of its own, so its symbols must be its own too: its float sum's zeroing, the
    GEMM and the cast of the sum to its output."""
    pytest.importorskip("aie.iron.kernels", reason="the library binds the GEMM through mlir-aie")
    device, _ = _gemms(SHIPPED, (32, 64, 32), (64, 64, 64))
    declared = _declarations(device)
    assert len(declared) == 6 and len(set(declared.values())) == 6
    assert sum(symbol.endswith("_matmul_bf16_f32") for symbol in declared) == 2
    assert sum(symbol.endswith("_cast_f32_bf16_row") for symbol in declared) == 2


def test_the_recorded_bindings_rebuild_the_objects_the_design_links(tmp_path):
    """A host that finds the design cached rebuilds its kernel objects from kernels.json alone."""
    pytest.importorskip("aie.iron.kernels", reason="the library binds the GEMM through mlir-aie")
    device, bindings = _gemms(SHIPPED, (32, 64, 32), (64, 64, 64))
    bindings.write(tmp_path / "kernels.json")
    rebuilt = load_bindings(tmp_path / "kernels.json")
    assert {(b.name, b.object_file_name) for b in rebuilt} == set(_declarations(device).items())


def test_silu_binds_mlir_aies_lut_build_of_the_sized_entry_point():
    """SiLU computes tanh from the interpolated table, as upstream IRON's SiLU does, through ``(in, out, size)``."""
    pytest.importorskip("aie.iron.kernels", reason="the library binds SiLU through mlir-aie")
    from ml_dtypes import bfloat16

    from stream.compiler.kernels.mlir_aie import silu

    binding = silu("npu2", 32, 64)
    tile = np.ndarray[(2048,), np.dtype[bfloat16]]
    assert binding.name.endswith("_silu_bf16_size")
    assert binding.arg_types() == [tile, tile, np.int32]
    assert "-DACTIVATIONS_TANH_LUT=1" in binding.compile_flags


def test_a_gemm_sums_its_k_tiles_in_float_and_casts_the_sum_into_its_output():
    """As IRON's GEMMs do: the core zeroes a float tile of its own, adds each call into it, and only then acquires the
    output, casts the sum into it and sends it on, so its side of the output fifo holds one tile."""
    pytest.importorskip("aie.iron.kernels", reason="the library binds the GEMM through mlir-aie")
    device, _ = _gemms(SHIPPED, (64, 64, 64))
    (core,) = [op for op in device.walk() if isinstance(op, aie.CoreOp)]
    steps = [
        op.callee.string_value().rsplit("_", 3)[-1] if isinstance(op, CallOp) else op.name
        for op in core.walk()
        if isinstance(op, CallOp | aie.ObjectFIFOReleaseOp)
    ]
    assert steps == ["zero", "f32", "row", "aie.objectfifo.release"]
    acquires = [op for op in core.walk() if isinstance(op, aie.ObjectFifoAcquireOp)]
    acquired = [op for op in acquires if op.objFifo_name.string_value() == "result_0"]
    (cast_call,) = [op for op in core.walk() if isinstance(op, CallOp) and op.callee.string_value().endswith("_row")]
    assert [a.results[0] for a in acquired] == [cast_call.arguments[1]]
    (result,) = [op for op in device.walk() if isinstance(op, aie.ObjectFifoOp)]
    assert result.elemNumber.data[0].value.data == 1
    (accumulator,) = [op for op in device.walk() if isinstance(op, aie.BufferOp)]
    assert accumulator.buffer.type == MemRefType(f32, (64 * 64,))


def test_the_output_tiles_a_core_picks_between_keep_their_kernel_layout():
    """Eight output tiles held across a reduction, each cast to the kernel's tiled layout and picked by a switch: the
    tile acquired to send a sum on is written in that layout too."""
    from snaxc.dialects.snax import LayoutCast
    from snaxc.dialects.tsl import TiledStridedLayoutAttr
    from xdsl.dialects.builtin import DenseArrayBase, IntegerType
    from xdsl.dialects.scf import IndexSwitchOp, YieldOp

    from stream.compiler.kernels.aie_kernel import tiled_layout
    from stream.compiler.kernels.gemm import _layout

    produce = IntegerAttr.from_int_and_width(aie.ObjectFifoPortEnum.Produce.get_int(), 32)
    acquire = aie.ObjectFifoAcquireOp(produce, 2, "result", (32, 32), bf16)
    laid_out = MemRefType(bf16, (32, 32), layout=TiledStridedLayoutAttr(tiled_layout(32, 32, 8, 8)))
    casts = [LayoutCast(result, laid_out) for result in acquire.results]
    index = aie.TileOp(0, 2)
    switch = IndexSwitchOp(
        index.result,
        DenseArrayBase.from_list(IntegerType(64), [0, 1]),
        Region(Block([YieldOp(casts[0].dest)])),
        [Region(Block([YieldOp(cast.dest)])) for cast in casts],
        [laid_out],
    )
    assert _layout(switch.results[0]) == laid_out
    assert _layout(acquire.results[0]) is None
