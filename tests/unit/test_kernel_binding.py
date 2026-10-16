from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("snaxc", reason="the AIE kernels are a separate install, via stream-setup-aie")

from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp, bf16, f32, i32  # noqa: E402
from xdsl.dialects.func import FuncOp  # noqa: E402
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
    ops, kernels = [], {}
    for column, (m, k, n) in enumerate(shapes):
        gemm = AIE_KERNELS["gemm"](m=m, k=k, n=n, layout="default", library=library)
        acquires = [aie.ObjectFifoAcquireOp(consume, 1, "operand", shape, bf16) for shape in ((m, k), (k, n), (m, n))]
        space = stream.StrensorSpace(
            tuple(stream.StrensorVar(stream.StrensorVarType.KERNEL, size, LayerDim(d)) for d, size in enumerate((m, n)))
        )
        node = stream.ComputationNodeOp(
            [a.results[0] for a in acquires], (stream.StrensorType(bf16, space),), gemm.unique_name
        )
        tile = aie.TileOp(column, 2)
        ops += [tile, aie.CoreOp(None, tile, Region(Block([*acquires, node, aie.EndOp()])))]
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
    """Each tile shape compiles an object of its own, so its symbols must be its own too."""
    pytest.importorskip("aie.iron.kernels", reason="the library binds the GEMM through mlir-aie")
    device, _ = _gemms(SHIPPED, (32, 64, 32), (64, 64, 64))
    declared = _declarations(device)
    assert len(declared) == 4 and len(set(declared.values())) == 4
    assert sum(symbol.endswith("_matmul_bf16_bf16") for symbol in declared) == 2


def test_the_recorded_bindings_rebuild_the_objects_the_design_links(tmp_path):
    """A host that finds the design cached rebuilds its kernel objects from kernels.json alone."""
    pytest.importorskip("aie.iron.kernels", reason="the library binds the GEMM through mlir-aie")
    device, bindings = _gemms(SHIPPED, (32, 64, 32), (64, 64, 64))
    bindings.write(tmp_path / "kernels.json")
    rebuilt = load_bindings(tmp_path / "kernels.json")
    assert {(b.name, b.object_file_name) for b in rebuilt} == set(_declarations(device).items())
