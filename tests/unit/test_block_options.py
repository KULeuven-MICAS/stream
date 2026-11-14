"""A kernel's compiled block list is a candidate; a divisor only narrows what others offer."""

import pytest

pytest.importorskip("snaxc", reason="the AIE kernels are a separate install, via stream-setup-aie")

from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

from xdsl.ir.affine import AffineMap  # noqa: E402

from stream.compiler.kernels.flash import PartialSoftmaxKernel  # noqa: E402
from stream.compiler.kernels.gemm import GemmKernel  # noqa: E402
from stream.compiler.kernels.library import KernelLibrary  # noqa: E402
from stream.mapping.blocks import block_options  # noqa: E402
from stream.workload.workload import ComputationNode  # noqa: E402

M, K, N, S = "m", "k", "n", "scores"
TILED = ((M, 64), (K, 64), (N, 64), (S, 64))


def library(gemm_m_divisor=16):
    return KernelLibrary.from_dict(
        {
            "family": {
                "matmul": {"ops_per_cycle": 151.0, "mac": {"m": 8, "k": 8, "n": 8}},
                "vector": {"ops_per_cycle": 16.0},
            },
            "kernel": {
                "matmul_bf16_bf16": {
                    "family": "matmul",
                    "dims": [
                        {"name": "k", "divisor": 8},
                        {"name": "n", "divisor": 16},
                        {"name": "m", "divisor": gemm_m_divisor},
                    ],
                },
                "partial_softmax": {
                    "family": "vector",
                    "dims": [{"name": "n", "fixed": 64}, {"name": "m", "blocks": [16, 32, 64]}],
                },
            },
        }
    )


@dataclass
class _Entry:
    kernel: Any


class _Mapping:
    def __init__(self, entries):
        self._entries = entries

    def __contains__(self, node):
        return node in self._entries

    def get(self, node):
        return self._entries[node]


class _Workload:
    def __init__(self, dims):
        self._dims = dims

    def get_node_by_name(self, name):
        return next(n for n in self._dims if n.name == name)

    def get_dims(self, node):
        return self._dims[node]


@dataclass
class _Group:
    layers: tuple
    intra_core_tiling: tuple


# The iteration spaces the Gemm and elementwise parsers give: (m, k, n) and (m, n).
GEMM_MAPS = tuple(
    AffineMap.from_callable(f) for f in (lambda m, k, n: (m, k), lambda m, k, n: (k, n), lambda m, k, n: (m, n))
)
ELEMENTWISE_MAPS = (AffineMap.identity(2), AffineMap.identity(2))


def _node(name, maps):
    return ComputationNode(type=name, name=name, inputs=(), outputs=(), operand_mapping=maps)


def _attention(lib):
    gemm, softmax = _node("Attn_QK", GEMM_MAPS), _node("Attn_Softmax", ELEMENTWISE_MAPS)
    workload = _Workload({gemm: (M, K, N), softmax: (M, S)})
    mapping = _Mapping(
        {
            gemm: _Entry(GemmKernel(m=64, k=64, n=64, layout="default", library=lib)),
            softmax: _Entry(PartialSoftmaxKernel(m=64, n=64, layout="contiguous", library=lib)),
        }
    )
    return workload, mapping, _Group(layers=("Attn_QK", "Attn_Softmax"), intra_core_tiling=TILED)


def test_a_group_whose_only_block_rule_is_a_divisor_offers_no_candidates():
    gemm = _node("Gemm_Left", GEMM_MAPS)
    workload = _Workload({gemm: (M, K, N)})
    mapping = _Mapping({gemm: _Entry(GemmKernel(m=64, k=64, n=64, layout="default", library=library()))})
    assert block_options(workload, mapping, _Group(layers=("Gemm_Left",), intra_core_tiling=TILED)) == {}


def test_a_kernel_generic_over_a_dimension_does_not_veto_the_block_another_offers():
    assert block_options(*_attention(library())) == {M: (16, 32, 64)}


def test_a_divisor_narrows_what_another_kernel_offers():
    assert block_options(*_attention(library(gemm_m_divisor=32))) == {M: (32, 64)}


def test_a_group_is_offered_each_call_its_kernel_library_measures():
    """A GEMM built at 64x64x64 is offered the library's other measured calls that divide its dims, not a call whose
    shape the kernel cannot build."""
    from stream.mapping.blocks import measured_calls

    lib = KernelLibrary.from_dict(
        {
            "family": {"matmul": {"ops_per_cycle": 151.0, "mac": {"m": 8, "k": 8, "n": 8}}},
            "kernel": {
                "matmul_bf16_f32": {
                    "family": "matmul",
                    "binding": "stream.compiler.kernels.mlir_aie:mm",
                    "dims": [{"name": "k", "divisor": 8}, {"name": "n", "divisor": 16}, {"name": "m", "divisor": 16}],
                    "cycles": [
                        {"m": 64, "k": 64, "n": 64, "cycles": 1620.0},
                        {"m": 32, "k": 128, "n": 64, "cycles": 1267.0},
                        {"m": 32, "k": 64, "n": 24, "cycles": 600.0},
                        {"m": 32, "k": 256, "n": 32, "cycles": 1099.0},
                    ],
                }
            },
        }
    )
    gemm = _node("Gemm", GEMM_MAPS)
    workload = _Workload({gemm: (M, K, N)})
    workload.get_dimension_size = lambda d: {M: 512, K: 128, N: 512}[d]
    mapping = _Mapping({gemm: _Entry(GemmKernel(m=64, k=64, n=64, layout="default", library=lib))})
    group = _Group(layers=("Gemm",), intra_core_tiling=((K, 64), (N, 64), (M, 64)))
    assert measured_calls(workload, mapping, group) == [{K: 128, N: 64, M: 32}]
    assert mapping.get(gemm).kernel.library_name == "matmul_bf16_f32"
    assert GemmKernel(m=64, k=64, n=64, layout="default", library=library()).library_name == "matmul_bf16_bf16"


def test_a_call_is_bound_by_the_tiles_its_core_streams_in():
    """A 32x128x64 call fetches 196608 bits of operand tiles, 3072 cycles over a 64-bit link against its 1267 of
    compute; a 32-row call that holds four output tiles fetches the weight tile once per four calls, and needs 60 KB
    of the core's memory: two of each operand tile, four float accumulators and its output tile."""
    from types import SimpleNamespace

    from stream.mapping.blocks import call_bound

    lib = KernelLibrary.from_dict(
        {
            "family": {"matmul": {"ops_per_cycle": 151.0, "mac": {"m": 8, "k": 8, "n": 8}}},
            "kernel": {
                "matmul_bf16_f32": {
                    "family": "matmul",
                    "binding": "stream.compiler.kernels.mlir_aie:mm",
                    "dims": [{"name": "k", "divisor": 8}, {"name": "n", "divisor": 16}, {"name": "m", "divisor": 16}],
                    "cycles": [
                        {"m": 32, "k": 128, "n": 64, "cycles": 1267.0},
                        {"m": 32, "k": 64, "n": 64, "cycles": 815.0},
                    ],
                }
            },
        }
    )

    class _Tensor:
        operand_type = SimpleNamespace(bitwidth=16)

    x, w, y = _Tensor(), _Tensor(), _Tensor()
    gemm = ComputationNode(type="Gemm", name="Gemm", inputs=(), outputs=(), operand_mapping=GEMM_MAPS)
    gemm.__dict__["inputs"], gemm.__dict__["outputs"] = (x, w), (y,)
    workload = _Workload({gemm: (M, K, N)})
    indexed = {id(x): (M, K), id(w): (K, N), id(y): (M, N)}
    workload.strides_for_tensor = lambda t: {d: (1,) for d in indexed[id(t)]}
    wide = _Mapping({gemm: _Entry(GemmKernel(m=32, k=128, n=64, layout="default", library=lib))})
    group = _Group(layers=("Gemm",), intra_core_tiling=((K, 128), (N, 64), (M, 32)))
    assert call_bound(workload, wide, group, 64) == 3072 / (32 * 128 * 64)
    held = _Mapping({gemm: _Entry(GemmKernel(m=32, k=64, n=64, layout="default", library=lib))})
    group = _Group(layers=("Gemm",), intra_core_tiling=((M, 32), (K, 64), (N, 64), (M, 128)))
    assert call_bound(workload, held, group, 64) == 815 / (32 * 64 * 64)
    assert call_bound(workload, held, group, 64, capacity_bits=63 * 8192) == 815 / (32 * 64 * 64)
    assert call_bound(workload, held, group, 64, capacity_bits=56 * 8192) == float("inf")


def test_a_call_takes_the_innermost_tile_of_a_dim_and_keeps_its_larger_ones():
    """A GEMM whose k is staged at 512 around 64-wide calls takes a 128-wide call inside the same 512 block; a call
    that does not divide the block is not offered."""
    from stream.mapping.blocks import with_call

    gemm = _node("Gemm", GEMM_MAPS)
    workload = _Workload({gemm: (M, K, N)})

    class _Groups(_Mapping):
        def __init__(self, entries, groups):
            super().__init__(entries)
            self.fused_groups = groups

        def copy(self):
            return _Groups(dict(self._entries), list(self.fused_groups))

        def set(self, node, entry):
            self._entries[node] = entry

        def with_fused_groups(self, groups):
            return _Groups(dict(self._entries), list(groups))

    lib = library()
    group = _Group(layers=("Gemm",), intra_core_tiling=((K, 64), (K, 512), (N, 64), (M, 64)))
    mapping = _Groups({gemm: _Entry(GemmKernel(m=64, k=64, n=64, layout="default", library=lib))}, [group])
    called = with_call(workload, mapping, 0, group, {K: 128, N: 64, M: 32})
    assert called.fused_groups[0].intra_core_tiling == ((K, 128), (K, 512), (N, 64), (M, 32))
    assert (called.get(gemm).kernel.k, called.get(gemm).kernel.m) == (128, 32)
    assert with_call(workload, mapping, 0, group, {K: 96, N: 64, M: 32}) is None
