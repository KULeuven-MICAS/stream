"""Which block sizes a fused group offers the search, and which it merely tolerates.

A kernel library says what it is built at two ways. ``blocks`` is a list of sizes somebody
compiled and timed. ``divisor`` is the source's own legality rule -- mm.cc's
``static_assert(k % 8 == 0)`` -- and says nothing about what was ever built.

Conflating them is not academic. The fused SwiGLU's only block information is mm.cc's
divisor; read as an offer it handed the search k in (8, 16, 32, 64), and the per-operation
anchor prices those at equal cycles while making each transfer smaller, so it took k=8. The
512-deep reduction then ran as 64 rounds accumulating into a bf16 buffer: on device 4.6x
slower and 77% of elements outside tolerance, against 14.5% at the declared block.
"""

import pytest

pytest.importorskip("snaxc", reason="the AIE kernels are a separate install, via stream-setup-aie")

from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

from stream.compiler.kernels import manifest  # noqa: E402
from stream.compiler.kernels.flash import PartialSoftmaxKernel  # noqa: E402
from stream.compiler.kernels.gemm import GemmKernel  # noqa: E402
from stream.mapping.blocks import block_options  # noqa: E402
from stream.workload.workload import ComputationNode  # noqa: E402

SHAPE = dict(element_type="bf16", bfp16_mmul=True, utilization=61.8, layout="default")
M, K, N, S = "m", "k", "n", "scores"

LIBRARY = {
    # As the real manifest declares mm.cc: a legality rule, no list of builds.
    "matmul_bf16_bf16": {
        "divisor": {"m": 16, "k": 8, "n": 16},
        "cycles": {"64,64,64": 1595.0, "per_op": {"cycles": 1595.0, "ops": 262144}},
    },
    # As the real manifest declares mha.cc: the query blocks it is actually built for.
    "partial_softmax": {"blocks": {"m": [16, 32, 64]}, "cycles": {"32,64": 2323.0}},
}


@pytest.fixture
def library():
    manifest.adopt(LIBRARY)
    yield
    manifest.adopt({})


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

    def get_dimension_size(self, dim):
        return 512


@dataclass
class _Group:
    layers: tuple
    intra_core_tiling: tuple


def _node(name):
    return ComputationNode(type=name, name=name, inputs=(), outputs=(), operand_mapping=())


TILED = ((M, 64), (K, 64), (N, 64), (S, 64))


def test_a_group_whose_only_block_rule_is_a_divisor_offers_no_candidates(library):
    """The fused SwiGLU. mm.cc takes any multiple of 8; that is not a menu to search.

    Its declared block comes from the caller, which enumerates the blocks it built kernel
    variants for. A search that invents a finer one picks a kernel nobody compiled.
    """
    gemm = _node("Gemm_Left")
    workload = _Workload({gemm: (M, K, N)})
    mapping = _Mapping({gemm: _Entry(GemmKernel(m=64, k=64, n=64, **SHAPE))})
    group = _Group(layers=("Gemm_Left",), intra_core_tiling=TILED)

    assert block_options(workload, mapping, group) == {}


def test_a_kernel_generic_over_a_dimension_does_not_veto_the_block_another_offers(library):
    """Attention. mha.cc names the query blocks it is built for; the QK matmul follows.

    Making the divisor silent must not make it a veto: the 32-vs-64 query block search is
    the whole point of the attention study, and it lives on a dimension mm.cc shares.
    """
    gemm, softmax = _node("Attn_QK"), _node("Attn_Softmax")
    workload = _Workload({gemm: (M, K, N), softmax: (M, S)})
    mapping = _Mapping({
        gemm: _Entry(GemmKernel(m=64, k=64, n=64, **SHAPE)),
        softmax: _Entry(PartialSoftmaxKernel(m=64, n=64, **{**SHAPE, "layout": "contiguous"})),
    })
    group = _Group(layers=("Attn_QK", "Attn_Softmax"), intra_core_tiling=TILED)

    assert block_options(workload, mapping, group) == {M: (16, 32, 64)}


def test_a_divisor_narrows_what_another_kernel_offers(library):
    """The rule still binds: a block below mm.cc's floor is not legal for the group."""
    manifest.adopt({**LIBRARY, "matmul_bf16_bf16": {"divisor": {"m": 32, "k": 8, "n": 16}}})
    gemm, softmax = _node("Attn_QK"), _node("Attn_Softmax")
    workload = _Workload({gemm: (M, K, N), softmax: (M, S)})
    mapping = _Mapping({
        gemm: _Entry(GemmKernel(m=64, k=64, n=64, **SHAPE)),
        softmax: _Entry(PartialSoftmaxKernel(m=64, n=64, **{**SHAPE, "layout": "contiguous"})),
    })
    group = _Group(layers=("Attn_QK", "Attn_Softmax"), intra_core_tiling=TILED)

    assert block_options(workload, mapping, group) == {M: (32, 64)}
