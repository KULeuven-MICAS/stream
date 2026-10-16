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
