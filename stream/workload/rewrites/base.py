"""Rewrite protocol and the shared chunk-chain builder."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Protocol, runtime_checkable

from xdsl.dialects.builtin import FixedBitwidthType
from xdsl.ir.affine import AffineMap

from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload


@dataclass(frozen=True)
class RewriteParams:
    """Parameters for a rewrite. ``chunk_size`` is the DSE lever (swept by an outer stage)."""

    chunk_size: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive int, got {self.chunk_size!r}")


@runtime_checkable
class Rewrite(Protocol):
    name: str

    def matches(self, node: ComputationNode) -> bool: ...

    def apply(self, node: ComputationNode, params: RewriteParams) -> Workload: ...


def build_chunk_chain(
    base_name: str, seq_len: int, hidden: int, chunk_size: int, dtype: FixedBitwidthType, chunk_type: str
) -> Workload:
    """Chain of ``ceil(seq_len/chunk_size)`` per-chunk reduction nodes: ``c`` (chunk-local seq) is
    REDUCTION, ``d`` (hidden) is PARALLEL; consecutive chunks share the state tensor."""
    n_chunks = ceil(seq_len / chunk_size)
    x_map = AffineMap.from_callable(lambda c, d: (c, d))
    state_map = AffineMap.from_callable(lambda c, d: (d,))

    nodes: list = []
    initial_state = Tensor.create(f"{base_name}_h_init", dtype, (hidden,))
    nodes.append(InEdge(name=f"{base_name}_h_init", outputs=(initial_state,)))

    prev_state = initial_state
    for i in range(n_chunks):
        this_chunk = min(chunk_size, seq_len - i * chunk_size)
        x_chunk = Tensor.create(f"{base_name}_x{i}", dtype, (this_chunk, hidden))
        h_out = Tensor.create(f"{base_name}_h{i}", dtype, (hidden,))
        nodes.append(InEdge(name=f"{base_name}_x{i}", outputs=(x_chunk,)))
        nodes.append(
            ComputationNode(
                type=chunk_type,
                name=f"{base_name}_chunk{i}",
                inputs=(x_chunk, prev_state),
                outputs=(h_out,),
                operand_mapping=(x_map, state_map, state_map),
            )
        )
        prev_state = h_out

    nodes.append(OutEdge(name=f"{base_name}_out", inputs=(prev_state,)))
    return Workload(nodes)


@dataclass(frozen=True)
class ChunkRewrite:
    """A sequence-mixing op decomposed into a chunk chain. All chunked recurrences share the graph
    shape :func:`build_chunk_chain` builds; only ``source_type`` (matched op) and the emitted
    ``chunk_type`` differ -- the intra-chunk math lives in the cost model."""

    name: str
    source_type: str
    chunk_type: str

    def matches(self, node: ComputationNode) -> bool:
        return node.type == self.source_type

    def apply(self, node: ComputationNode, params: RewriteParams) -> Workload:
        seq_len, hidden = node.inputs[0].shape
        return build_chunk_chain(
            node.name, seq_len, hidden, params.chunk_size, node.inputs[0].operand_type, self.chunk_type
        )
