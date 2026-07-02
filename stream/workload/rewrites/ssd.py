"""Mamba2 SSD (state-space dual) -> chunked blockwise form (see reference.chunked_ssd)."""

from __future__ import annotations

from stream.workload.node import ComputationNode
from stream.workload.rewrites.base import RewriteParams, build_chunk_chain
from stream.workload.workload import Workload


class SSDRewrite:
    name = "ssd"
    source_type = "SSD"
    chunk_type = "SSDChunk"

    def matches(self, node: ComputationNode) -> bool:
        return node.type == self.source_type

    def apply(self, node: ComputationNode, params: RewriteParams) -> Workload:
        seq_len, hidden = node.inputs[0].shape
        return build_chunk_chain(
            node.name, seq_len, hidden, params.chunk_size, node.inputs[0].operand_type, self.chunk_type
        )
