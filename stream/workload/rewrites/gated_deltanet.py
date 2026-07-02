"""Gated DeltaNet -> chunked WY form (see reference.chunked_deltanet)."""

from __future__ import annotations

from stream.workload.node import ComputationNode
from stream.workload.rewrites.base import RewriteParams, build_chunk_chain
from stream.workload.workload import Workload


class GatedDeltaNetRewrite:
    name = "gated_deltanet"
    source_type = "GatedDeltaNet"
    chunk_type = "DeltaNetChunk"

    def matches(self, node: ComputationNode) -> bool:
        return node.type == self.source_type

    def apply(self, node: ComputationNode, params: RewriteParams) -> Workload:
        seq_len, hidden = node.inputs[0].shape
        return build_chunk_chain(
            node.name, seq_len, hidden, params.chunk_size, node.inputs[0].operand_type, self.chunk_type
        )
