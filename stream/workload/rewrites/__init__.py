"""Registry of rewrites: chunked decompositions of sequence-mixing ops, and rewrites of patterns in a whole workload."""

from __future__ import annotations

from stream.workload.node import ComputationNode
from stream.workload.rewrites.base import ChunkRewrite, Rewrite, RewriteParams, WorkloadRewrite, build_chunk_chain
from stream.workload.rewrites.flash_attention import FlashAttentionRewrite
from stream.workload.rewrites.flash_decoding import FlashDecodingRewrite
from stream.workload.workload import Workload

__all__ = [
    "Rewrite",
    "RewriteParams",
    "WorkloadRewrite",
    "build_chunk_chain",
    "register_rewrite",
    "get_rewrite",
    "get_workload_rewrite",
    "registered_rewrites",
    "apply_rewrites",
]

_REGISTRY: dict[str, Rewrite] = {}
_WORKLOAD_REGISTRY: dict[str, WorkloadRewrite] = {}


def register_rewrite[R: Rewrite | WorkloadRewrite](rewrite: R) -> R:
    """Register a rewrite by its ``name`` (idempotent overwrite)."""
    if isinstance(rewrite, WorkloadRewrite):
        _WORKLOAD_REGISTRY[rewrite.name] = rewrite
    else:
        _REGISTRY[rewrite.name] = rewrite
    return rewrite


def get_rewrite(name: str) -> Rewrite:
    return _REGISTRY[name]


def get_workload_rewrite(name: str) -> WorkloadRewrite:
    return _WORKLOAD_REGISTRY[name]


def registered_rewrites() -> list[str]:
    return sorted([*_REGISTRY, *_WORKLOAD_REGISTRY])


def apply_rewrites(node: ComputationNode, params: RewriteParams) -> Workload | None:
    """Apply the first rewrite that matches ``node``; return None if none match."""
    for rewrite in _REGISTRY.values():
        if rewrite.matches(node):
            return rewrite.apply(node, params)
    return None


# Chunked recurrences: identical graph shape, differing only in matched op and emitted chunk type.
_CHUNK_REWRITES = (
    ChunkRewrite("chunked_scan", "Scan", "ScanChunk"),
    ChunkRewrite("ssd", "SSD", "SSDChunk"),
    ChunkRewrite("gated_deltanet", "GatedDeltaNet", "DeltaNetChunk"),
)
for _rewrite in (*_CHUNK_REWRITES, FlashAttentionRewrite(), FlashDecodingRewrite()):
    register_rewrite(_rewrite)
