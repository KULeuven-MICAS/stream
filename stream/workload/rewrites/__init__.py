"""Registry of chunked-decomposition rewrites for sequence-mixing ops."""

from __future__ import annotations

from stream.workload.node import ComputationNode
from stream.workload.rewrites.base import ChunkRewrite, Rewrite, RewriteParams, build_chunk_chain
from stream.workload.rewrites.flash_attention import FlashAttentionRewrite
from stream.workload.workload import Workload

__all__ = [
    "Rewrite",
    "RewriteParams",
    "build_chunk_chain",
    "register_rewrite",
    "get_rewrite",
    "registered_rewrites",
    "apply_rewrites",
]

_REGISTRY: dict[str, Rewrite] = {}


def register_rewrite(rewrite: Rewrite) -> Rewrite:
    """Register a rewrite by its ``name`` (idempotent overwrite)."""
    _REGISTRY[rewrite.name] = rewrite
    return rewrite


def get_rewrite(name: str) -> Rewrite:
    return _REGISTRY[name]


def registered_rewrites() -> list[str]:
    return sorted(_REGISTRY)


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
for _rewrite in (*_CHUNK_REWRITES, FlashAttentionRewrite()):
    register_rewrite(_rewrite)
