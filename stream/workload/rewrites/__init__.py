"""Canonical decomposition (rewrite) library.

Each rewrite matches a monolithic sequence-mixing op (by ``type``) and rewrites it into a chunked
subgraph via :func:`build_chunk_chain`, with chunk size as a DSE parameter. Adding a new rewrite is
a one-file change plus its reference math -- register it here.
"""

from __future__ import annotations

from stream.workload.node import ComputationNode
from stream.workload.rewrites.base import Rewrite, RewriteParams, build_chunk_chain
from stream.workload.rewrites.chunked_scan import ChunkedScanRewrite
from stream.workload.rewrites.gated_deltanet import GatedDeltaNetRewrite
from stream.workload.rewrites.ssd import SSDRewrite
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


for _rewrite in (ChunkedScanRewrite(), SSDRewrite(), GatedDeltaNetRewrite()):
    register_rewrite(_rewrite)
