"""One registry for operator decompositions -- the reuse seam for coarse-vs-fine granularity.

Some operators are *schedulable as one kernel* yet are *internally a subgraph of affine sub-operators*:
a softmax (max -> exp -> sum -> div), a flash-attention block (two matmuls + the online-softmax stats),
and whatever comes next. Different hardware processes them at different granularities -- a fused-softmax
unit takes the whole kernel; a matmul-array + vector-unit accelerator wants the sub-operators, because
each sub-op's *affine access relation* is what maps to the array (a contraction) or the vector unit (a
reduction / elementwise).

So the framework needs a single, extensible way to ask "what is this op, one level down, as affine
nodes?". A decomposer is just ``node -> Workload`` of affine sub-ops (the ordinary IR -- no new node
types), registered by op ``type``. Every consumer (the graph view, fusion analysis, a future cost pass)
calls :func:`decompose` uniformly; adding a new operator is one :func:`register_decomposition` call plus
its small builder, never a new special-case in the consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream.workload.node import ComputationNode
    from stream.workload.workload import Workload

# A decomposer expands one node into a Workload of its affine sub-operators (its dataflow, one level down).
Decomposer = Callable[["ComputationNode"], "Workload"]

_REGISTRY: dict[str, Decomposer] = {}
_LOAD_STATE: dict[str, bool] = {"builtins": False}


def register_decomposition(op_type: str, decomposer: Decomposer) -> None:
    """Register (or override) the affine sub-operator decomposition for op ``op_type`` -- the seam a new
    operator (or a private overlay) extends without touching any consumer."""
    _REGISTRY[op_type] = decomposer


def _ensure_builtins() -> None:
    """Register the in-tree decomposers. Done lazily + here (not in the decomposer modules) so those
    modules never import this one -- no import cycle, and the registry stays the single source of truth."""
    if _LOAD_STATE["builtins"]:
        return
    _LOAD_STATE["builtins"] = True
    from stream.workload.normalization import NORMALIZATION_OPS, decompose_normalization  # noqa: PLC0415
    from stream.workload.rewrites.flash_attention import decompose_attention_block  # noqa: PLC0415

    for op_type in NORMALIZATION_OPS:
        register_decomposition(op_type, decompose_normalization)
    register_decomposition("AttentionBlock", decompose_attention_block)


def has_decomposition(node: ComputationNode) -> bool:
    """Whether ``node``'s op type has a registered affine sub-operator decomposition."""
    _ensure_builtins()
    return getattr(node, "type", None) in _REGISTRY


def decompose(node: ComputationNode) -> Workload | None:
    """The affine sub-operator subgraph of ``node``, or ``None`` if its op type has no decomposition."""
    _ensure_builtins()
    decomposer = _REGISTRY.get(getattr(node, "type", None))
    return decomposer(node) if decomposer is not None else None
