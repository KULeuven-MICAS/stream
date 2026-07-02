"""Canonical, content-addressed identity for a computation node.

The key is a blake2 digest over the node's *mapping-relevant identity*: op type, and per operand its
precision, shape, and affine map (with iteration dimensions relabelled by first-use order, so a node
is invariant under a consistent renaming of its loop dimensions). It is deliberately sensitive to op
type and the affine maps -- the previous shapes-and-precision-only equality collided a Conv and a
Gemm with matching tensor shapes (AUDIT.md §4). Being a content hash of a canonical byte encoding,
it is stable across processes and machines, so it can key an on-disk or shared cost cache.

This module imports no other ``stream.workload`` module, so ``node.py`` can depend on it.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr, AffineExpr

if TYPE_CHECKING:
    from stream.workload.node import ComputationNode

_BINARY_OP: dict[AffineBinaryOpKind, str] = {
    AffineBinaryOpKind.Add: "+",
    AffineBinaryOpKind.Mul: "*",
    AffineBinaryOpKind.Mod: "%",
    AffineBinaryOpKind.FloorDiv: "//",
    AffineBinaryOpKind.CeilDiv: "^/",
}


def _expr_str(expr: AffineExpr, canon) -> str:
    if isinstance(expr, AffineDimExpr):
        return f"d{canon(expr.position)}"
    if isinstance(expr, AffineConstantExpr):
        return f"c{expr.value}"
    if isinstance(expr, AffineBinaryOpExpr):
        return f"({_expr_str(expr.lhs, canon)}{_BINARY_OP[expr.kind]}{_expr_str(expr.rhs, canon)})"
    return f"?{type(expr).__name__}"


def canonical_form(node: ComputationNode) -> str:
    """Deterministic string identity of ``node`` under canonical dimension relabelling."""
    relabel: dict[int, int] = {}

    def canon(position: int) -> int:
        if position not in relabel:
            relabel[position] = len(relabel)
        return relabel[position]

    parts = [f"type={node.type}"]
    for i, tensor in enumerate(node.tensors):
        results = ",".join(_expr_str(result, canon) for result in node.get_mapping(tensor).results)
        parts.append(f"o{i}|w{tensor.operand_type.bitwidth}|s{tuple(int(s) for s in tensor.shape)}|m[{results}]")
    return ";".join(parts)


def node_key(node: ComputationNode) -> str:
    """16-byte blake2 hex digest of :func:`canonical_form` -- the cache/dedup key for ``node``."""
    return hashlib.blake2b(canonical_form(node).encode(), digest_size=16).hexdigest()
