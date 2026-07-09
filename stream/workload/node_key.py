"""Canonical, content-addressed identity for a computation node: a blake2 digest over op type and
per-operand precision/shape/affine map, with iteration dims relabelled by first-use order; stable
across processes so it can key a shared cost cache. Imports no other ``stream.workload`` module so
``node.py`` can depend on it.
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
