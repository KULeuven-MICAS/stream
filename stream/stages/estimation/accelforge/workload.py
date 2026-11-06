"""Stream computation nodes as AccelForge Einsums: one Einsum over the node's dims (``<einsum>_d<i>``), a rank
``R<axis>_<tensor>`` per tensor axis, and the renames ``input``, ``weight``, ``output`` for what a ZigZag core serves as
``I1``, ``I2``, ``O``, with ``partial_sums`` for a MAC node's output."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr, AffineExpr

from stream.workload.utils import is_mac_operator_type, operand_inputs

if TYPE_CHECKING:
    from stream.workload.node import ComputationNode
    from stream.workload.tensor import Tensor

ROLE_OF_OPERAND: dict[str, str] = {"I1": "input", "I2": "weight", "O": "output"}
"""The rename an arch uses for the tensor a ZigZag memory operand holds."""
PARTIAL_SUMS = "partial_sums"
NOTHING = "Nothing"
RESERVED_NAMES = frozenset(
    {"All", NOTHING, "Inputs", "Outputs", "Intermediates", PARTIAL_SUMS, *ROLE_OF_OPERAND.values()}
)
"""Names AccelForge set expressions and the converters' renames use, so no tensor or component may take them."""
ZIGZAG_INPUTS = 2
"""Inputs a ZigZag core serves (``I1``, ``I2``), and those it prices of a MAC node."""


class UnsupportedConversion(ValueError):  # noqa: N818 -- a refusal, not a failure of the converter
    """A Stream node or core AccelForge cannot express."""


@dataclass(frozen=True)
class Access:
    """How an Einsum reads or writes one tensor: each rank's projection as ISL text."""

    name: str
    projection: tuple[tuple[str, str], ...]
    output: bool
    bits_per_value: int


@dataclass(frozen=True)
class EinsumDescription:
    """An Einsum as plain data, which the host builds and checks without importing accelforge."""

    name: str
    accesses: tuple[Access, ...]
    sizes: tuple[int, ...]
    renames: tuple[tuple[str, str], ...]


def describe(node: ComputationNode, dim_sizes: Sequence[int]) -> EinsumDescription:
    """``node`` as an Einsum over ``dim_sizes``, the extent of each of its dims on the core that runs it, with the
    inputs Stream's ZigZag estimator prices (``operand_inputs``). Raises :class:`UnsupportedConversion` for a node
    AccelForge cannot express."""
    if len(dim_sizes) != node.num_dims:
        raise UnsupportedConversion(f"{node.name}: {len(dim_sizes)} dim sizes given for {node.num_dims} dims")
    name = _identifier(node.name)
    variables = rank_variables(name, dim_sizes)
    priced = operand_inputs(node)
    operands = [(t, m, False) for t, m in zip(node.inputs, node.operand_mapping, strict=False)][: len(priced)]
    outputs = [(t, m, True) for t, m in zip(node.outputs, node.operand_mapping[len(node.inputs) :], strict=True)]
    accesses: dict[str, Access] = {}
    owners: dict[str, Tensor] = {}
    for tensor, affine_map, output in [*operands, *outputs]:
        access = _access(node.name, tensor, affine_map.results, variables, dim_sizes, output)
        if (owner := owners.setdefault(access.name, tensor)) != tensor:
            raise UnsupportedConversion(
                f"{node.name}: tensors {owner.name} and {tensor.name} both become {access.name}"
            )
        seen = accesses.setdefault(access.name, access)
        if seen is not access and (seen.projection != access.projection or access.output):
            raise UnsupportedConversion(f"{node.name}: tensor {tensor.name} is accessed twice with different maps")
    names = [tensor_name(tensor) for tensor in priced[:ZIGZAG_INPUTS]] + [None, None]
    output_name = tensor_name(node.outputs[0])
    renames = {
        "input": names[0] or NOTHING,
        "weight": names[1] or NOTHING,
        "output": output_name,
        PARTIAL_SUMS: output_name if is_mac_operator_type(node.type) else NOTHING,
    }
    return EinsumDescription(
        name=name,
        accesses=tuple(accesses.values()),
        sizes=tuple(dim_sizes),
        renames=tuple(renames.items()),
    )


def rank_variables(einsum: str, sizes: Sequence[int]) -> tuple[str, ...]:
    """The Einsum's rank variables, one per dim and lowercase as AccelForge requires."""
    return tuple(f"{einsum.lower()}_d{i}" for i in range(len(sizes)))


def tensor_name(tensor: Tensor) -> str:
    """The AccelForge name of ``tensor``: an identifier, and no word AccelForge or the renames reserve."""
    name = _identifier(tensor.name)
    return f"t_{name}" if name in RESERVED_NAMES else name


def _access(
    node: str,
    tensor: Tensor,
    expressions: Sequence[AffineExpr],
    variables: Sequence[str],
    sizes: Sequence[int],
    output: bool,
) -> Access:
    """An index's constant, a padding or an offset, only shifts the indices reached, so it is dropped as ZigZag drops
    it; the indices start at 0, which a negative coefficient would otherwise pass below."""
    name = tensor_name(tensor)
    projection = []
    for axis, expr in enumerate(expressions):
        coefficients = _box_projection(node, expr, sizes)
        shift = -sum(min(0, c * (sizes[d] - 1)) for d, c in coefficients.items())
        projection.append((f"R{axis}_{name}", _render(shift, coefficients, variables)))
    return Access(name, tuple(projection), output, tensor.operand_type.bitwidth)


def _box_projection(node: str, expr: AffineExpr, sizes: Sequence[int]) -> dict[int, int]:
    """The coefficients of ``expr`` per dim, made to index a gap-free range, as AccelForge without Barvinok only
    handles box-shaped data spaces: one strided dim (``2*oy``, a 1x1 stride-2 convolution) becomes that dim."""
    coefficients = _linear(node, expr)[1]
    live = {d: c for d, c in coefficients.items() if c and sizes[d] > 1}
    if len(live) == 1 and abs(next(iter(live.values()))) > 1:
        return {next(iter(live)): 1}
    reach = 0
    for d, coefficient in sorted(live.items(), key=lambda item: abs(item[1])):
        if abs(coefficient) > reach + 1:
            raise UnsupportedConversion(f"{node}: {expr} leaves gaps between the indices it reaches")
        reach += abs(coefficient) * (sizes[d] - 1)
    return coefficients


def _render(constant: int, coefficients: dict[int, int], variables: Sequence[str]) -> str:
    """An affine expression as ISL text over ``variables``."""
    terms = [
        (c, variables[d] if abs(c) == 1 else f"{abs(c)}*{variables[d]}") for d, c in sorted(coefficients.items()) if c
    ]
    if constant or not terms:
        terms.append((constant, str(abs(constant))))
    text = ("-" if terms[0][0] < 0 else "") + terms[0][1]
    return text + "".join(f" {'-' if c < 0 else '+'} {term}" for c, term in terms[1:])


def _linear(node: str, expr: AffineExpr) -> tuple[int, dict[int, int]]:
    """The constant and the coefficient per dim position of an affine-linear ``expr``."""
    if isinstance(expr, AffineConstantExpr):
        return expr.value, {}
    if isinstance(expr, AffineDimExpr):
        return 0, {expr.position: 1}
    if isinstance(expr, AffineBinaryOpExpr) and expr.kind is AffineBinaryOpKind.Add:
        (lhs_constant, lhs), (rhs_constant, rhs) = _linear(node, expr.lhs), _linear(node, expr.rhs)
        return lhs_constant + rhs_constant, {d: lhs.get(d, 0) + rhs.get(d, 0) for d in lhs.keys() | rhs.keys()}
    if isinstance(expr, AffineBinaryOpExpr) and expr.kind is AffineBinaryOpKind.Mul:
        for factor, term in ((expr.lhs, expr.rhs), (expr.rhs, expr.lhs)):
            if isinstance(factor, AffineConstantExpr):
                constant, coefficients = _linear(node, term)
                return factor.value * constant, {d: factor.value * c for d, c in coefficients.items()}
        raise UnsupportedConversion(f"{node}: {expr} multiplies two dims, which is not affine")
    raise UnsupportedConversion(f"{node}: {expr} is not affine-linear (mod, floordiv, ceildiv and symbols are not)")


def _identifier(name: str) -> str:
    identifier = re.sub(r"\W", "_", name)
    return identifier if re.match(r"[A-Za-z]", identifier) else f"n{identifier}"
