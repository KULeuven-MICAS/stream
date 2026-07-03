"""Data-movement access nodes (Slice, Gather) for KV-cache and indexing dataflows.

These operators move/select tensor slices rather than computing; what matters for the hardware is
*which* data they touch, so they are represented by their affine (or conservatively-bounded) access
relation and the derived-dependency machinery (:mod:`stream.workload.affine_access`) reads the moved
region straight off the maps.

- **Slice** is affine: ``y[..,p,..] = x[..,start+step*p,..]`` -- an offset access along one axis, so
  ``compose_dependency`` recovers the exact source region (e.g. a KV cache read ``cache[0:t]``).
- **Gather** is data-dependent (``y[..,m,..] = x[..,idx[m],..]``), hence *not* affine. It is modelled
  conservatively: the gathered output index ``m`` and the source position are separate iteration
  dimensions, and the source axis is read in *full* (a REDUCTION-shaped read). That over-approximates
  the moved data to the whole axis -- the safe assumption for paged/sparse KV-cache gathers, where the
  indices are only known at runtime.
"""

from __future__ import annotations

from xdsl.ir.affine import AffineExpr, AffineMap

from stream.workload.node import ComputationNode
from stream.workload.tensor import Tensor

__all__ = ["slice_access_maps", "gather_access_maps", "slice_node", "gather_node"]


def _axis_result(pos: int, start: int, step: int) -> AffineExpr:
    """Affine index into a sliced axis: ``dim(pos)*step + start`` (kept minimal when start/step trivial)."""
    expr: AffineExpr = AffineExpr.dimension(pos)
    if step != 1:
        expr = expr * step
    if start != 0:
        expr = expr + start
    return expr


def slice_access_maps(rank: int, offsets: dict[int, int], steps: dict[int, int]) -> tuple[AffineMap, AffineMap]:
    """(input, output) affine maps for a slice: the input indexes each axis with ``dim*step + start``
    (offset from ``offsets``/``steps``, default 0/1), the output is identity over the sliced shape."""
    in_map = AffineMap(
        rank,
        0,
        tuple(_axis_result(i, offsets.get(i, 0), steps.get(i, 1)) for i in range(rank)),
    )
    return in_map, AffineMap.identity(rank)


def gather_access_maps(rank: int, axis: int) -> tuple[AffineMap, AffineMap]:
    """(input, output) affine maps for a conservative gather: the source axis is read in full via an
    extra iteration dim (position ``rank``), the output gather index ``m`` sits on ``axis``."""
    src = rank
    in_map = AffineMap(rank + 1, 0, tuple(AffineExpr.dimension(src if i == axis else i) for i in range(rank)))
    out_map = AffineMap(rank + 1, 0, tuple(AffineExpr.dimension(i) for i in range(rank)))
    return in_map, out_map


def slice_node(
    x: Tensor, axis: int, start: int, length: int, step: int = 1, name: str = "slice"
) -> tuple[ComputationNode, Tensor]:
    """A ``Slice`` access node: ``y[..,p,..] = x[..,start+step*p,..]`` along ``axis``.

    The access is affine (an offset), so the derived footprint/dependency give the exact source region
    the slice reads -- the core KV-cache mechanism (read the valid cache prefix ``cache[0:t]``)."""
    rank = len(x.shape)
    out_shape = tuple(length if i == axis else s for i, s in enumerate(x.shape))
    in_map, out_map = slice_access_maps(rank, {axis: start}, {axis: step})
    y = Tensor.create(f"{name}_out", x.operand_type, out_shape)
    node = ComputationNode(type="Slice", name=name, inputs=(x,), outputs=(y,), operand_mapping=(in_map, out_map))
    return node, y


def gather_node(x: Tensor, axis: int, num_indices: int, name: str = "gather") -> tuple[ComputationNode, Tensor]:
    """A ``Gather`` access node: ``y[..,m,..] = x[..,idx[m],..]`` along ``axis`` (data-dependent).

    The index is a runtime value, so the source access is not affine. It is bounded conservatively:
    the source position is its own iteration dimension read in full (so the moved region is the whole
    ``axis``), while the output gather index ``m`` is a separate parallel dimension. That is the safe
    data-movement model for paged/sparse KV-cache gathers."""
    rank = len(x.shape)
    out_shape = tuple(num_indices if i == axis else s for i, s in enumerate(x.shape))
    in_map, out_map = gather_access_maps(rank, axis)
    y = Tensor.create(f"{name}_out", x.operand_type, out_shape)
    node = ComputationNode(type="Gather", name=name, inputs=(x,), outputs=(y,), operand_mapping=(in_map, out_map))
    return node, y
