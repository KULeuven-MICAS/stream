"""Attention -> its split-sequence (flash-decoding) form: each part of the sequence attended to alone, then combined."""

from __future__ import annotations

from dataclasses import dataclass, replace

from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr, AffineMap

from stream.workload.affine_access import map_dim_positions
from stream.workload.node import ComputationNode, HasInputs, HasOutputs, InEdge, Node, NormalizationNode, OutEdge
from stream.workload.normalization import expand_normalizations
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

__all__ = ["Attention", "FlashDecodingRewrite"]

PAIR = 2


@dataclass(frozen=True)
class Attention:
    """Attention found by its structure: scores, element-wise ops, a softmax along ``axis`` and the context."""

    scores: ComputationNode
    elementwise: tuple[ComputationNode, ...]
    softmax: tuple[ComputationNode, ...]
    context: ComputationNode
    probs: Tensor
    replaced: tuple[Node, ...]
    axis: int
    length: int
    output_axes: tuple[int | None, ...]

    @property
    def nodes(self) -> tuple[ComputationNode, ...]:
        return (self.scores, *self.elementwise, *self.softmax, self.context)

    @property
    def statistics(self) -> tuple[Tensor, Tensor]:
        """The softmax's maximum and sum."""
        return self.softmax[0].outputs[0], self.softmax[-1].outputs[0]

    @property
    def along_sequence(self) -> set[Tensor]:
        """The tensors the attention makes along the sequence."""
        return {t for n in (self.scores, *self.elementwise, *self.softmax[1:-1]) for t in n.outputs} | {self.probs}

    def position(self, node: ComputationNode) -> int:
        """The loop of ``node`` walking the sequence."""
        return next(
            position
            for t in node.tensors
            if t in self.along_sequence and (position := _dim(node.get_mapping(t).results[self.axis])) is not None
        )

    def sequence_reads(self) -> list[tuple[Tensor, int]]:
        """The tensors from outside the attention read along the sequence, each with the axis the sequence indexes."""
        made = {t for node in self.nodes for t in node.outputs} | {self.probs}
        return [
            (tensor, axis)
            for node in self.nodes
            for tensor in node.inputs
            if tensor not in made
            for axis, result in enumerate(node.get_mapping(tensor).results)
            if self.position(node) in _positions(result, node.num_dims)
        ]

    def views(self, workload: Workload) -> dict[InEdge, int]:
        """The inputs only the attention reads, along the sequence at one axis alone, with that axis."""
        edges = {edge.outputs[0]: edge for edge in workload.get_in_edges()}
        return {
            edges[tensor]: axis
            for tensor, axis in self.sequence_reads()
            if tensor in edges
            and (readers := _readers(workload, tensor)) <= set(self.nodes)
            and all(_dim(node.get_mapping(tensor).results[axis]) == self.position(node) for node in readers)
        }


class FlashDecodingRewrite:
    """Split attention's sequence into a part loop and a loop within it, the parts combined by their maxima."""

    name = "flash_decoding"

    def find(self, workload: Workload) -> list[Attention]:
        return [found for node in workload.get_computation_nodes() if (found := _attention(workload, node)) is not None]

    def apply(self, workload: Workload, attention: Attention, parts: int) -> Workload:
        if parts < 1 or attention.length % parts:
            raise ValueError(
                f"{attention.context.name} attends over a sequence of {attention.length}, which {parts} parts do not "
                "split evenly"
            )
        views = attention.views(workload)
        viewed = {e.outputs[0]: _split_axis(e.outputs[0], axis, parts, e.outputs[0].name) for e, axis in views.items()}
        made = attention.along_sequence - {attention.probs}
        part_tensors = {t: _split_axis(t, attention.axis, parts, f"{t.name}_part") for t in made} | viewed
        part_tensors[attention.probs] = part_tensors[attention.softmax[-1].inputs[0]]
        statistics = (*attention.statistics, attention.context.outputs[0])
        part_tensors |= {t: Tensor.create(f"{t.name}_part", t.operand_type, (*t.shape, parts)) for t in statistics}
        split = attention.along_sequence | set(viewed)
        within = attention.length // parts
        per_part = [_per_part(node, attention.position(node), within, part_tensors, split) for node in attention.nodes]
        combine = _combine(attention, *(part_tensors[t] for t in statistics))
        first = next(i for i, node in enumerate(workload.nodes) if node in attention.replaced)
        edges = {edge: replace(edge, outputs=(viewed[edge.outputs[0]],)) for edge in views}
        kept = [edges.get(node, node) for node in workload.nodes if node not in attention.replaced]
        return Workload([*kept[:first], *per_part, *combine, *kept[first:]])


def _positions(expr: AffineExpr, num_dims: int) -> frozenset[int]:
    return map_dim_positions(AffineMap(num_dims, 0, (expr,)))


def _dim(expr: AffineExpr) -> int | None:
    return expr.position if isinstance(expr, AffineDimExpr) else None


def _producer(workload: Workload, tensor: Tensor) -> Node | None:
    return next((n for n in workload.nodes if isinstance(n, HasOutputs) and tensor in n.outputs), None)


def _readers(workload: Workload, tensor: Tensor) -> set[Node]:
    return {n for n in workload.nodes if isinstance(n, HasInputs) and tensor in n.inputs}


def _split_axis(tensor: Tensor, axis: int, parts: int, name: str) -> Tensor:
    """``tensor`` as ``name``, its ``axis`` cut into a part axis of ``parts`` and the axis within a part."""
    shape = tensor.shape
    return Tensor.create(name, tensor.operand_type, (*shape[:axis], parts, shape[axis] // parts, *shape[axis + 1 :]))


def _drop(rank: int, axis: int) -> AffineMap:
    return AffineMap(rank, 0, tuple(AffineExpr.dimension(i) for i in range(rank) if i != axis))


def _reduced_axis(statistic: AffineMap) -> int | None:
    """The axis a reduction's output map drops or keeps at 0; None for any other map."""
    rank = statistic.num_dims
    dims = [AffineExpr.dimension(i) for i in range(rank)]
    kept = [(*dims[:axis], AffineConstantExpr(0), *dims[axis + 1 :]) for axis in range(rank)]
    return next((a for a in range(rank) if statistic == _drop(rank, a) or statistic.results == kept[a]), None)


def _kind(node: Node | None, kind: str, inputs: int) -> ComputationNode | None:
    """``node`` if it is a ``kind`` of ``inputs`` operands."""
    ok = isinstance(node, ComputationNode) and node.type == kind and len(node.inputs) == inputs
    return node if ok else None


def _expanded(node: NormalizationNode) -> tuple[ComputationNode, ...]:
    """The maximum, exponent and sum of a Softmax's expansion."""
    edges = [InEdge(name=t.name, outputs=(t,)) for t in node.inputs]
    whole = Workload([*edges, node, OutEdge(name=node.name, inputs=node.outputs)])
    return expand_normalizations(whole).get_computation_nodes()[:-1]


def _softmax(workload: Workload, node: Node | None) -> tuple[tuple[ComputationNode, ...], tuple[Node, ...]] | None:
    """The softmax ``node`` ends: its maximum, its shift and exponent, its sum, and the workload nodes they are."""
    if isinstance(node, NormalizationNode) and node.type == "Softmax" and len(node.reduction_axes) == 1:
        return _expanded(node), (node,)
    if (div := _kind(node, "Div", PAIR)) is None:
        return None
    total = _kind(_producer(workload, div.inputs[1]), "ReduceSum", 1)
    exp = _kind(made := _producer(workload, div.inputs[0]), "Exp", PAIR) or _kind(made, "Exp", 1)
    if total is None:
        return None
    shift = exp if exp is None or len(exp.inputs) == PAIR else _kind(_producer(workload, exp.inputs[0]), "Sub", PAIR)
    maximum = _kind(_producer(workload, shift.inputs[1]), "ReduceMax", 1) if shift is not None else None
    if exp is None or shift is None or maximum is None or total.inputs != exp.outputs:
        return None
    middle = (shift, exp) if shift is not exp else (exp,)
    if not _maps_agree(maximum, middle, total, div):
        return None
    return (maximum, *middle, total), (maximum, *middle, total, div)


def _maps_agree(
    maximum: ComputationNode, middle: tuple[ComputationNode, ...], total: ComputationNode, div: ComputationNode
) -> bool:
    """Whether the softmax's nodes read the scores whole and its statistics along one reduced axis."""
    scores = maximum.inputs[0]
    identity = AffineMap.identity(len(scores.shape))
    statistic = maximum.get_mapping(maximum.outputs[0])
    expected = [
        (maximum, (identity, statistic)),
        (total, (identity, statistic)),
        (div, (identity, statistic, identity)),
    ]
    expected += [
        (n, (identity, identity)) if len(n.inputs) == 1 else (n, (identity, statistic, identity)) for n in middle
    ]
    return (
        middle[0].inputs[0] == scores
        and _reduced_axis(statistic) is not None
        and all(n.operand_mapping == maps for n, maps in expected)
    )


def _is_contraction(node: ComputationNode, axis: int) -> bool:
    """Whether ``node`` contracts two operands into scores whose ``axis`` only the second indexes."""
    if len(node.inputs) != PAIR or len(node.outputs) != 1:
        return False
    output = node.get_mapping(node.outputs[0])
    if (position := _dim(output.results[axis])) is None or len(map_dim_positions(output)) == node.num_dims:
        return False
    first, second = (map_dim_positions(node.get_mapping(t)) for t in node.inputs)
    return position in second and position not in first


def _scores(workload: Workload, tensor: Tensor, axis: int) -> list[ComputationNode] | None:
    """The contraction making ``tensor`` then the element-wise ops after it, in order; None unless made so."""
    node = _producer(workload, tensor)
    if not isinstance(node, ComputationNode) or len(node.outputs) != 1:
        return None
    if _is_contraction(node, axis):
        return [node]
    identity = AffineMap.identity(len(tensor.shape))
    if node.get_mapping(tensor) != identity:
        return None
    chains = (
        _scores(workload, read, axis)
        for read in node.inputs
        if read.shape == tensor.shape and node.get_mapping(read) == identity
    )
    return next(([*chain, node] for chain in chains if chain is not None), None)


def _attention(workload: Workload, context: ComputationNode) -> Attention | None:
    """The attention whose context is ``context``, its tensors read only within it; None if there is none."""
    if len(context.inputs) != PAIR or len(context.outputs) != 1:
        return None
    for probs, values in (context.inputs, context.inputs[::-1]):
        if (found := _softmax(workload, _producer(workload, probs))) is not None:
            if (attention := _around(workload, context, probs, values, *found)) is not None:
                return attention
    return None


def _around(
    workload: Workload,
    context: ComputationNode,
    probs: Tensor,
    values: Tensor,
    softmax: tuple[ComputationNode, ...],
    replaced: tuple[Node, ...],
) -> Attention | None:
    """The attention around ``softmax`` whose ``context`` reduces its axis of ``probs`` against ``values``."""
    statistic = softmax[0].get_mapping(softmax[0].outputs[0])
    axis = _reduced_axis(statistic)
    output = context.get_mapping(context.outputs[0])
    reads = [_dim(r) for r in context.get_mapping(probs).results]
    out_dims = [_dim(r) for r in output.results]
    if axis is None or None in reads or reads[axis] in out_dims:
        return None
    if reads[axis] not in map_dim_positions(context.get_mapping(values)):
        return None
    if any(reads[a] not in out_dims for a in range(len(reads)) if a != axis):
        return None
    if (chain := _scores(workload, softmax[0].inputs[0], axis)) is None:
        return None
    nodes = (*chain, *replaced, context)
    made = {t for n in (*chain, *softmax) for t in n.outputs} | {probs}
    if any(not _readers(workload, t) <= set(nodes) | set(softmax) for t in made):
        return None
    kept = len(statistic.results) == len(reads)
    return Attention(
        scores=chain[0],
        elementwise=tuple(chain[1:]),
        softmax=softmax,
        context=context,
        probs=probs,
        replaced=nodes,
        axis=axis,
        length=softmax[0].inputs[0].shape[axis],
        output_axes=tuple(
            None if a == axis else out_dims.index(reads[a]) for a in range(len(reads)) if kept or a != axis
        ),
    )


def _per_part(
    node: ComputationNode, position: int, within: int, part_tensors: dict[Tensor, Tensor], split: set[Tensor]
) -> ComputationNode:
    """``node`` with its sequence loop at ``position`` cut into a part loop and a loop ``within`` the part."""
    dims = [AffineExpr.dimension(k) for k in range(node.num_dims + 1)]
    part, inside = dims[position], dims[position + 1]
    substitute = [*dims[:position], part * within + inside, *dims[position + 2 :]]

    def part_map(tensor: Tensor) -> AffineMap:
        results: list[AffineExpr] = []
        for result in node.get_mapping(tensor).results:
            whole = tensor in split and _dim(result) == position
            results += [part, inside] if whole else [result.replace_dims_and_symbols(substitute, [])]
        if tensor in part_tensors and tensor not in split:
            results.append(part)
        return AffineMap(node.num_dims + 1, 0, tuple(results))

    return replace(
        node,
        name=f"{node.name}_part",
        inputs=tuple(part_tensors.get(t, t) for t in node.inputs),
        outputs=tuple(part_tensors.get(t, t) for t in node.outputs),
        operand_mapping=tuple(part_map(t) for t in node.tensors),
    )


def _combine(attention: Attention, maxima: Tensor, sums: Tensor, contexts: Tensor) -> list[ComputationNode]:
    """The parts' maximum, their weights ``exp(max - maximum)``, and the weighted contexts over the weighted sums."""
    output = attention.context.outputs[0]
    dt, base = output.operand_type, attention.context.name
    rank, out_rank = len(maxima.shape) - 1, len(output.shape)
    stats, outs = AffineMap.identity(rank + 1), AffineMap.identity(out_rank + 1)
    stat, whole = _drop(rank + 1, rank), _drop(out_rank + 1, out_rank)
    in_output = tuple(AffineConstantExpr(0) if a is None else AffineExpr.dimension(a) for a in attention.output_axes)
    weight_in_output = AffineMap(out_rank + 1, 0, (*in_output, AffineExpr.dimension(out_rank)))

    def made(name: str, shape: tuple[int, ...]) -> Tensor:
        return Tensor.create(f"{base}_{name}", dt, shape)

    maximum, weight = made("combine_max", maxima.shape[:-1]), made("combine_weight", maxima.shape)
    weighted_sums, total = made("combine_weighted_sum", maxima.shape), made("combine_sum", maxima.shape[:-1])
    weighted, combined = made("combine_weighted", contexts.shape), made("combine_total", output.shape)

    def node(kind: str, name: str, inputs: tuple[Tensor, ...], out: Tensor, maps: tuple[AffineMap, ...]):
        return ComputationNode(type=kind, name=f"{base}_{name}", inputs=inputs, outputs=(out,), operand_mapping=maps)

    normalised = (AffineMap.identity(out_rank), AffineMap(out_rank, 0, in_output), AffineMap.identity(out_rank))
    return [
        node("ReduceMax", "combine_max", (maxima,), maximum, (stats, stat)),
        node("Exp", "combine_weight", (maxima, maximum), weight, (stats, stat, stats)),
        node("Mul", "combine_weighted_sum", (weight, sums), weighted_sums, (stats, stats, stats)),
        node("ReduceSum", "combine_sum", (weighted_sums,), total, (stats, stat)),
        node("Mul", "combine_weighted", (weight, contexts), weighted, (weight_in_output, outs, outs)),
        node("ReduceSum", "combine_total", (weighted,), combined, (outs, whole)),
        node("Div", "combine", (combined, total), output, normalised),
    ]
