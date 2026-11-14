"""Elementwise layers run as post-ops of a multiply-accumulate node: on its cores, at its output tile, once its sum is
done, as oneDNN's post-ops and IRON's ProjectionGEMM activation do."""

import dataclasses
from collections.abc import Collection

from xdsl.ir.affine import AffineDimExpr, AffineMap

from stream.workload.node import ComputationNode, HasInputs, Node
from stream.workload.utils import is_mac_operator_type
from stream.workload.workload import Workload


def fold_post_ops(workload: Workload, layers: Collection[str]) -> tuple[Workload, dict[str, str]]:
    """``workload`` with each of ``layers`` folded into the multiply-accumulate node that produces one of its inputs,
    and per folded layer the node it runs on.

    A layer folds where it reads that input at the index it writes its output, and nothing else reads that input:
    the producer then writes the layer's output, and reads the layer's other inputs at its own output index. Of two
    such producers, the later one in the workload takes it, so the other input is ready before its sum is done.

    Raises:
        ValueError: A layer is not elementwise over a multiply-accumulate node's output.
    """
    nodes: list[Node] = list(workload.dataflow_sort())
    into: dict[str, str] = {}
    for layer in [n.name for n in nodes if n.name in layers]:
        node = next(n for n in nodes if n.name == layer)
        producer = _producer(nodes, node)
        if producer is None:
            raise ValueError(f"{layer} is not elementwise over the output of a multiply-accumulate node it alone reads")
        source = node.inputs.index(producer.output)
        written = producer.operand_mapping[-1].results
        at = {r.position: written[i] for i, r in enumerate(node.operand_mapping[-1].results)}
        others = [i for i in range(len(node.inputs)) if i != source]
        maps = [
            AffineMap(producer.num_dims, 0, tuple(at[r.position] for r in node.operand_mapping[i].results))
            for i in others
        ]
        folded = dataclasses.replace(
            producer,
            inputs=(*producer.inputs, *(node.inputs[i] for i in others)),
            outputs=node.outputs,
            operand_mapping=(*producer.operand_mapping[:-1], *maps, producer.operand_mapping[-1]),
        )
        nodes = [folded if n is producer else n for n in nodes if n is not node]
        into[layer] = producer.name
    return Workload(nodes), into


def _producer(nodes: list[Node], node: Node) -> ComputationNode | None:
    """The latest multiply-accumulate node whose output ``node`` alone reads, at the index ``node`` writes."""
    if not isinstance(node, ComputationNode) or not _over_dims(node.operand_mapping[-1]):
        return None
    candidates = []
    for tensor, access in zip(node.inputs, node.operand_mapping, strict=False):
        producer = next((n for n in nodes if isinstance(n, ComputationNode) and tensor in n.outputs), None)
        readers = [n for n in nodes if isinstance(n, HasInputs) and tensor in n.inputs]
        if (
            producer is not None
            and is_mac_operator_type(producer.type)
            and readers == [node]
            and access == node.operand_mapping[-1]
            and _over_dims(producer.operand_mapping[-1])
            and all(_over_dims(m) for m in node.operand_mapping)
        ):
            candidates.append(producer)
    return max(candidates, key=nodes.index, default=None)


def _over_dims(access: AffineMap) -> bool:
    return all(isinstance(r, AffineDimExpr) for r in access.results)
