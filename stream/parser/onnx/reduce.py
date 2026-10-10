import dataclasses

from onnx import numpy_helper
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr, AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.node import Node
from stream.workload.utils import is_mac_operator_type
from stream.workload.workload import ComputationNode, Tensor


class ReduceParser(OnnxOperatorParser):
    """Parses an ONNX ReduceSum, ReduceMean or ReduceMax into a ``ComputationNode`` over its input's axes: the
    output indexes the axes it keeps (a kept reduced axis at 0), so the reduced ones are the node's reduction loops,
    which a vector unit walks accumulating into the output, as the reductions of a decomposed softmax do."""

    def _axes(self, rank: int) -> tuple[int, ...]:
        axes = self.get_node_attribute_ints("axes")
        if len(self.node.input) > 1 and self.node.input[1]:
            initializer = next(t for t in self.onnx_model.graph.initializer if t.name == self.node.input[1])
            axes = [int(a) for a in numpy_helper.to_array(initializer).ravel()]
        if not axes:
            if self.get_node_attribute_int("noop_with_empty_axes"):
                return ()
            axes = list(range(rank))
        return tuple(sorted(a % rank for a in axes))

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        data = name_to_tensor_dict[self.node.input[0]]
        rank = len(data.shape)
        axes = self._axes(rank)
        keepdims = self.get_node_attribute_int("keepdims")
        keep = keepdims is None or keepdims == 1
        outputs = self.get_output_tensors()
        assert len(outputs) == 1, f"{self.node.op_type} must have exactly 1 output."

        result = tuple(
            AffineExpr.constant(0) if axis in axes else AffineExpr.dimension(axis)
            for axis in range(rank)
            if keep or axis not in axes
        )
        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=(data,),
            outputs=outputs,
            operand_mapping=(AffineMap.identity(rank), AffineMap(rank, 0, result)),
        )


def fold_sums_into_contractions(nodes: list[Node]) -> tuple[list[Node], set[str]]:
    """Fold each ReduceSum whose input only it reads into the multiply-accumulate node producing that input: a sum
    over an axis of a sum of products is a sum of products over that axis too, so the producer contracts the loops
    that indexed the summed axes and writes the reduced output, as an accelerator accumulates them in place rather
    than writing every partial result out to add them up afterwards. Also returns the names of the nodes folded
    into."""
    readers: dict[str, int] = {}
    for node in nodes:
        for tensor in getattr(node, "inputs", ()):
            readers[tensor.name] = readers.get(tensor.name, 0) + 1
    producers = {t.name: node for node in nodes if isinstance(node, ComputationNode) for t in node.outputs}
    folded: dict[int, ComputationNode] = {}
    removed: set[int] = set()
    for node in nodes:
        if not isinstance(node, ComputationNode) or node.type != "ReduceSum":
            continue
        source = node.inputs[0]
        producer = producers.get(source.name)
        if producer is None or readers[source.name] != 1 or not is_mac_operator_type(producer.type):
            continue
        producer = folded.get(id(producer), producer)
        written = producer.operand_mapping[-1].results
        if not all(isinstance(r, AffineDimExpr) for r in written):
            continue
        kept = node.operand_mapping[-1].results
        result = tuple(r if isinstance(r, AffineConstantExpr) else written[r.position] for r in kept)
        maps = (*producer.operand_mapping[:-1], AffineMap(producer.num_dims, 0, result))
        folded[id(producers[source.name])] = dataclasses.replace(producer, outputs=node.outputs, operand_mapping=maps)
        removed.add(id(node))
    return [folded.get(id(n), n) for n in nodes if id(n) not in removed], {n.name for n in folded.values()}
