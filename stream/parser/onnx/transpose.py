from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.transpose_node import TransposeNode


class TransposeParser(OnnxOperatorParser):
    """Parses an onnx reshape operator into a TransposeNode."""

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        assert len(predecessors) == 1, "An ONNX transpose node with multiple input nodes is not supported"
        predecessor = predecessors.pop()

        permute_axes = self.get_permute_indices()
        input_names = list(self.node.input)

        return TransposeNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            permute_axes=permute_axes,
            input_names=input_names,
        )

    def get_permute_indices(self):
        """`perm` can be attached as an attribute of a transpose node"""
        try:
            perm_attr = next(filter(lambda x: x.name == "perm", self.node.attribute))
            return list(perm_attr.ints)
        except StopIteration:
            return None
