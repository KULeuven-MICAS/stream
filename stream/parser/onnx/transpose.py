from stream.classes.io.onnx.operator_parser import OnnxOperatorParser
from stream.classes.workload.transpose_node import TransposeNode


class TransposeParser(OnnxOperatorParser):
    """Parses an onnx reshape operator into a TransposeNode."""

    def generate_layer_node_for_transpose(self):
        predecessors = self.get_node_predecessors()
        assert len(predecessors) == 1, "An ONNX transpose node with multiple input nodes is not supported"
        predecessor = predecessors.pop()

        permute_axes = self.get_permute_indices()
        input_names = [self.node.input[0]]
        output_names = [self.node.output[0]]

        return TransposeNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            input_names=input_names,
            output_names=output_names,
            permute_axes=permute_axes,
        )

    def get_permute_indices(self):
        """`perm` can be attached as an attribute of a transpose node"""
        try:
            perm_attr = next(filter(lambda x: x.name == "perm", self.node.attribute))
            return list(perm_attr.ints)
        except StopIteration:
            return None
