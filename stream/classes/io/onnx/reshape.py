from zigzag.parser.onnx.utils import get_node_input_output_dimension_shapes

from stream.classes.io.onnx.operator_parser import OnnxOperatorParser
from stream.classes.workload.reshape_node import ReshapeNode


class ReshapeParser(OnnxOperatorParser):
    """Parses an onnx reshape operator into a ReshapeNode."""

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        assert len(predecessors) == 1, "An ONNX reshape node with multiple input nodes is not supported"
        predecessor = predecessors.pop()

        # The operator shape is saved as the second input, so we need to get the input's dimension shape
        shape = tuple(get_node_input_output_dimension_shapes(self.node, self.onnx_model)[1])
        input_names = [self.node.input[0]]
        output_names = [self.node.output[0]]

        return ReshapeNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            shape=shape,
            input_names=input_names,
            output_names=output_names,
        )
