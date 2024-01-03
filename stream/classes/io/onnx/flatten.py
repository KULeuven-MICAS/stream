from stream.classes.workload.flatten_node import FlattenNode
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_attribute_ints_with_name


class FlattenParser(Parser):
    """Parses an onnx flatten operator into a FlattenNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        return self.generate_flatten_node()

    def generate_flatten_node(self):
        # Get the predecessors of this node
        predecessors = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)
        attrs = self.node.attribute
        # Get the axis which indicates how to flatten the input tensor
        axis = get_attribute_ints_with_name("axis", attrs, default=None)
        # Get the input names of the operator
        input_names = [self.node.input[0]]
        # Get the output names of the operator
        output_names = [self.node.output[0]]
        node_obj = FlattenNode(self.node_id, predecessors, axis, input_names, output_names)
        return node_obj
