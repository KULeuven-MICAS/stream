from stream.classes.io.onnx.operator_parser import OnnxOperatorParser
from stream.classes.workload.lpnormalization_node import LpNormalizationNode


class LpNormalizationParser(OnnxOperatorParser):
    """Parses an onnx reshape operator into a LpNormalizationNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        raise NotImplementedError

        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def generate_node(self):
        # Get the predecessors of this node
        predecessors = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)

        # Get the input names of the operator
        input_names = [self.node.input[0]]
        # Get the output names of the operator
        output_names = [self.node.output[0]]
        node_obj = LpNormalizationNode(predecessors, input_names, output_names)
        return node_obj
