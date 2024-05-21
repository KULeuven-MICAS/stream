from stream.classes.workload.transpose_node import TransposeNode
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser


class TransposeParser(ONNXOperatorParser):
    """Parses an onnx reshape operator into a ReshapeNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        raise NotImplementedError

        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        return self.generate_layer_node_for_transpose()

    def generate_layer_node_for_transpose(self):
        # Get the predecessors of this node
        predecessors: list[int] = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)
        # print(predecessors)

        # Get the input names of the operator
        input_names = [self.node.input[0]]
        # Get the output names of the operator
        output_names = [self.node.output[0]]
        node_obj = TransposeNode(predecessors, input_names, output_names)
        return node_obj
