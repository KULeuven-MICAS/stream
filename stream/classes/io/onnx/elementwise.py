from stream.classes.workload.elementwise_node import ElementwiseNode
from zigzag.classes.io.onnx.parser import Parser


class ElementwiseParser(Parser):
    """Parser for onnx operators that perform an elementwise operation on two input tensors into a single output tensor.
    For example, an Add operator adds two tensors together in every position into one output tensor.
    """
    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)
        self.type = node.op_type.lower()
        self.name = node.name

    def run(self):
        return self.generate_elementwise_node()

    def generate_elementwise_node(self):
        # Get the predecessors of this node
        predecessors = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)

        # Get the names of the two inputs
        assert len(self.node.input) == 2, f"Elementwise node has more than two inputs: {self.node.input}"
        input_names = [self.node.input[0], self.node.input[1]]
        # Get the output name 
        output_names = [self.node.output[0]]
        node_obj = ElementwiseNode(self.type, self.name, predecessors, input_names, output_names)
        return node_obj