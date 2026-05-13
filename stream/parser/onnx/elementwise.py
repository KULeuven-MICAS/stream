from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.elementwise_node import ElementwiseNode


class ElementwiseParser(OnnxOperatorParser):
    """Parser for onnx operators that perform an elementwise operation on two input tensors into a single output tensor.
    For example, an Add operator adds two tensors together in every position into one output tensor.
    """

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        raise NotImplementedError
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)
        self.type = node.op_type.lower()
        self.name = node.name

    def generate_node(self):
        input_names = list(self.node.input)

        # Get the predecessors of this node
        predecessors = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)

        EXPECTED_PREDECESSORS = 2
        # Get the names of the two inputs
        assert len(self.node.input) == EXPECTED_PREDECESSORS, (
            f"Elementwise node has more than two inputs: {self.node.input}"
        )
        # Get the output name
        node_obj = ElementwiseNode(
            node_id=self.node_id,
            node_name=self.name,  # type: ignore
            predecessor=predecessors,  # type: ignore
            input_names=input_names,
        )
        return node_obj
