from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from stream.classes.workload.dummy_node import DummyNode


class DefaultNodeParser(ONNXOperatorParser):
    """Parse an ONNX node into a DummyNode."""

    def __init__(self, node_id, node, nodes_outputs) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping=None, onnx_model=None)

    def run(self):
        """Run the parser"""
        dummy_node = self.generate_dummy_node()
        return dummy_node

    def generate_dummy_node(self):
        preds = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    preds.append(n)

        # Get the input names of this operator
        input_names = list(self.node.input)
        output_names = list(self.node.output)
        op_type = self.node.op_type.lower()
        node_obj = DummyNode(
            self.node_id, preds, self.node.name, input_names, output_names, op_type
        )

        return node_obj
