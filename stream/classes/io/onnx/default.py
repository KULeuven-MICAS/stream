from stream.classes.io.onnx.operator_parser import OnnxOperatorParser
from stream.classes.workload.dummy_node import DummyNode


class DefaultNodeParser(OnnxOperatorParser):
    """Parse an ONNX node into a DummyNode."""

    def run(self):
        """Run the parser"""
        return self.generate_dummy_node()

    def generate_dummy_node(self):
        predecessors = self.get_node_predecessors()
        input_names = list(self.node.input)
        output_names = list(self.node.output)

        return DummyNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            input_names=input_names,
            output_names=output_names,
            op_type=self.node.op_type.lower(),
        )
