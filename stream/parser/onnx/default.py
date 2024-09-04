from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dummy_node import DummyNode

from stream.workload.dummy_node import DummyNode


class DefaultNodeParser(OnnxOperatorParser):
    """Parse an ONNX node into a DummyNode."""

    def generate_node(self):
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
