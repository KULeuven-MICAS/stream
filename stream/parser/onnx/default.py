from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dummy_node import DummyNode

from stream.workload.dummy_node import DummyNode


class DefaultNodeParser(OnnxOperatorParser):
    """Parse an ONNX node into a DummyNode."""

    def generate_node(self):
        predecessors = self.get_node_predecessors()

        return DummyNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            op_type=self.node.op_type.lower(),
        )