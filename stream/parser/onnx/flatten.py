from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.flatten_node import FlattenNode


class FlattenParser(OnnxOperatorParser):
    """Parses an onnx flatten operator into a FlattenNode."""

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        assert len(predecessors) == 1
        predecessor = predecessors[0]

        input_names = list(self.node.input)
        axis = self.get_axis_attribute()

        return FlattenNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            axis=axis,
            input_names=input_names,
        )
