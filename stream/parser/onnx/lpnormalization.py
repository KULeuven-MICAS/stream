from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.lpnormalization_node import LpNormalizationNode


class LpNormalizationParser(OnnxOperatorParser):
    """Parses an onnx reshape operator into a LpNormalizationNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        raise NotImplementedError

        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def generate_node(self):
        input_names = list(self.node.input)

        # Get the predecessors of this node
        # TODO use superclass' `get_node_predecessors`
        predecessors = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)

        node_obj = LpNormalizationNode(
            node_id=self.node_id,
            node_name=self.node_name,  # type: ignore
            predecessor=self.predecessor,  # type: ignore
            input_names=input_names,  # type: ignore
        )
        return node_obj
