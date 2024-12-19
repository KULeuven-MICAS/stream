from stream.onnx_utils import get_split_attribute
from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.split_node import SplitNode


class SplitParser(OnnxOperatorParser):
    """Parses an onnx gather operator into a SplitNode."""

    def generate_node(self):
        # Single predecessor
        predecessors = self.get_node_predecessors()
        if len(predecessors) > 1:
            raise ValueError("Split node should not have more than one input")
        predecessor = predecessors.pop()

        axis = self.get_axis_attribute()
        splits = get_split_attribute(self.node, self.onnx_model)
        input_names = list(self.node.input)
        output_names = list(self.node.output)

        if len(splits) != len(output_names):
            raise ValueError

        return SplitNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            axis=axis,
            splits=splits,
            input_names=input_names,
            output_names=output_names,
        )
