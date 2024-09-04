from zigzag.parser.onnx.utils import get_attribute_ints_with_name

from stream.classes.io.onnx.operator_parser import OnnxOperatorParser
from stream.classes.workload.flatten_node import FlattenNode


class FlattenParser(OnnxOperatorParser):
    """Parses an onnx flatten operator into a FlattenNode."""

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        assert len(predecessors) == 1
        predecessor = predecessors[0]

        attrs = self.node.attribute
        # Get the axis which indicates how to flatten the input tensor
        axis: int | None = get_attribute_ints_with_name("axis", attrs, default=None)  # type: ignore
        input_names = [self.node.input[0]]
        output_names = [self.node.output[0]]
        return FlattenNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            axis=axis,
            input_names=input_names,
            output_names=output_names,
        )
