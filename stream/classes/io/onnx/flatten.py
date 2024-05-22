from networkx import predecessor
from stream.classes.workload.flatten_node import FlattenNode
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import get_attribute_ints_with_name


class FlattenParser(ONNXOperatorParser):
    """Parses an onnx flatten operator into a FlattenNode."""

    def run(self):
        return self.generate_flatten_node()

    def generate_flatten_node(self):
        # Get the predecessors of this node
        predecessors: list[int] = []
        for node_input in self.node.input:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    predecessors.append(n)
        assert len(predecessors) <= 1
        predecessor = predecessors[0] if len(predecessors) == 1 else None

        attrs = self.node.attribute
        # Get the axis which indicates how to flatten the input tensor
        axis: int | None = get_attribute_ints_with_name("axis", attrs, default=None)  # type: ignore
        input_names = [self.node.input[0]]
        output_names = [self.node.output[0]]
        return FlattenNode(
            node_id=self.node_id,
            node_name="",
            predecessor=predecessor,
            axis=axis,
            input_names=input_names,
            output_names=output_names,
        )
