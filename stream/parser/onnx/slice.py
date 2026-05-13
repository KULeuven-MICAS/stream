from stream.onnx_utils import get_slice_attributes
from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.slice_node import SliceNode


class SliceParser(OnnxOperatorParser):
    """Parses an onnx gather operator into a SliceNode."""

    def generate_node(self):
        if len(self.node.output) > 1:
            raise NotImplementedError("Slice node with multiple output slices not yet supported.")

        # Single predecessor
        predecessors = self.get_node_predecessors()
        if len(predecessors) > 1:
            raise ValueError("Slice node should not have more than one input")
        predecessor = predecessors.pop()

        starts_value, ends_value, axes_value, steps_value = get_slice_attributes(self.node, self.onnx_model)
        input_names = list(self.node.input)
        output_names = list(self.node.output)

        return SliceNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor,
            starts=starts_value,
            ends=ends_value,
            axes=axes_value,
            steps=steps_value,
            input_names=input_names,
            output_names=output_names,
        )
