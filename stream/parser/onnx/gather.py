from onnx import numpy_helper
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser

from stream.workload.gather_node import GatherNode


class GatherParser(ONNXOperatorParser):
    """Parses an onnx gather operator into a GatherNode."""

    def run(self):
        return self.generate_node()

    def generate_node(self):
        predecessors = self.get_node_predecessors()

        axis = self.get_axis_value()
        indices = self.get_indices_value()

        input_names = [self.node.input[0]]
        output_names = [self.node.output[0]]

        return GatherNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            gather_axis=axis,
            gather_indices=indices,
            input_names=input_names,
            output_names=output_names,
        )

    def get_indices_value(self):
        """Find the value of the indices tensor associated with this gather node in ONNX"""
        DEFAULT = 0

        # `indices` is the second input to the node
        indices_tensor_name = self.node.input[1]
        try:
            indices_tensor = next(
                filter(
                    lambda x: x.output[0] == indices_tensor_name and x.op_type == "Constant", self.onnx_model.graph.node
                )
            )
            indices_attr = next(filter(lambda x: x.name == "value", indices_tensor.attribute))
            indices_array = numpy_helper.to_array(indices_attr.t)  # type: ignore
            indices = list(indices_array) if len(indices_array.shape) > 0 else DEFAULT  # type: ignore
        except StopIteration:
            indices = DEFAULT

        return indices

    def get_axis_value(self):
        """Find the value of the axis associated with this gather node in ONNX"""
        # `axis` is an attribute of the node
        try:
            axis_attr = next(filter(lambda x: x.name == "axis", self.node.attribute))
            axis = axis_attr.i
        except StopIteration:
            axis = 0
        return axis
