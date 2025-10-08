from onnx import numpy_helper

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.squeeze_node import SqueezeNode


class SqueezeParser(OnnxOperatorParser):
    """Parses an ONNX Unsqueeze operator"""

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        axes = self.get_axis_value()
        input_names = list(self.node.input)
        return SqueezeNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessors[0],
            input_names=input_names,
            squeeze_axes=axes,
        )

    def get_axis_value(self):
        """Find the value of the axes tensor associated with this gather node in ONNX"""
        DEFAULT = 0

        # `axes` is the second input to the node
        axes_tensor_name = self.node.input[1]
        # Try to find the axes in the graph nodes first
        try:
            axes_tensor = next(
                filter(
                    lambda x: x.output[0] == axes_tensor_name and x.op_type == "Constant", self.onnx_model.graph.node
                )
            )
            axes_attr = next(filter(lambda x: x.name == "value", axes_tensor.attribute))
            axes_array = numpy_helper.to_array(axes_attr.t)  # type: ignore
            axes = tuple(axes_array) if len(axes_array.shape) > 0 else DEFAULT  # type: ignore
        except StopIteration:
            # Try to find the axes in the initializers
            try:
                axes_tensor = next(filter(lambda x: x.name == axes_tensor_name, self.onnx_model.graph.initializer))
                axes_array = numpy_helper.to_array(axes_tensor)
                axes = tuple(axes_array) if len(axes_array.shape) > 0 else DEFAULT
            except StopIteration:
                axes = DEFAULT

        return axes
