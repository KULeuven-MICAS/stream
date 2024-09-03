from zigzag.parser.onnx.utils import OnnxTensorCategory, get_onnx_tensor_type

from stream.classes.io.onnx.operator_parser import OnnxOperatorParser
from stream.classes.workload.concat_node import ConcatNode


class ConcatParser(OnnxOperatorParser):
    """Parses an onnx gather operator into a ConcatNode."""

    def run(self):
        return self.generate_node()

    def generate_node(self):
        predecessors = self.get_node_predecessors()

        axis = self.get_axis_value()
        output_names = [self.node.output[0]]

        input_1, input_2 = self.node.input[0], self.node.input[1]

        try:  # Try first one as constant input
            constant_tensor = get_onnx_tensor_type(input_1, self.onnx_model)
            if constant_tensor.category != OnnxTensorCategory.HIDDEN or "constant" not in input_1.lower():
                raise ValueError

            constant_shape = tuple(constant_tensor.shape)
            variable_input_first = True
            input_names = [input_2]
        except ValueError:  # Try second one as constant input
            constant_tensor = get_onnx_tensor_type(input_2, self.onnx_model)
            if constant_tensor.category != OnnxTensorCategory.HIDDEN or "constant" not in input_2.lower():
                raise ValueError

            constant_shape = tuple(constant_tensor.shape)
            variable_input_first = True
            input_names = [input_1]

        return ConcatNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            axis=axis,
            constant_shape=constant_shape,
            variable_input_first=variable_input_first,
            input_names=input_names,
            output_names=output_names,
        )

    def get_axis_value(self):
        """Find the value of the axis associated with this concat node in ONNX"""
        # `axis` is an attribute of the node
        try:
            axis_attr = next(filter(lambda x: x.name == "axis", self.node.attribute))
            return axis_attr.i
        except StopIteration:
            raise ValueError("Axis attribute not found in ONNX node")
