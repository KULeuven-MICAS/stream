from zigzag.parser.onnx.utils import OnnxTensorCategory, get_onnx_tensor_type

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.concat_node import ConcatConstantNode


class ConcatParser(OnnxOperatorParser):
    """Parses an ONNX Concat operator with one constant input into a ConcatConstantNode.
    # TODO also parse concat nodes with non-constant inputs
    """

    def get_axis_value(self):
        """Find the value of the axis associated with this concat node in ONNX"""
        axis_attr = "axis"
        # `axis` is an attribute of the node
        try:
            axis_attr = next(filter(lambda x: x.name == axis_attr, self.node.attribute))
            return axis_attr.i
        except StopIteration as exc:
            raise ValueError("Axis attribute not found in ONNX node") from exc

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        axis = self.get_axis_value()
        input_names = list(self.node.input)

        input_1, input_2 = self.node.input[0], self.node.input[1]

        try:  # Try first one as constant input
            constant_tensor = get_onnx_tensor_type(input_1, self.onnx_model)
            if constant_tensor.category != OnnxTensorCategory.HIDDEN or "constant" not in input_1.lower():
                raise ValueError

            constant_shape = tuple(constant_tensor.shape)
            variable_input_first = True
        except ValueError as exc:  # Try second one as constant input
            constant_tensor = get_onnx_tensor_type(input_2, self.onnx_model)
            if constant_tensor.category != OnnxTensorCategory.HIDDEN or "constant" not in input_2.lower():
                raise ValueError("Second input is not a constant tensor") from exc

            constant_shape = tuple(constant_tensor.shape)
            variable_input_first = True

        return ConcatConstantNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            axis=axis,
            constant_shape=constant_shape,
            variable_input_first=variable_input_first,
            input_names=input_names,
        )
