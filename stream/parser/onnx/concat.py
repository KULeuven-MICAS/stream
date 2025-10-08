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

        input_1_tensor = get_onnx_tensor_type(input_1, self.onnx_model)
        input_2_tensor = get_onnx_tensor_type(input_2, self.onnx_model)

        input_1_shape = tuple(input_1_tensor.shape)
        input_2_shape = tuple(input_2_tensor.shape)

        # Find if any input is constant and update the state with it
        if input_1_tensor.category is OnnxTensorCategory.CONSTANT:
            variable_input_first = True
            mode = "constant"
        elif input_2_tensor.category is OnnxTensorCategory.CONSTANT:
            variable_input_first = False
            mode = "constant"
        elif (
            input_2_tensor.category is OnnxTensorCategory.HIDDEN
            and input_1_tensor.category is OnnxTensorCategory.HIDDEN
        ):
            variable_input_first = False
            mode = "variable"

        state = (mode, variable_input_first)
        return ConcatConstantNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            axis=axis,
            input_names=input_names,
            state=state,
            input_1_shape=input_1_shape,
            input_2_shape=input_2_shape,
        )
