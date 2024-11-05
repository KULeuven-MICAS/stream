from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.utils import get_onnx_tensor_type


def get_onnx_input_shapes(node: NodeProto, onnx_model: ModelProto) -> list[list[int]]:
    """Return the shape of each input operand"""
    input_names = node.input
    input_shapes = [get_onnx_tensor_type(name, onnx_model).shape for name in input_names]
    return input_shapes


def get_onnx_output_shapes(node: NodeProto, onnx_model: ModelProto) -> list[list[int]]:
    """Return the shape of each output operand"""

    output_names = node.output
    output_shapes = [get_onnx_tensor_type(name, onnx_model).shape for name in output_names]
    return output_shapes


def has_asymmetric_input_data(node: NodeProto, onnx_model: ModelProto):
    """Return true iff the node has two inputs and the input nodes have a different shape"""
    if len(node.input) != 2:
        return False

    input_shape1, input_shape2 = get_onnx_input_shapes(node, onnx_model)
    return input_shape1 != input_shape2
