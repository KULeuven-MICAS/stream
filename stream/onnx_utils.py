import numpy as np
from onnx import AttributeProto, ModelProto, NodeProto, numpy_helper
from zigzag.parser.onnx.utils import get_onnx_tensor_type


def get_attribute_as_ints(
    node: NodeProto, attribute_name: str, default: list[int] | int | None = None
) -> list[int] | int:
    """! Return the value of an attribute of given name from the given attributes
    If name does not exist in attrs, the default provided by the caller is used.
    If the caller doesn't supply a default, an error is thrown.

    """
    attrs = node.attribute
    attrs_names = [attr.name for attr in attrs]
    try:
        name_idx = attrs_names.index(attribute_name)
        value = attrs[name_idx]
        attr_type = value.type
        if attr_type == AttributeProto.AttributeType.INT:  # type: ignore
            return int(value.i)
        elif attr_type == AttributeProto.AttributeType.INTS:  # type: ignore
            return list(value.ints)
        elif attr_type == AttributeProto.AttributeType.TENSOR:  # type: ignore
            return list(numpy_helper.to_array(value.t).tolist())  # type: ignore
        else:
            raise NotImplementedError(f"Attribute extraction of type {attr_type} not supported.")
    except ValueError as exc:
        if default is not None:
            return default
        else:
            raise ValueError(
                f"Node {node.name} has no attribute called {attribute_name} and no default. Attributes = {attrs_names}."
            ) from exc


def get_onnx_input_shapes(node: NodeProto, onnx_model: ModelProto) -> list[tuple[int, ...]]:
    """Return the shape of each input operand"""
    input_names = node.input
    input_shapes = [tuple(get_onnx_tensor_type(name, onnx_model).shape) for name in input_names]
    return input_shapes


def get_onnx_output_shapes(node: NodeProto, onnx_model: ModelProto) -> list[tuple[int, ...]]:
    """Return the shape of each output operand"""

    output_names = node.output
    output_shapes = [tuple(get_onnx_tensor_type(name, onnx_model).shape) for name in output_names]
    return output_shapes


def has_asymmetric_input_data(node: NodeProto, onnx_model: ModelProto):
    """Return true iff the node has two inputs and the input nodes have a different shape"""
    EXPECTED_INPUT_LENGTH = 2
    if len(node.input) != EXPECTED_INPUT_LENGTH:
        return False

    input_shape1, input_shape2 = get_onnx_input_shapes(node, onnx_model)
    return input_shape1 != input_shape2


def get_constant_tensor_int(onnx_model: ModelProto, constant_output_name: str):
    """In some cases, the constants to a node (e.g. slice and split indices) are saved as tensors within a constant
    node. The output name of the constant nodes corresponds to the input name of the node that uses this constant
    tensor."""

    for node in onnx_model.graph.node:
        if node.op_type == "Constant" and node.output[0] == constant_output_name:
            for attr in node.attribute:
                if attr.name == "value":
                    tensor = attr.t  # This is an ONNX TensorProto
                    # Decode tensor to a numpy array
                    array = np.frombuffer(tensor.raw_data, dtype=int)
                    array = array.reshape([dim for dim in tensor.dims])

                    return [int(i) for i in array]

    raise ValueError(f"Cannot find {constant_output_name}")


def get_axis_attribute(node: NodeProto):
    """Find the value of the axis associated with this ONNX node"""
    ATTR_NAME = "axis"
    DEFAULT = -1

    try:
        value = get_attribute_as_ints(node, ATTR_NAME)
    except ValueError:
        return DEFAULT
    if not isinstance(value, int):
        raise ValueError(f"{ATTR_NAME} attribute as list of ints not supported")
    return value


def get_split_attribute(node: NodeProto, onnx_model: ModelProto):
    output_name = next(n for n in node.input if "split" in n.lower())
    return get_constant_tensor_int(onnx_model, output_name)


def get_slice_attributes(node: NodeProto, onnx_model: ModelProto):
    """Get the `starts`, `ends`, `axes` and `steps` tensors for a slice node.
    NOTE: this assumes that the attributes are given as inputs in this order"""
    EXPECTED_INPUT_LENGTH = 5
    if len(node.input) != EXPECTED_INPUT_LENGTH:
        raise NotImplementedError("Unsure how to get slice attributes from Node")

    starts_output_name, ends_output_name, axes_output_name, steps_output_name = node.input[1:5]

    starts_value = get_constant_tensor_int(onnx_model, starts_output_name)
    ends_value = get_constant_tensor_int(onnx_model, ends_output_name)
    axes_value = get_constant_tensor_int(onnx_model, axes_output_name)
    steps_value = get_constant_tensor_int(onnx_model, steps_output_name)
    return starts_value, ends_value, axes_value, steps_value
