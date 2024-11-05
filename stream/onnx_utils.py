from onnx import AttributeProto, ModelProto, NodeProto, numpy_helper
from zigzag.parser.onnx.utils import get_onnx_tensor_type

import numpy as np
import onnx


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
                f"Node {node.name} has no attribute called {attribute_name} and no default was given. Attributes = {attrs_names}."
            ) from exc


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


def get_axis_attribute(node: NodeProto):
    """Find the value of the axis associated with this ONNX node"""
    ATTR_NAME = "axis"

    value = get_attribute_as_ints(node, ATTR_NAME)
    if not isinstance(value, int):
        raise ValueError(f"{ATTR_NAME} attribute as list of ints not supported")
    return value


def get_split_attribute(node: NodeProto, onnx_model: ModelProto):
    # ATTR_NAME = "split"

    output_name = next(n for n in node.input if "split" in n.lower())

    for node in onnx_model.graph.node:
        if node.op_type == "Constant" and node.output[0] == output_name:
            for attr in node.attribute:
                if attr.name == "value":
                    tensor = attr.t  # This is an ONNX TensorProto
                    # Decode tensor to a numpy array
                    array = np.frombuffer(tensor.raw_data, dtype=int)
                    array = array.reshape([dim for dim in tensor.dims])

                    return [int(i) for i in array]

    raise ValueError
