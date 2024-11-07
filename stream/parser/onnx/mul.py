from typing import Any

from stream.onnx_utils import get_onnx_input_shapes, get_onnx_output_shapes
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser


class MulParser(OnnxComputeOperatorParser):
    """Parses an ONNX operator representing an elementwise operation (Mul) into a ComputationNode."""

    def get_common_and_broadcast_shape(self):
        """This node assumes that the ONNX node has 2 inputs and 1 output. One input shape is identical to the output
        shape, and the other shape can broadcast in dimensions.
        Returns the common shape (in and out) and the broadcast shape"""
        input_shapes = get_onnx_input_shapes(self.node, self.onnx_model)
        output_shapes = get_onnx_output_shapes(self.node, self.onnx_model)

        if len(input_shapes) != 2 or len(output_shapes) != 1:
            raise NotImplementedError

        output_shape = output_shapes.pop()
        if not any(shape == output_shape for shape in input_shapes):
            raise NotImplementedError

        input_shape = output_shape
        input_shapes.remove(output_shape)
        broadcast_shape = input_shapes.pop()

        # e.g. (3,5) * (8,3,5) is ok (broadcast over dim 0), but (3,2) * (8,3,5) is unclear
        for broadcast_size, in_size in zip(reversed(broadcast_shape), reversed(input_shape)):
            if broadcast_size != in_size and broadcast_size != 1:
                raise ValueError

        return input_shape, broadcast_shape

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        common_shape, broadcast_shape = self.get_common_and_broadcast_shape()

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = common_shape

        match len(common_shape):
            case 1:
                loop_dims = ["K"]
            case 2:
                loop_dims = ["D", "K"]
            case 3:
                loop_dims = ["B", "D", "K"]
            case 4:
                loop_dims = ["B", "H", "D", "K"]
            case _:
                raise NotImplementedError

        loop_dims_broadcast = reversed(
            [dim for dim, size in zip(reversed(loop_dims), reversed(broadcast_shape)) if size > 1]
        )

        equation_dims_common = "".join([f"[{dim.lower()}]" for dim in loop_dims])
        equation_dims_broadcast = "".join([f"[{dim.lower()}]" for dim in loop_dims_broadcast])
        equation = f"O{equation_dims_common}+=I{equation_dims_common}*W{equation_dims_broadcast}"

        data["loop_dims"] = loop_dims
        data["equation"] = equation

        return data
