import logging
import re
from typing import Any

from stream.onnx_utils import get_onnx_input_shapes, get_onnx_output_shapes
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser

logger = logging.getLogger(__name__)


class EinsumParser(OnnxComputeOperatorParser):

    def get_einsum_equation(self):
        ATTR_NAME = "equation"

        attrs_names = [attr.name for attr in self.node.attribute]
        name_idx = attrs_names.index(ATTR_NAME)
        value = self.node.attribute[name_idx]
        return str(value)

    def get_layer_dims_per_op(self):
        einsum_equation = self.get_einsum_equation()

        return re.split(",|->", einsum_equation)

    def get_layer_equation(self, layer_dims_per_op: list[str]):
        def put_in_brackets(s: str):
            """e.g. `abc` -> `[a][b][c]"""
            return "".join([f"[{char}]" for char in s])

        if len(layer_dims_per_op) != 3:
            raise NotImplementedError

        dims_I, dims_W, dims_O = layer_dims_per_op
        equation = f"O{put_in_brackets(dims_O)}+=I{put_in_brackets(dims_I)}*{put_in_brackets(dims_W)}"
        return equation

    # def get_layer_dims(self, layer_dims_per_op: list[str]):
    #     all_dims = {char.upper() for group in layer_dims_per_op for char in group}
    #     return list(all_dims)

    def get_layer_dim_sizes_dict(self, layer_dims_per_op: list[str]):
        input_shapes = get_onnx_input_shapes(self.node, self.onnx_model)
        output_shapes = get_onnx_output_shapes(self.node, self.onnx_model)

        if len(output_shapes) != 1:
            raise ValueError("Einsum with more than one output not supported")

        shapes = input_shapes + output_shapes

        if len(layer_dims_per_op) != len(shapes):
            raise ValueError("Einsum equation has more parts than node inputs")

        sizes_dict: dict[str, int] = {}
        for layer_dims, sizes in zip(layer_dims_per_op, shapes):
            if len(layer_dims) != len(sizes):
                # TODO is the order of the equation guaranteed to be the same as the input order?
                raise ValueError(f"Einsum equation part {layer_dims} and operand input shape {sizes} do not match")
            for layer_dim, size in zip(layer_dims.upper(), sizes):
                if layer_dim not in sizes_dict:
                    sizes_dict[layer_dim] = size
                else:
                    if sizes_dict[layer_dim] != size:
                        raise ValueError(f"Not clear what the size of {layer_dim} is in Einsum")

        return sizes_dict

    def get_layer_node_user_format(
        self,
        input_shape: list[int],  # Argument required because of a caller function in superclass
        output_shape: list[int],  # TODO put shape logic in this method for all `OnnxComputeOperatorParser` subclasses
    ) -> dict[str, Any]:
        """! Generate layer data in user input format for Einsum."""
        predecessors = self.get_node_predecessors()

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["dimension_relations"] = []
        data["operand_source"] = self.get_operand_source_user_format(predecessors)
        data["operand_precision"] = self.get_operand_precision_user_format()

        #
        layer_dims_per_op = self.get_layer_dims_per_op()
        sizes_dict = self.get_layer_dim_sizes_dict(layer_dims_per_op)

        data["loop_dims"] = list(sizes_dict.keys())
        data["loop_sizes"] = list(sizes_dict.values())
        data["equation"] = self.get_layer_equation(layer_dims_per_op)

        return data
