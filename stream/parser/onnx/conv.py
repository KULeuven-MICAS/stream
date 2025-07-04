import logging
from math import ceil
from typing import Any

from zigzag.parser.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes,
)
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import InterCoreMappingAttributes

logger = logging.getLogger(__name__)


class ConvParser(OnnxComputeOperatorParser):
    """Parser for ONNX Conv and QLinearConv nodes into LayerNode."""

    OP_TYPE = "conv"

    def get_layer_node_user_format(
        self,
        input_shape: list[int],
        output_shape: list[int],
        mapping: InterCoreMappingAttributes | None = None,
    ) -> dict[str, Any]:
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        predecessors = self.get_node_predecessors()
        attrs = self.node.attribute

        kernel_shape = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type:ignore
        strides = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type:ignore
        dilations = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type:ignore
        group_size = get_attribute_ints_with_name("group", attrs, default=1)  # type:ignore
        padding = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type:ignore
        # Check that kernel_shape, strides, dilations, group_size and padding are list of ints
        assert isinstance(kernel_shape, list), "ConvParser: kernel_shape must be a list of ints."
        assert isinstance(strides, list), "ConvParser: strides must be a list of ints."
        assert isinstance(dilations, list), "ConvParser: dilations must be a list of ints."
        assert isinstance(group_size, int), "ConvParser: group_size must be an int."
        assert isinstance(padding, list), "ConvParser: padding must be a list of ints."

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = ConvParser.OP_TYPE
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["operand_source"] = self.get_operand_source_user_format(predecessors)

        is_1d_conv = len(kernel_shape) == 1

        if is_1d_conv:
            loop_size_dict, equation, pr_loop_dims, pr_loop_sizes, dimension_relations, padding_info = (
                self._get_1d_conv_params(
                    input_shape, output_shape, kernel_shape, strides, dilations, group_size, padding
                )
            )
        else:
            loop_size_dict, equation, pr_loop_dims, pr_loop_sizes, dimension_relations, padding_info = (
                self._get_2d_conv_params(
                    input_shape, output_shape, kernel_shape, strides, dilations, group_size, padding
                )
            )

        # Remove C/K if they have size 1
        equation = self._remove_singleton_channel_dims(loop_size_dict, equation)

        data["equation"] = equation
        data["pr_loop_dims"] = pr_loop_dims
        data["pr_loop_sizes"] = pr_loop_sizes
        data["dimension_relations"] = dimension_relations
        data["padding"] = padding_info
        data["loop_dims"] = list(loop_size_dict.keys())
        data["loop_sizes"] = list(loop_size_dict.values())

        return data

    def _get_1d_conv_params(
        self,
        input_shape: list[int],
        output_shape: list[int],
        kernel_shape: list[int],
        strides: list[int],
        dilations: list[int],
        group_size: int,
        padding: list[int],
    ):
        B = output_shape[0]
        G = group_size
        K = ceil(output_shape[1] / G)
        C = ceil(input_shape[1] / G)
        FX = kernel_shape[0]
        IX = input_shape[2]
        OX = output_shape[2]
        weight_dim = "g" if group_size > 1 else "k"

        loop_size_dict = {"B": B, "K": K, "G": G, "OX": OX, "C": C, "FX": FX}
        equation = f"O[b][g][k][ox]+=W[{weight_dim}][c][fx]*I[b][g][c][ix]"
        pr_loop_dims = ["IX"]
        pr_loop_sizes = [IX]
        dimension_relations = [f"ix={strides[0]}*ox+{dilations[0]}*fx"]
        padding_info = [[padding[0], padding[1]]]

        return loop_size_dict, equation, pr_loop_dims, pr_loop_sizes, dimension_relations, padding_info

    def _get_2d_conv_params(
        self,
        input_shape: list[int],
        output_shape: list[int],
        kernel_shape: list[int],
        strides: list[int],
        dilations: list[int],
        group_size: int,
        padding: list[int],
    ):
        EXPECTED_INPUT_SHAPE_LENGTH = 4
        EXPECTED_OUTPUT_SHAPE_LENGTH = 4
        EXPECTED_PADDING_LENGTH = 4
        EXPECTED_STRIDES_LENGTH = 2
        assert (
            len(input_shape) == EXPECTED_INPUT_SHAPE_LENGTH
            and len(output_shape) == EXPECTED_OUTPUT_SHAPE_LENGTH
            and len(padding) == EXPECTED_PADDING_LENGTH
            and len(strides) == EXPECTED_STRIDES_LENGTH
        ), "ConvParser: Input and output shapes, padding and strides must have the expected lengths."

        B = output_shape[0]
        G = group_size
        K = ceil(output_shape[1] / G)
        C = ceil(input_shape[1] / G)
        FX = kernel_shape[0]
        FY = kernel_shape[1]
        IX = input_shape[2]
        IY = input_shape[3]
        OX = output_shape[2]
        OY = output_shape[3]
        weight_dim = "g" if group_size > 1 else "k"

        loop_size_dict = {"B": B, "K": K, "G": G, "OX": OX, "C": C, "FX": FX, "OY": OY, "FY": FY}
        equation = f"O[b][g][k][oy][ox]+=W[{weight_dim}][c][fy][fx]*I[b][g][c][iy][ix]"
        pr_loop_dims = ["IX", "IY"]
        pr_loop_sizes = [IX, IY]
        dimension_relations = [
            f"ix={strides[0]}*ox+{dilations[0]}*fx",
            f"iy={strides[1]}*oy+{dilations[1]}*fy",
        ]
        padding_info = [
            [padding[0], padding[2]],
            [padding[1], padding[3]],
        ]

        return loop_size_dict, equation, pr_loop_dims, pr_loop_sizes, dimension_relations, padding_info

    def _remove_singleton_channel_dims(self, loop_size_dict: dict, equation: str) -> str:
        for dim in ["C", "K"]:
            if loop_size_dict.get(dim, None) == 1:
                del loop_size_dict[dim]
                equation = equation.replace(f"[{dim.lower()}]", "")
        return equation

    def generate_node(self):
        # Get the input and output activation shapes
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        node_data: dict[str, Any] = self.get_layer_node_user_format(input_shape, output_shape)

        node_factory = LayerNodeFactory(node_data, mapping_data=None)
        node_attrs = node_factory.create_node_attr()
        mapping = self.get_mapping_this_node()
        input_names = list(self.node.input)

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            mapping_attr=mapping,
            op_type=ConvParser.OP_TYPE,
            operand_tensor_reshape=None,
            input_names=input_names,
        )
