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

logger = logging.getLogger(__name__)


class ConvTransposeParser(OnnxComputeOperatorParser):
    """Parser for ONNX Conv and QLinearConv nodes into LayerNode."""

    OP_TYPE = "convtranspose"

    def get_layer_node_user_format(  # type: ignore
        self,
        kernel_shape: list[int],
        strides: list[int],
        dilations: list[int],
        group_size: int,
        padding: list[int],
        ia_shape: list[int],
        oa_shape: list[int],
    ) -> dict[str, Any]:
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        # convert the data types to precisions based on the onnx definition

        # Equation
        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = f"Layer{self.node_id}"
        data["operator_type"] = ConvTransposeParser.OP_TYPE
        # IMPORTANT: If any of the input loops require padding, they should be defined as the rightmost dimensions in
        # the equation. This is because we construct the dimensionality order and then add the padding to those last
        # dimensions in the order
        if group_size > 1:
            data["equation"] = "O[b][g][k][oy][ox]+=W[g][c][fy][fx]*I[b][g][c][iy][ix]"
        else:
            data["equation"] = "O[b][g][k][oy][ox]+=W[k][c][fy][fx]*I[b][g][c][iy][ix]"

        # Get dimension sizes from input parameters
        assert ia_shape[0] == oa_shape[0], "Batch size is different for input and output activations."
        B = oa_shape[0]
        G = group_size
        K = ceil(oa_shape[1] / G)
        OX = oa_shape[3]
        OY = oa_shape[2]
        C = ceil(ia_shape[1] / G)
        IX = ia_shape[3]
        IY = ia_shape[2]
        FX = kernel_shape[0]
        FY = kernel_shape[1]
        data["loop_dims"] = ["B", "K", "G", "IX", "IY", "C", "FX", "FY"]
        data["loop_sizes"] = [B, K, G, IX, IY, C, FX, FY]

        data["pr_loop_dims"] = ["OX", "OY"]
        data["pr_loop_sizes"] = [OX, OY]
        data["dimension_relations"] = [
            f"ox={strides[0]}*ix+{dilations[0]}*fx",
            f"oy={strides[1]}*iy+{dilations[1]}*fy",
        ]
        data["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}

        # Add information wrt how this conv node's input/output tensors
        # are represented in the onnx model vs how they are represented in the equation above.
        # Because onnx doesn't actually encode the group dimension in a separate dimension
        # but instead keeps it as a "groups" parameter.
        # Concretely, this entry contains for the I and O operand how the G + C/K should be converted
        # to a single "CH" (channel) dimension.

        # Add padding information
        data["padding"] = [
            [padding[0], padding[2]],
            [padding[1], padding[3]],
        ]

        # Find the previous layer(s) that should be this node's parent(s)
        node_inputs = self.node.input
        NUM_INPUTS_EXPECTED = 2
        assert len(node_inputs) >= NUM_INPUTS_EXPECTED, (
            f"Conv should have at least two input names, but has: {node_inputs}."
        )
        (first_input_name, second_input_name) = node_inputs[:2]

        source_list_I = [
            src for (src, src_output_names) in self.nodes_outputs.items() if first_input_name in src_output_names
        ]
        source_list_W = [
            src for (src, src_output_names) in self.nodes_outputs.items() if second_input_name in src_output_names
        ]
        assert len(source_list_I) <= 1
        assert len(source_list_W) <= 1

        source_I = source_list_I[0] if len(source_list_I) == 1 else self.node_id
        source_W = source_list_W[0] if len(source_list_W) == 1 else self.node_id

        data["operand_source"] = {
            "I": source_I,
            "W": source_W,
        }

        return data

    def generate_node(self):
        attrs = self.node.attribute
        kernel_shape: list[int] = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type:ignore
        strides: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type:ignore
        dilations: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type:ignore
        group_size: int = get_attribute_ints_with_name("group", attrs, default=1)  # type:ignore
        padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type:ignore
        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        node_data: dict[str, Any] = self.get_layer_node_user_format(
            kernel_shape,
            strides,
            dilations,
            group_size,
            padding,
            ia_dimension_shape,
            oa_dimension_shape,
        )

        node_factory = LayerNodeFactory(node_data, mapping_data=None)
        node_attrs = node_factory.create_node_attr()
        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            mapping_attr=mapping,
            op_type=ConvTransposeParser.OP_TYPE,
            operand_tensor_reshape=None,
        )
