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

        # Extract extra attributes
        attrs = self.node.attribute
        kernel_shape: list[int] = get_attribute_ints_with_name("kernel_shape", attrs, default=None)  # type:ignore
        strides: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type:ignore
        dilations: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type:ignore
        group_size: int = get_attribute_ints_with_name("group", attrs, default=1)  # type:ignore
        padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type:ignore

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = ConvParser.OP_TYPE
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["operand_source"] = self.get_operand_source_user_format(predecessors)

        # 1D Conv case: append dimensions of size 1 so equation holds. Conv in FY dimension
        is_1d_conv = len(kernel_shape) == 1

        # Get dimension sizes from input parameters
        assert input_shape[0] == output_shape[0], "Batch size is different for input and output activations."
        B = output_shape[0]
        G = group_size
        K = ceil(output_shape[1] / G)
        C = ceil(input_shape[1] / G)
        FX = kernel_shape[0]
        IX = input_shape[2]
        OX = output_shape[2]

        weight_dim = "g" if group_size > 1 else "k"

        # IMPORTANT: If any of the input loops require padding, they should be defined as the rightmost dimensions in
        # the equation. This is because we construct the dimensionality order and then add the padding to those last
        # dimensions in the order.
        # Add information wrt how this conv node's input/output tensors are represented in the onnx model vs how they
        # are represented in the equation. Because onnx doesn't actually encode the group dimension in a separate
        # dimension but instead keeps it as a "groups" parameter. Concretely, this entry contains for the I and O
        # operand how the G + C/K should be converted to a single "CH" (channel) dimension.

        if is_1d_conv:
            # No FY, OY, IY
            loop_size_dict = {"B": B, "K": K, "G": G, "OX": OX, "C": C, "FX": FX}
            data["equation"] = f"O[b][g][k][ox]+=W[{weight_dim}][c][fx]*I[b][g][c][ix]"
            data["pr_loop_dims"] = ["IX"]
            data["pr_loop_sizes"] = [IX]
            data["dimension_relations"] = [
                f"ix={strides[0]}*ox+{dilations[0]}*fx",
            ]
            data["padding"] = [
                [padding[0], padding[1]],
            ]
        else:
            assert len(input_shape) == 4 and len(output_shape) == 4 and len(padding) == 4 and len(strides) == 2
            FY = kernel_shape[1]
            IY = input_shape[3]
            OY = output_shape[3]
            loop_size_dict = {"B": B, "K": K, "G": G, "OX": OX, "C": C, "FX": FX, "OY": OY, "FY": FY}
            data["equation"] = f"O[b][g][k][oy][ox]+=W[{weight_dim}][c][fy][fx]*I[b][g][c][iy][ix]"
            data["pr_loop_dims"] = ["IX", "IY"]
            data["pr_loop_sizes"] = [IX, IY]
            data["dimension_relations"] = [
                f"ix={strides[0]}*ox+{dilations[0]}*fx",
                f"iy={strides[1]}*oy+{dilations[1]}*fy",
            ]
            data["padding"] = [
                [padding[0], padding[2]],
                [padding[1], padding[3]],
            ]

        # Remove C/K if they have size 1
        for dim in ["C", "K"]:
            if loop_size_dict[dim] == 1:
                del loop_size_dict[dim]
                # Remove from equation
                data["equation"] = data["equation"].replace(f"[{dim.lower()}]", "")

        data["loop_dims"] = list(loop_size_dict.keys())
        data["loop_sizes"] = list(loop_size_dict.values())

        return data

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
