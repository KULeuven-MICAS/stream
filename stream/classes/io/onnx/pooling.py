from typing import Any
from onnx import ModelProto, NodeProto
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.pooling_node import PoolingNode
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes,
)
from zigzag.parser.workload_factory import LayerNodeFactory


class PoolingParser(ONNXOperatorParser):
    """Parses an onnx pooling operator into a PoolingNode.
    e.g. MaxPool, AveragePool, etc.
    """

    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        mapping_data: list[dict[str, Any]],
        onnx_model: ModelProto,
        accelerator: Accelerator,
    ) -> None:
        super().__init__(node_id, node, nodes_outputs, onnx_model)
        self.onnx_model = onnx_model
        self.mapping_data = mapping_data
        self.accelerator = accelerator
        self.op_type = self.node.op_type
        self.node_name = f"Layer{self.node_id}"

    def run(self):
        return self.generate_layer_node_for_pooling()

    def get_kernel_shape(self, attrs, ia_dimension_shape) -> list[int]:
        """Return the kernel shape of the pooling operator depending on the type of node

        Args:
            attrs (_type_): _description_
        """
        if self.node.op_type in ["MaxPool", "AveragePool"]:
            # Find kernel shape in attrs
            kernel_shape = get_attribute_ints_with_name("kernel_shape", attrs, default=None)
        elif self.node.op_type in ["GlobalMaxPool", "GlobalAveragePool"]:
            assert len(ia_dimension_shape) == 4  # assume the last two dimensions are the pooling kernel dimensions
            kernel_shape: list[int] = [ia_dimension_shape[2], ia_dimension_shape[3]]
        else:
            raise NotImplementedError(
                f"Pooling node kernel shape extraction not implemented for operand type {self.node.op_type}."
            )
        return kernel_shape

    def get_layer_node_input_format(
        self,
        kernel_shape: list[int],
        strides: list[int],
        dilations: list[int],
        padding: list[int],
        ia_shape: list[int],
        oa_shape: list[int],
    ) -> dict[str, Any]:
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        For the pooling node, we pick K as the "channel" dimension. It should be equal to C anyways.
        """
        # convert the data types to precisions based on the onnx definition

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node_name
        data["operator_type"] = self.op_type
        data["equation"] = "O[b][k][oy][ox]+=W[fy][fx]*I[b][k][iy][ix]"
        # Get dimension sizes from input parameters
        assert ia_shape[0] == oa_shape[0], "Batch size is different for input and output activations."
        B = oa_shape[0]
        K = oa_shape[1]
        OX = oa_shape[2]
        OY = oa_shape[3]
        C = ia_shape[1]
        IX = ia_shape[2]
        IY = ia_shape[3]
        FX = kernel_shape[0]
        FY = kernel_shape[1]
        assert K == C, f"Input and output channels not equal for pooling node {self.node.name}."
        data["loop_dims"] = ["B", "K", "OX", "OY", "C", "FX", "FY"]
        data["loop_sizes"] = [B, K, OX, OY, C, FX, FY]
        data["pr_loop_dims"] = ["IX", "IY"]
        data["pr_loop_sizes"] = [IX, IY]
        data["dimension_relations"] = [
            f"ix={strides[0]}*ox+{dilations[0]}*fx",
            f"iy={strides[1]}*oy+{dilations[1]}*fy",
        ]
        data["operand_precision"] = {"O": 8, "O_final": 8, "W": 0, "I": 8}

        # Find the previous layer(s) that should be this node's parent(s)
        data["operand_source"] = {}
        node_inputs = self.node.input
        preds: list[int] = []
        for node_input in node_inputs:
            for n in self.nodes_outputs:
                if node_input in self.nodes_outputs[n]:
                    preds.append(n)
        assert len(preds) <= 1
        if len(preds) == 1:
            data["operand_source"]["I"] = preds[0]

        data["operand_source"]["W"] = self.node_id

        data["padding"] = [
            [padding[0], padding[2]],
            [padding[1], padding[3]],
        ]

        return data

    def generate_layer_node_for_pooling(self):

        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        attrs = self.node.attribute
        kernel_shape = self.get_kernel_shape(attrs, ia_dimension_shape)
        strides: list[int] = get_attribute_ints_with_name("strides", attrs, default=[1, 1])  # type: ignore
        dilations: list[int] = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])  # type: ignore
        padding: list[int] = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])  # type: ignore

        node_data: dict[str, Any] = self.get_layer_node_input_format(
            kernel_shape,
            strides,
            dilations,
            padding,
            ia_dimension_shape,
            oa_dimension_shape,
        )
        node_factory = LayerNodeFactory(node_data, self.mapping_data)
        node_attrs = node_factory.create_node_attr()

        # Override spatial mapping by the one defined in the core's dataflows
        core_allocation = node_attrs.core_allocation
        spatial_mapping = self.accelerator.get_spatial_mapping_from_core(core_allocation)
        node_attrs.spatial_mapping = spatial_mapping

        # Get the node's input(s) and output(s) tensor names
        node_input_names = list(self.node.input)
        node_output_names = list(self.node.output)

        return PoolingNode(
            node_id=self.node_id,
            node_name=self.node_name,
            node_attr=node_attrs,
            input_names=node_input_names,
            output_names=node_output_names,
        )
