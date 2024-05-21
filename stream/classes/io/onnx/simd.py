from typing import Any
from onnx import ModelProto, NodeProto
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.simd_node import SimdNode
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import (
    get_node_input_output_dimension_shapes,
)
from zigzag.parser.workload_factory import LayerNodeFactory


class SimdParser(ONNXOperatorParser):
    """Parses an ONNXOperatorParser operator representing an elementwise operation (simd) into a SimdNode.
    e.g. Add, etc.
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
        self.op_type = self.node.op_type  # .lower()
        self.node_name = f"Layer{self.node_id}"

    def run(self):
        return self.generate_layer_node_for_simd()

    def get_layer_node_input_format(self, ia_shape: list[int], oa_shape: list[int]):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        For the pooling node, we pick K as the "channel" dimension. It should be equal to C anyways.
        """
        # convert the data types to precisions based on the onnx definition

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node_name
        data["operator_type"] = self.op_type
        data["equation"] = "O[b][k][oy][ox]+=I[b][k][oy][ox]*W[b][k][oy][ox]"
        # data["equation"] = "O[b][k][oy][ox]+=A[b][k][oy][ox]*B[b][k][oy][ox]"

        # Get dimension sizes from input parameters
        assert ia_shape == oa_shape, "Input and output of simd operation should be identical."
        B = oa_shape[0]
        K = oa_shape[1]
        OX = oa_shape[2]
        OY = oa_shape[3]
        data["loop_dims"] = ["B", "K", "OX", "OY"]
        data["loop_sizes"] = [B, K, OX, OY]
        data["operand_precision"] = {"O": 8, "O_final": 8, "I": 8, "W": 8}
        data["dimension_relations"] = []

        # Find the previous layer(s) that should be this node's parent(s)
        node_inputs = self.node.input
        assert len(node_inputs) == 2, f"Simd layer {self.node.name} doesn't have 2 inputs: {node_inputs}."
        (first_input_name, second_input_name) = node_inputs

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
        # if any((input_name_A in output_names for output_names in self.nodes_outputs.values())):
        #     memory_operand_A = "I1"
        # else:
        #     memory_operand_A = "I2"
        #     constant_operands.append("A")
        # if any((input_name_B in output_names for output_names in self.nodes_outputs.values())):
        #     memory_operand_B = "I2"  # TODO: Change this to I1 and fix subsequent uses
        # else:
        #     memory_operand_B = "I2"
        #     constant_operands.append("B")

        data["operand_source"] = {
            "I": source_I,
            "W": source_W,
        }
        # data["memory_operand_links"] = {
        #     "O": "O",
        #     "B": memory_operand_B,
        #     "A": memory_operand_A,
        # }

        return data

    def generate_layer_node_for_simd(self):
        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        node_data: dict[str, Any] = self.get_layer_node_input_format(ia_dimension_shape, oa_dimension_shape)
        node_factory = LayerNodeFactory(node_data, self.mapping_data)
        node_attrs = node_factory.create_node_attr()

        # Override spatial mapping by the one defined in the core's dataflows
        core_allocation = node_attrs.core_allocation
        spatial_mapping = self.accelerator.get_spatial_mapping_from_core(core_allocation)
        node_attrs.spatial_mapping = spatial_mapping

        # Get the node's input(s) and output(s) tensor names
        node_input_names = list(self.node.input)
        node_output_names = list(self.node.output)

        return SimdNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            input_names=node_input_names,
            output_names=node_output_names,
            op_type=self.op_type,
        )
