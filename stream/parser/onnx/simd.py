from typing import Any

from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import (
    get_node_input_output_dimension_shapes,
)
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.hardware.architecture.accelerator import Accelerator
from stream.workload.simd_node import SimdNode


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
        return self.generate_node()

    def get_layer_node_input_format(self, ia_shape: list[int], oa_shape: list[int]):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        For the pooling node, we pick K as the "channel" dimension. It should be equal to C anyways.
        """
        assert ia_shape == oa_shape, "Input and output of simd operation should be identical."

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node_name
        data["operator_type"] = self.op_type

        match len(oa_shape):
            case 1:
                data["equation"] = "O[ox]+=I[ox]*W[ox]"
                data["loop_dims"] = ["OX"]
                data["loop_sizes"] = oa_shape
            case 2:
                data["equation"] = "O[oy][ox]+=I[oy][ox]*W[oy][ox]"
                data["loop_dims"] = ["OX", "OY"]
                data["loop_sizes"] = oa_shape
            case 3:
                data["equation"] = "O[b][oy][ox]+=I[b][oy][ox]*W[b][oy][ox]"
                data["loop_dims"] = ["B", "OX", "OY"]
                data["loop_sizes"] = oa_shape
            case 4:
                data["equation"] = "O[b][k][oy][ox]+=I[b][k][oy][ox]*W[b][k][oy][ox]"
                data["loop_dims"] = ["B", "K", "OX", "OY"]
                data["loop_sizes"] = oa_shape
            case _:
                raise NotImplementedError

        data["dimension_relations"] = []

        predecessors = self.get_node_predecessors()
        act_precision = self.get_activation_precision()
        weight_precision = self.get_weight_precision()
        intermediate_output_precision = self.get_intermediate_output_precision()
        match len(predecessors):
            case 0:
                # No source operands -> assume one is constant
                # TODO should this be 2?
                data["operand_source"] = {"W": self.node_id}
                data["operand_precision"] = {
                    "W": weight_precision,
                    "I": act_precision,
                    "O_final": act_precision,
                    "O": intermediate_output_precision,
                }
            case 1:
                # One source operand, one constant
                data["operand_source"] = {"W": self.node_id, "I": predecessors[0]}
                data["operand_precision"] = {
                    "W": weight_precision,
                    "I": act_precision,
                    "O_final": act_precision,
                    "O": intermediate_output_precision,
                }
            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                data["operand_source"] = {"W": predecessors[0], "I": predecessors[1]}
                data["operand_precision"] = {
                    "W": act_precision,
                    "I": act_precision,
                    "O_final": act_precision,
                    "O": intermediate_output_precision,
                }

            case _:
                raise ValueError("No more than 2 layer predecessors expected")

        return data

    def generate_node(self):
        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        node_data = self.get_layer_node_input_format(ia_dimension_shape, oa_dimension_shape)
        node_factory = LayerNodeFactory(node_data, self.mapping_data)
        node_attrs = node_factory.create_node_attr()

        # Override spatial mapping by the one defined in the core's dataflows
        core_allocation = node_attrs.core_allocation
        spatial_mapping = self.accelerator.get_spatial_mapping_from_core(core_allocation)
        node_attrs.spatial_mapping = spatial_mapping

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
