from typing import Any
import logging
from onnx import ModelProto, NodeProto
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.workload.computation_node import ComputationNode
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import (
    get_node_input_output_dimension_shapes,
)
from zigzag.parser.workload_factory import LayerNodeFactory

logger = logging.getLogger(__name__)


class MatMulParser(ONNXOperatorParser):
    """Parses an ONNX MatMul operator into a LayerNode"""

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
        self.node_outputs = nodes_outputs
        self.onnx_model = onnx_model
        self.mapping_data = mapping_data
        self.accelerator = accelerator
        self.op_type = "matmul"

    def run(self):
        """Run the parser"""
        return self.generate_layer_node_for_matmul()

    def get_layer_node_input_format(self, B: int, C: int, K: int):
        """
        Generate the necessary dictionary items required for the Node creation.
        """
        # convert the data types to precisions based on the onnx definition

        # Equation
        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = f"Layer{self.node_id}"
        data["operator_type"] = self.op_type
        data["equation"] = "O[b][k]+=I[b][c]*W[c][k]"

        data["loop_dims"] = ["K", "C", "B"]
        data["loop_sizes"] = [K, C, B]

        data["dimension_relations"] = []
        data["operand_precision"] = {"O": 16, "O_final": 8, "I": 8, "W": 8}

        # Find the previous layer(s) that should be this node's parent(s)
        node_inputs = self.node.input
        assert len(node_inputs) >= 2, f"MatMul should have at least two input names, but has: {node_inputs}."
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

    def generate_layer_node_for_matmul(self):
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # First element is batch size, second is input/output channel
        assert len(ia_dimension_shape) == len(oa_dimension_shape) == 2
        assert ia_dimension_shape[0] == oa_dimension_shape[0]  # Batch size should be the same for input and output
        # If the batch size is 0, we discard it by setting it to 1 internally inside ZigZag
        batch_size = ia_dimension_shape[0]
        if batch_size == 0:
            B = 1
        else:
            B = batch_size
        C = ia_dimension_shape[1]
        K = oa_dimension_shape[1]

        node_data = self.get_layer_node_input_format(B, C, K)

        node_factory = LayerNodeFactory(node_data, self.mapping_data)
        node_attrs = node_factory.create_node_attr()

        # Override spatial mapping by the one defined in the core's dataflows
        core_allocation = node_attrs.core_allocation
        spatial_mapping = self.accelerator.get_spatial_mapping_from_core(core_allocation)
        node_attrs.spatial_mapping = spatial_mapping

        # Get the node's input(s) and output(s) tensor names
        node_input_names = list(self.node.input)
        node_output_names = list(self.node.output)
        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            input_names=node_input_names,
            output_names=node_output_names,
            op_type=self.op_type,
            operand_tensor_reshape=None,
        )
