import logging
from typing import Any

from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.GemmParser import GemmParser as GemmParserZigZag

from stream.hardware.architecture.accelerator import Accelerator
from stream.classes.io.onnx.operator_parser import OnnxOperatorParser
from stream.workload.computation_node import ComputationNode

logger = logging.getLogger(__name__)


class GemmParser(GemmParserZigZag, OnnxOperatorParser):
    """Parses an ONNX Gemm operator into a ComputationNode"""

    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
        *,
        mapping_data: list[dict[str, Any]],
        accelerator: Accelerator,
    ) -> None:
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.onnx_model = onnx_model
        self.mapping_data = mapping_data
        self.accelerator = accelerator

    def run(self):
        """Run the parser"""
        return self.generate_node()

    def generate_node(self):
        layer_node = self.generate_layer_node()
        node_attrs = layer_node.extract_node_attr()

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
            op_type=node_attrs.layer_type,
            operand_tensor_reshape=None,
        )
