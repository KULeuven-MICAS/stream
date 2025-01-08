import logging
from typing import Generator

from zigzag.parser.onnx.gemm_parser import GemmParser as GemmParserZigZag
from zigzag.parser.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.computation.computation_node import ComputationNode

logger = logging.getLogger(__name__)


class GemmParser(GemmParserZigZag, OnnxComputeOperatorParser):
    """Parses an ONNX Gemm operator into a ComputationNode"""

    def run(self) -> Generator[ComputationNode, None, None]:  # type: ignore
        yield self.generate_node()

    def generate_node(self):
        # Get the input and output activation shapes
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # From the ONNX node
        mapping = self.get_mapping_this_node()
        node_data = self.get_layer_node_user_format(input_shape, output_shape)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        input_names = list(self.node.input)

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type=self.node.op_type,
            node_attr=node_attrs,
            mapping_attr=mapping,
            input_names=input_names,
        )
