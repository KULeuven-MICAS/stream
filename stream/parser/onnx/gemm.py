import logging
from typing import Generator

from zigzag.parser.onnx.gemm_parser import GemmParser as GemmParserZigZag
from zigzag.parser.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes,
)

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
        transpose_first_input = get_attribute_ints_with_name("transA", self.node.attribute, default=0)
        if transpose_first_input:
            assert len(input_shape) == 2, "Transpose only supported for GEMMs with two input dimensions"
            input_shape = [input_shape[1], input_shape[0]]

        logger.info("%s node name %s input shape %s output shape", self.node.name, input_shape, output_shape)
        # From the ONNX node
        node_data = self.get_layer_node_user_format(input_shape, output_shape)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()

        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type=self.node.op_type,
            node_attr=node_attrs,
            mapping_attr=mapping,
        )
