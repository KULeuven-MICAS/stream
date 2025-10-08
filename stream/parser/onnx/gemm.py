import logging
from collections.abc import Generator
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


class GemmParser(OnnxComputeOperatorParser):
    """Parses an ONNX Gemm operator into a ComputationNode"""

    def run(self) -> Generator[ComputationNode, None, None]:  # type: ignore
        yield self.generate_node()

    def generate_node(self):
        # Get the input and output activation shapes
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)
        transpose_first_input = get_attribute_ints_with_name("transA", self.node.attribute, default=0)
        if transpose_first_input:
            NUM_INPUTS_EXPECTED = 2
            assert len(input_shape) == NUM_INPUTS_EXPECTED, (
                "Transpose only supported for GEMMs with two input dimensions"
            )
            input_shape = [input_shape[1], input_shape[0]]

        logger.info("%s node name %s input shape %s output shape", self.node.name, input_shape, output_shape)
        # From the ONNX node
        mapping = self.get_mapping_this_node()
        node_data = self.get_layer_node_user_format(input_shape, output_shape, mapping)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type=self.node.op_type,
            node_attr=node_attrs,
            mapping_attr=mapping,
        )

    DEFAULT_LAYER_DIMENSIONS = ["B", "H", "D", "C", "K"]

    def get_layer_node_user_format(
        self, input_shape: list[int], output_shape: list[int], mapping: InterCoreMappingAttributes
    ) -> dict[str, Any]:
        """! Generate layer data in user input format for MatMul or GEMM ONNX nodes.
        I[B][H][D][C] * W([B][H])[C][K]-> O [B][H][D][K]

        """
        assert input_shape[-2] == output_shape[-2], "First dimension of input and output matrix must be the same"

        predecessors = self.get_node_predecessors()

        # If there are 2 input nodes, `weights` represents a variable input
        NUMBER_OF_PREDECESSORS_FOR_CONSTANT_WEIGHTS = 1
        weights_are_constant = len(predecessors) <= NUMBER_OF_PREDECESSORS_FOR_CONSTANT_WEIGHTS

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["dimension_relations"] = []
        data["operand_source"] = self.get_operand_source_user_format(predecessors)
        data["operand_precision"] = self.get_operand_precision_user_format()

        # Loop dims
        possible_loop_dims = (
            mapping.layer_dimension_names
            if len(mapping.layer_dimension_names) >= len(output_shape) + 1
            else GemmParser.DEFAULT_LAYER_DIMENSIONS
        )
        nb_batch_dims = len(input_shape) - 2
        batch_dims = possible_loop_dims[:nb_batch_dims]
        non_batch_dims = possible_loop_dims[-3:]
        data["loop_dims"] = batch_dims + non_batch_dims

        # Loop sizes
        output_rows_dim, inner_dim, output_cols_dim = non_batch_dims
        output_rows_size = output_shape[-2]
        inner_size = input_shape[-1]
        output_cols_size = output_shape[-1]
        assert input_shape[-2] == output_rows_size
        batch_sizes = input_shape[:nb_batch_dims]
        data["loop_sizes"] = batch_sizes + [output_rows_size, inner_size, output_cols_size]

        # Construct equation
        batch_dims_W = [] if weights_are_constant else batch_dims
        equation_dims_I = "".join([f"[{dim.lower()}]" for dim in batch_dims + [output_rows_dim, inner_dim]])
        equation_dims_W = "".join([f"[{dim.lower()}]" for dim in batch_dims_W + [inner_dim, output_cols_dim]])
        equation_dims_O = "".join([f"[{dim.lower()}]" for dim in batch_dims + [output_rows_dim, output_cols_dim]])
        equation = f"O{equation_dims_O}+=I{equation_dims_I}*W{equation_dims_W}"
        data["equation"] = equation

        return data
