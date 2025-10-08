import logging
from collections.abc import Generator
from typing import Any

from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.utils import (
    get_attribute_ints_with_name,
    get_onnx_tensor_type,
)
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import InterCoreMappingAttributes

logger = logging.getLogger(__name__)


class FusedMatMulParser(OnnxComputeOperatorParser):
    """Parses an ONNX Gemm operator into a ComputationNode"""

    def run(self) -> Generator[ComputationNode, None, None]:  # type: ignore
        yield self.generate_node()

    def generate_node(self):
        # Get the input and output activation shapes
        first_input_shape, second_input_shape, output_shape = get_fused_matmul_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

        # logger.info("%s node name %s input shape %s output shape", self.node.name, input_shape, output_shape)
        # From the ONNX node
        mapping = self.get_mapping_this_node()
        node_data = self.get_layer_node_user_format(
            first_input_shape,
            second_input_shape,
            output_shape,
            mapping,
        )
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
        self,
        first_input_shape: list[int],
        second_input_shape: list[int],
        output_shape: list[int],
        mapping: InterCoreMappingAttributes,
    ) -> dict[str, Any]:
        """! Generate layer data in user input format for FusedMatMul or GEMM ONNX nodes.
        I[B][H][D][C] * W([B][H])[C][K] -> O[B][H][D][K]
        """
        transpose_first_input = get_attribute_ints_with_name("transA", self.node.attribute, default=0)
        transpose_second_input = get_attribute_ints_with_name("transB", self.node.attribute, default=0)
        transpose_first_batch = get_attribute_ints_with_name("transBatchA", self.node.attribute, default=0)
        transpose_second_batch = get_attribute_ints_with_name("transBatchB", self.node.attribute, default=0)

        # Transpose batch dimensions if required
        if transpose_first_batch or transpose_second_batch:
            MINIMAL_BATCH_SIZE_TO_TRANSPOSE = 2
            assert len(first_input_shape) > MINIMAL_BATCH_SIZE_TO_TRANSPOSE and len(first_input_shape) == len(
                second_input_shape
            )

            if transpose_first_batch:
                # Reverse batch dimensions (excluding the last two dimensions)
                batch_dims = first_input_shape[:-2]
                first_input_shape = first_input_shape[-2:] + batch_dims[::-1]

            if transpose_second_batch:
                # Reverse batch dimensions (excluding the last two dimensions)
                batch_dims = second_input_shape[:-2]
                second_input_shape = second_input_shape[-2:] + batch_dims[::-1]

        # Transpose the last two dimensions if required
        if transpose_first_input:
            first_input_shape = first_input_shape[:-2] + [first_input_shape[-1], first_input_shape[-2]]

        if transpose_second_input:
            second_input_shape = second_input_shape[:-2] + [second_input_shape[-1], second_input_shape[-2]]

        # Validate shapes
        assert first_input_shape[-2] == output_shape[-2], "First dimension of input and output matrix must be the same"
        assert first_input_shape[-1] == second_input_shape[-2], "Inner dimensions of input matrices must match"

        predecessors = self.get_node_predecessors()
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
            else FusedMatMulParser.DEFAULT_LAYER_DIMENSIONS
        )
        nb_batch_dims = len(first_input_shape) - 2
        batch_dims = possible_loop_dims[:nb_batch_dims]
        non_batch_dims = possible_loop_dims[-3:]

        data["loop_dims"] = batch_dims + non_batch_dims

        # Loop sizes
        output_rows_dim, inner_dim, output_cols_dim = non_batch_dims
        output_rows_size = output_shape[-2]
        inner_size = first_input_shape[-1]
        output_cols_size = output_shape[-1]

        batch_sizes = first_input_shape[:nb_batch_dims]
        data["loop_sizes"] = batch_sizes + [output_rows_size, inner_size, output_cols_size]

        # Construct equation
        batch_dims_W = [] if weights_are_constant else batch_dims
        equation_dims_I = "".join([f"[{dim.lower()}]" for dim in batch_dims + [output_rows_dim, inner_dim]])
        equation_dims_W = "".join([f"[{dim.lower()}]" for dim in batch_dims_W + [inner_dim, output_cols_dim]])
        equation_dims_O = "".join([f"[{dim.lower()}]" for dim in batch_dims + [output_rows_dim, output_cols_dim]])
        equation = f"O{equation_dims_O}+=I{equation_dims_I}*W{equation_dims_W}"

        data["equation"] = equation
        return data


def get_fused_matmul_node_input_output_dimension_shapes(node: NodeProto, model: ModelProto):
    # assumed it is the first input, don't see a way to otherwise know
    first_input_name = node.input[0]
    first_input_shape = get_onnx_tensor_type(first_input_name, model).shape

    second_input_name = node.input[0]
    second_input_shape = get_onnx_tensor_type(second_input_name, model).shape

    output_name = node.output[0]
    output_shape = get_onnx_tensor_type(output_name, model).shape

    return first_input_shape, second_input_shape, output_shape
