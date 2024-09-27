from typing import Any

from zigzag.parser.onnx.utils import (
    get_node_input_output_dimension_shapes,
)
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.utils import get_onnx_input_shapes
from stream.workload.computation_node import ComputationNode


class AsymmetricSimdParser(OnnxComputeOperatorParser):
    """Similar to SIMD parser, but for nodes with asymmetric input shapes such as (B,D,K)*(D,K)
    e.g. Add, Mul, etc.
    """

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        Args:
            input_shape: the non-batched input shape
            output_shape: the batched output shape, equal to the batched input shape
        """
        if not (len(output_shape) == 3 and len(input_shape) == 2):
            raise NotImplementedError

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_input_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = output_shape

        data["equation"] = "O[b][d][k]+=I[b][d][k]*W[d][k]"
        data["loop_dims"] = ["B", "D", "k"]

        return data

    def generate_node(self):
        # Get the input and output activation shapes
        input_shape1, input_shape2 = get_onnx_input_shapes(self.node, self.onnx_model)
        _, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        if input_shape1 == output_shape:
            non_batched_input_shape = input_shape2
        elif input_shape2 == output_shape:
            non_batched_input_shape = input_shape1
        else:
            raise ValueError(
                "At least one of the two input shapes should equal the output shape in an asymmetric SIMD node"
            )

        node_data = self.get_layer_node_user_format(non_batched_input_shape, output_shape)
        node_factory = LayerNodeFactory(node_data, self.mapping_data)
        node_attrs = node_factory.create_node_attr()

        # Override spatial mapping by the one defined in the core's dataflows
        core_allocation = node_attrs.core_allocation
        spatial_mapping = self.accelerator.get_spatial_mapping_from_core(core_allocation)
        node_attrs.spatial_mapping = spatial_mapping

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            op_type=self.node.op_type,
        )
