from typing import Any

from zigzag.parser.workload_factory import LayerNodeFactory

from stream.onnx_utils import get_onnx_input_shapes, get_onnx_output_shapes
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import InterCoreMappingAttributes


class AsymmetricSimdParser(OnnxComputeOperatorParser):
    """Similar to SIMD parser, but for nodes with asymmetric input shapes such as (B,D,K)*(D,K)
    e.g. Add, Mul, etc.
    """

    DEFAULT_LAYER_DIMENSIONS = ["B", "D", "K"]

    def get_layer_node_user_format(
        self,
        input_shape: list[int],
        output_shape: list[int],
        mapping: InterCoreMappingAttributes | None = None,
    ):
        """
        Generate the necessary dictionary items required for the LayerNode creation.

        Args:
            input_shape: the non-batched input shape
            output_shape: the batched output shape, equal to the batched input shape
        """

        raise DeprecationWarning("use MulParser instead")

        if not (len(output_shape) == 3 and len(input_shape) == 2):
            raise NotImplementedError

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = output_shape

        if len(output_shape) > len(AsymmetricSimdParser.DEFAULT_LAYER_DIMENSIONS):
            raise NotImplementedError

        data["equation"] = "O[b][d][k]+=I[b][d][k]*W[d][k]"
        data["loop_dims"] = ["B", "D", "k"]

        return data

    def generate_node(self):
        # Get the input and output activation shapes
        input_shapes = get_onnx_input_shapes(self.node, self.onnx_model)
        if len(input_shapes) != 2:
            raise NotImplementedError("Only SIMD nodes with input length 2 are supported")
        input_shape1, input_shape2 = input_shapes

        output_shapes = get_onnx_output_shapes(self.node, self.onnx_model)
        if len(output_shapes) != 1:
            raise NotImplementedError("Only SIMD nodes with input length 2 are supported")
        output_shape = output_shapes.pop()

        if input_shape1 == output_shape:
            non_batched_input_shape = input_shape2
        elif input_shape2 == output_shape:
            non_batched_input_shape = input_shape1
        else:
            raise ValueError(
                "At least one of the two input shapes should equal the output shape in an asymmetric SIMD node"
            )

        node_data = self.get_layer_node_user_format(non_batched_input_shape, output_shape)
        node_factory = LayerNodeFactory(node_data, mapping_data=None)
        node_attrs = node_factory.create_node_attr()
        mapping = self.get_mapping_this_node()
        input_names = list(self.node.input)

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            mapping_attr=mapping,
            op_type=self.node.op_type,
            input_names=input_names,
        )
