from typing import Any

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.mapping import InterCoreMappingAttributes


class SimdParser(OnnxComputeOperatorParser):
    """Parses an ONNX operator representing an elementwise operation (simd) into a ComputationNode.
    e.g. Add, etc.
    # TODO this functionality is exactly the same as Mul but without support for broadcast (asymmetric) shapes
    """

    DEFAULT_LAYER_DIMENSIONS = ["B", "H", "D", "K"]

    def get_layer_node_user_format(
        self, input_shape: list[int], output_shape: list[int], mapping: InterCoreMappingAttributes
    ):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        assert input_shape == output_shape, "Input and output of simd operation should be identical."
        predecessors = self.get_node_predecessors()
        # Nodes with only 1 input (e.g. Relu, Max, add/mul with constant, etc) have an empty `W` part in equation
        has_single_input = len(predecessors) == 1

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = output_shape

        if len(output_shape) > len(SimdParser.DEFAULT_LAYER_DIMENSIONS):
            raise NotImplementedError

        possible_loop_dims = (
            mapping.layer_dimension_names
            if len(mapping.layer_dimension_names) == len(output_shape)
            else SimdParser.DEFAULT_LAYER_DIMENSIONS
        )

        loop_dims = possible_loop_dims[0 : len(output_shape)]
        equation_dims = "".join([f"[{dim.lower()}]" for dim in loop_dims])
        equation_dims_W = "[]" if has_single_input else equation_dims
        equation = f"O{equation_dims}+=I{equation_dims}*W{equation_dims_W}"

        data["equation"] = equation
        data["loop_dims"] = loop_dims

        return data
