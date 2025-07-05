from typing import Any

from stream.onnx_utils import get_onnx_input_shapes, get_onnx_output_shapes
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.mapping import InterCoreMappingAttributes


class MulParser(OnnxComputeOperatorParser):
    """Parses an ONNX operator representing an elementwise operation (Mul) into a ComputationNode."""

    DEFAULT_LAYER_DIMENSIONS = ["B", "H", "D", "K"]

    def get_common_and_broadcast_shape(self):
        """This node assumes that the ONNX node has 2 inputs and 1 output. One input shape is identical to the output
        shape, and the other shape can broadcast in dimensions.
        Returns the common shape (in and out) and the broadcast shape"""
        input_shapes = get_onnx_input_shapes(self.node, self.onnx_model)
        output_shapes = get_onnx_output_shapes(self.node, self.onnx_model)

        EXPECTED_INPUTS = 2
        EXPECTED_OUTPUTS = 1
        if len(input_shapes) != EXPECTED_INPUTS or len(output_shapes) != EXPECTED_OUTPUTS:
            raise NotImplementedError(
                f"Expected {EXPECTED_INPUTS} inputs and {EXPECTED_OUTPUTS} output, "
                f"got {len(input_shapes)} inputs and {len(output_shapes)} outputs"
            )

        output_shape = output_shapes.pop()
        if not any(shape == output_shape for shape in input_shapes):
            raise NotImplementedError(f"Expected input shape {output_shape} to be one of {input_shapes}")

        input_shape = output_shape
        input_shapes.remove(output_shape)
        broadcast_shape = input_shapes.pop()

        # e.g. (3,5) * (8,3,5) is ok (broadcast over dim 0), but (3,2) * (8,3,5) is unclear
        for broadcast_size, in_size in zip(reversed(broadcast_shape), reversed(input_shape), strict=False):
            if broadcast_size not in (1, in_size):
                raise ValueError(f"Cannot broadcast {broadcast_shape} to {input_shape}")

        return input_shape, broadcast_shape

    def get_operand_source_input_format(
        self,
        shape_of_w: tuple[int, ...],
    ):
        """This method needs more care in this subclass, since the equation assumes that the input with 'broadcast'
        shape is always at `W`"""
        predecessors = self.get_node_predecessors()
        match len(predecessors):
            case 0:
                # e.g. first node of graph
                return {"W": self.node_id, "I": self.node_id}
            case 1:
                # One source operand, one constant
                return {"W": self.node_id, "I": predecessors[0]}
            case 2:
                # Two source operands, none are constant
                # Name of the input that corresponds to the W shape
                broadcast_intput = self.node.input[get_onnx_input_shapes(self.node, self.onnx_model).index(shape_of_w)]
                try:
                    node_id_W = next(
                        node_id
                        for node_id, outputs in self.nodes_outputs.items()
                        if broadcast_intput in outputs and node_id in predecessors
                    )
                    node_id_I = (
                        node_id_W
                        if predecessors[0] == predecessors[1]
                        else next(i for i in predecessors if i != node_id_W)
                    )
                    return {"W": node_id_W, "I": node_id_I}
                except StopIteration as exc:
                    raise ValueError(f"Cannot find correct inputs of {self.node.name}") from exc
            case _:
                raise ValueError("No more than 2 layer predecessors expected")

    def get_layer_node_user_format(
        self,
        input_shape: list[int],
        output_shape: list[int],
        mapping: InterCoreMappingAttributes,
    ):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        # TODO use layer dimension names from mapping
        """
        common_shape, broadcast_shape = self.get_common_and_broadcast_shape()

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format(shape_of_w=broadcast_shape)
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = common_shape

        if len(common_shape) > len(MulParser.DEFAULT_LAYER_DIMENSIONS):
            raise NotImplementedError

        possible_loop_dims = (
            mapping.layer_dimension_names
            if len(mapping.layer_dimension_names) == len(common_shape)
            else MulParser.DEFAULT_LAYER_DIMENSIONS
        )

        loop_dims = possible_loop_dims[0 : len(common_shape)]
        loop_dims_broadcast = reversed(
            [dim for dim, _ in zip(reversed(loop_dims), reversed(broadcast_shape), strict=False)]
        )

        equation_dims_common = "".join([f"[{dim.lower()}]" for dim in loop_dims])
        equation_dims_broadcast = "".join([f"[{dim.lower()}]" for dim in loop_dims_broadcast])
        equation = f"O{equation_dims_common}+=I{equation_dims_common}*W{equation_dims_broadcast}"

        data["loop_dims"] = loop_dims
        data["equation"] = equation

        return data
