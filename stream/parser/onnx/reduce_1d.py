from typing import Any

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.mapping import InterCoreMappingAttributes


class Reduce1DParser(OnnxComputeOperatorParser):
    """Parses an operator that reduces the data in a single dimension.
    e.g. sum over one row or max of a single row
    """

    DEFAULT_LAYER_DIMENSIONS = ["B", "H", "D", "K"]

    def get_reduction_dim(self, input_shape: list[int], output_shape: list[int]):
        """Returns the axis in which the dimension is reduced"""

        # The case that keepdim=True: the reduced dimension is kept with size 1
        if len(input_shape) == len(output_shape):
            different_size = [a != b for a, b in zip(input_shape, output_shape, strict=False)]
            match sum(different_size):
                case 0:
                    # Input and output size are the same: can happen with when a Reduce1D node is inferred but
                    # not present in ONNX -> default to -1
                    reduction_dim = len(input_shape) - 1
                case 1:
                    reduction_dim = different_size.index(True)
                    if output_shape[reduction_dim] != 1:
                        raise ValueError(
                            f"The reduced dimension at axis {reduction_dim} in {output_shape} is larger than 1"
                        )
                case _:
                    # More than 1 dimension has different size
                    raise ValueError(
                        f"Input and output shapes {input_shape}, {output_shape} should only differ in one dim"
                    )
            return reduction_dim

        # Other: assume that the reduction is at axis=-1
        if not all(a == b for a, b in zip(input_shape, output_shape, strict=False)):
            raise NotImplementedError("Reduce node with reduction axis other than -1 not implemented yet.")
        reduction_dim = len(input_shape) - 1  # Last dimension

    def get_layer_node_user_format(
        self,
        input_shape: list[int],
        output_shape: list[int],
        mapping: InterCoreMappingAttributes | None = None,
    ):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        if len(self.get_node_predecessors()) != 1:
            raise NotImplementedError

        if self.get_reduction_dim(input_shape, output_shape) != len(input_shape) - 1:
            raise NotImplementedError("Only reduction in axis=-1 is supported")

        # This is a ONNX node property but can be inferred from the shapes
        keep_dim = len(input_shape) == len(output_shape)

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = input_shape

        if len(input_shape) > len(Reduce1DParser.DEFAULT_LAYER_DIMENSIONS):
            raise NotImplementedError

        assert mapping is not None, "Mapping must be provided for Reduce1DParser"
        possible_loop_dims = (
            mapping.layer_dimension_names
            if len(mapping.layer_dimension_names) == len(output_shape)
            else Reduce1DParser.DEFAULT_LAYER_DIMENSIONS
        )
        loop_dims = possible_loop_dims[0 : len(output_shape)]

        # If keep_dim: add an arbitrary dim of size 1
        reduced_dim_output = "R"  # C reduced to 1
        assert not keep_dim or reduced_dim_output not in possible_loop_dims, (
            "Layer dimension `R` is reserved for the reduction axis"
        )

        # Output: drop the last dimension: this dimension is reduced
        loop_dims_O = loop_dims[0:-1]
        loop_dims_I = loop_dims.copy()
        if keep_dim:
            # Replace reduction dim with size-1 dimension
            loop_dims_O.append(reduced_dim_output)
            loop_dims.append(reduced_dim_output)
            data["loop_sizes"].append(1)

        equation_dims_I = "".join([f"[{dim.lower()}]" for dim in loop_dims_I])
        equation_dims_O = "".join([f"[{dim.lower()}]" for dim in loop_dims_O])
        equation = f"O{equation_dims_O}+=I{equation_dims_I}*W[]"

        data["equation"] = equation
        data["loop_dims"] = loop_dims

        return data
