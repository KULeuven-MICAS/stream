from typing import Any

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser


class Reduce1DParser(OnnxComputeOperatorParser):
    """Parses an operator that reduces the data in a single dimension.
    e.g. sum over one row or max of a single row
    """

    def get_reduction_dim(self, input_shape: list[int], output_shape: list[int]):
        """Returns the axis in which the dimension is reduced"""

        # The case that keepdim=True: the reduced dimension is kept with size 1
        if len(input_shape) == len(output_shape):
            different_size = [a != b for a, b in zip(input_shape, output_shape)]
            if sum(different_size) != 1:
                raise ValueError(f"Input and output shapes {input_shape}, {output_shape} should only differ in one dim")
            reduction_dim = different_size.index(True)
            if output_shape[reduction_dim] != 1:
                raise ValueError(f"The reduced dimension at axis {reduction_dim} in {output_shape} is larger than 1")
            return reduction_dim

        # Other: assume that the reduction is at axis=-1
        if not all(a == b for a, b in zip(input_shape, output_shape)):
            raise NotImplementedError("Reduce node with reduction axis other than -1 not implemented yet.")
        reduction_dim = len(input_shape) - 1  # Last dimension

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]):
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

        # C is always the reduction dim
        # If keep_dim: add an arbitrary dim of size 1
        reduced_dim_output = "CR"  # C reduced to 1
        eq_part_CR = f"[{reduced_dim_output}]" if keep_dim else ""
        match len(input_shape):
            case 2:
                data["equation"] = f"O[k]{eq_part_CR}+=I[k][c]*W[]"
                data["loop_dims"] = ["K", "C"]
            case 3:
                data["equation"] = f"O[b][k]{eq_part_CR}+=I[b][k][c]*W[]"
                data["loop_dims"] = ["B", "K", "C"]
            case 4:
                data["equation"] = f"O[b][h][k]{eq_part_CR}+=I[b][h][k][c]*W[]"
                data["loop_dims"] = ["B", "H", "K", "C"]
            case _:
                raise NotImplementedError

        if keep_dim:
            data["loop_dims"] += [reduced_dim_output]
            data["loop_sizes"] += [1]

        return data
