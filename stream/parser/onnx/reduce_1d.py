from typing import Any

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser


class Reduce1DParser(OnnxComputeOperatorParser):
    """Parses an operator that reduces the data in a single dimension.
    e.g. sum over one row or max of a single row
    """

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        # TODO check the output shape as well?
        assert len(self.get_node_predecessors()) == 1

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = input_shape

        match len(input_shape):
            case 2:
                data["equation"] = "O[k]+=I[k][c]*W[]"
                data["loop_dims"] = ["K", "C"]
            case 3:
                data["equation"] = "O[b][k]+=I[b][k][c]*W[]"
                data["loop_dims"] = ["B", "K", "C"]
            case 4:
                data["equation"] = "O[b][h][k]+=I[b][h][k][c]*W[]"
                data["loop_dims"] = ["B", "H", "K", "C"]
            case _:
                raise NotImplementedError

        return data
