from typing import Any

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser


class SimdParser(OnnxComputeOperatorParser):
    """Parses an ONNX operator representing an elementwise operation (simd) into a ComputationNode.
    e.g. Add, etc.
    """

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]):
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

        match len(output_shape):
            case 1:
                data["equation"] = f"O[k]+=I[k]*W{'[]' if has_single_input else '[k]'}"
                data["loop_dims"] = ["K"]
            case 2:
                data["equation"] = f"O[d][k]+=I[d][k]*W{'[]' if has_single_input else '[d][k]'}"
                data["loop_dims"] = ["D", "K"]
            case 3:
                data["equation"] = f"O[b][d][k]+=I[b][d][k]*W{'[]' if has_single_input else '[b][d][k]'}"
                data["loop_dims"] = ["B", "D", "K"]
            case 4:
                data["equation"] = f"O[b][h][d][k]+=I[b][h][d][k]*W{'[]' if has_single_input else '[b][h][d][k]'}"
                data["loop_dims"] = ["B", "H", "D", "K"]
            case _:
                raise NotImplementedError

        return data
