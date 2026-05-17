from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class ConvParser(OnnxOperatorParser):
    """Parses an ONNX Conv operator into a ComputationNode"""

    EXPECTED_NB_OF_INPUTS = 2  # activation and weight are required, bias is optional (D-01)

    def get_mappings_1d_conv(self) -> tuple[AffineMap, AffineMap, AffineMap]:
        raise NotImplementedError("1D convolution is not supported yet.")

    def get_mappings_2d_conv(self) -> tuple[AffineMap, AffineMap, AffineMap]:
        strides = self.get_node_attribute_ints("strides")
        required_strides_length = 2
        if not strides:
            strides = [1, 1]
        assert len(strides) == required_strides_length, "Strides should be 2D for 2D convolution."
        sx, sy = strides
        pads = self.get_node_attribute_ints("pads")
        if not pads:
            pads = [0, 0, 0, 0]
        if not all(p == pads[0] for p in pads):
            raise NotImplementedError("Asymmetric padding is not supported yet.")
        p = pads[0]

        return (
            AffineMap.from_callable(lambda b, ox, oy, fx, fy, c, k: (b, c, sy * oy + fy - p, sx * ox + fx - p)),
            AffineMap.from_callable(lambda b, ox, oy, fx, fy, c, k: (k, c, fy, fx)),
            AffineMap.from_callable(lambda b, ox, oy, fx, fy, c, k: (b, k, oy, ox)),
        )

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        # Inputs
        all_inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        assert len(all_inputs) >= self.EXPECTED_NB_OF_INPUTS, "Conv must have at least activation and weight inputs."
        inputs = all_inputs[: self.EXPECTED_NB_OF_INPUTS]  # drop optional bias silently (D-01)
        input_dimensionalities = tuple(len(tensor.shape) for tensor in inputs)
        assert all(dim == input_dimensionalities[0] for dim in input_dimensionalities), (
            "All input tensors must have the same dimensionality."
        )
        input_dimensionality = input_dimensionalities[0]
        # Outputs
        outputs = self.get_output_tensors()
        assert len(outputs) == 1, "Conv operator must have exactly 1 output."
        assert len(outputs[0].shape) == input_dimensionality, (
            "Output tensor dimensionality must match input tensor dimensionality."
        )
        match input_dimensionality:
            case 3:
                mappings = self.get_mappings_1d_conv()
            case 4:
                mappings = self.get_mappings_2d_conv()
            case _:
                raise NotImplementedError(
                    f"Conv operator with input dimensionality {input_dimensionality} is not supported yet."
                )

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=outputs,
            operand_mapping=mappings,
        )
