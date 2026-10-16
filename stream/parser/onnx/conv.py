from xdsl.ir.affine import AffineDimExpr, AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.utils import window_index
from stream.workload.workload import ComputationNode, Tensor


def conv_maps(
    activation: Tensor, weight: Tensor, biased: bool, window: tuple[list[int], ...], groups: int
) -> tuple[AffineMap, ...]:
    """The access maps of a 2D convolution over (b, ox, oy, fx, fy, c, k), and the group g when grouped, given its
    ``window`` strides, dilations and leading padding: its input, its weight, its bias where ``biased``, its output."""
    num_dims = 7 if groups == 1 else 8
    b, ox, oy, fx, fy, c, k, *g = (AffineDimExpr(i) for i in range(num_dims))
    in_channel = g[0] * weight.shape[1] + c if g else c
    out_channel = g[0] * (weight.shape[0] // groups) + k if g else k
    results = [(b, in_channel, *window_index((oy, ox), (fy, fx), *window)), (out_channel, c, fy, fx)]
    results += [(out_channel,)] * biased + [(b, out_channel, oy, ox)]
    return tuple(AffineMap(num_dims, 0, tuple(r)) for r in results)


class ConvParser(OnnxOperatorParser):
    """Parses an ONNX Conv into a ComputationNode over (b, ox, oy, fx, fy, c, k), and the group g when grouped;
    the optional bias is a third input, read per output channel."""

    EXPECTED_NB_OF_INPUTS = 2  # activation and weight are required, bias is optional

    def get_mappings_1d_conv(self, inputs: tuple[Tensor, ...]) -> tuple[AffineMap, ...]:
        raise NotImplementedError("1D convolution is not supported yet.")

    def get_mappings_2d_conv(self, inputs: tuple[Tensor, ...]) -> tuple[AffineMap, ...]:
        activation, weight = inputs[:2]
        window = self.get_window(activation.shape[2:], weight.shape[2:])
        biased = len(inputs) > self.EXPECTED_NB_OF_INPUTS
        return conv_maps(activation, weight, biased, window, self.get_node_attribute_int("group") or 1)

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input if inp)
        assert len(inputs) >= self.EXPECTED_NB_OF_INPUTS, "Conv must have at least activation and weight inputs."
        input_dimensionality = len(inputs[0].shape)
        assert len(inputs[1].shape) == input_dimensionality, "Activation and weight must have the same rank."
        outputs = self.get_output_tensors()
        assert len(outputs) == 1, "Conv operator must have exactly 1 output."
        assert len(outputs[0].shape) == input_dimensionality, (
            "Output tensor dimensionality must match input tensor dimensionality."
        )
        match input_dimensionality:
            case 3:
                mappings = self.get_mappings_1d_conv(inputs)
            case 4:
                mappings = self.get_mappings_2d_conv(inputs)
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
