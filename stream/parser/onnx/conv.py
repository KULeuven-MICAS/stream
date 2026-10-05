from math import ceil

from onnx import helper
from xdsl.ir.affine import AffineDimExpr, AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class ConvParser(OnnxOperatorParser):
    """Parses an ONNX Conv into a ComputationNode over (b, ox, oy, fx, fy, c, k), and the group g when grouped;
    the optional bias is a third input, read per output channel."""

    EXPECTED_NB_OF_INPUTS = 2  # activation and weight are required, bias is optional

    def _per_axis(self, name: str, default: int) -> list[int]:
        return self.get_node_attribute_ints(name) or [default, default]

    def _leading_pads(self, sizes: tuple[int, ...], kernel: tuple[int, ...], strides, dilations) -> list[int]:
        """The padding before rows and columns: the ``pads`` attribute, or what ``auto_pad`` derives (ONNX Conv)."""
        auto_pad = next((helper.get_attribute_value(a) for a in self.node.attribute if a.name == "auto_pad"), b"")
        if auto_pad in (b"", b"NOTSET"):
            return (self.get_node_attribute_ints("pads") or [0, 0])[:2]
        if auto_pad == b"VALID":
            return [0, 0]
        totals = [
            max(0, (ceil(n / s) - 1) * s + (f - 1) * d + 1 - n)
            for n, f, s, d in zip(sizes, kernel, strides, dilations, strict=True)
        ]
        return [t // 2 if auto_pad == b"SAME_UPPER" else t - t // 2 for t in totals]

    def get_mappings_1d_conv(self, inputs: tuple[Tensor, ...]) -> tuple[AffineMap, ...]:
        raise NotImplementedError("1D convolution is not supported yet.")

    def get_mappings_2d_conv(self, inputs: tuple[Tensor, ...]) -> tuple[AffineMap, ...]:
        activation, weight = inputs[:2]
        (sy, sx), (dy, dx) = self._per_axis("strides", 1), self._per_axis("dilations", 1)
        top, left = self._leading_pads(activation.shape[2:], weight.shape[2:], (sy, sx), (dy, dx))
        groups = self.get_node_attribute_int("group") or 1
        num_dims = 7 if groups == 1 else 8
        b, ox, oy, fx, fy, c, k, *g = (AffineDimExpr(i) for i in range(num_dims))
        in_channel = g[0] * weight.shape[1] + c if g else c
        out_channel = g[0] * (weight.shape[0] // groups) + k if g else k
        results = (
            (b, in_channel, sy * oy + dy * fy - top, sx * ox + dx * fx - left),
            (out_channel, c, fy, fx),
            (out_channel,),
        )
        return tuple(AffineMap(num_dims, 0, r) for r in (*results[: len(inputs)], (b, out_channel, oy, ox)))

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
