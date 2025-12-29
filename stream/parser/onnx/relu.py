from stream.parser.onnx.simd import SimdParser


class ReluParser(SimdParser):
    """Parses Relu as a SIMD op but with fake weights.

    For single-input elementwise ops (like Relu) we still model a second operand (W)
    in the internal equation. Those "weights" are not real and should not contribute
    any bit-precision cost.
    """

    def get_operand_precision_user_format(self) -> dict[str, int]:
        operand_precision = super().get_operand_precision_user_format()
        operand_precision["W"] = 0
        return operand_precision
