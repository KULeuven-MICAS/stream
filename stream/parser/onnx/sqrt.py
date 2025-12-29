from stream.parser.onnx.simd import SimdParser


class SqrtParser(SimdParser):
    """Parses Sqrt as a SIMD op but with fake weights."""

    def get_operand_precision_user_format(self) -> dict[str, int]:
        operand_precision = super().get_operand_precision_user_format()
        operand_precision["W"] = 0
        return operand_precision
