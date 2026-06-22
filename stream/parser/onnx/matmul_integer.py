from stream.parser.onnx.gemm import GemmParser


class MatMulIntegerParser(GemmParser):
    """Parses an ONNX MatMulInteger operator into a ComputationNode.

    MatMulInteger computes an integer matrix product ``C[m, n] = sum_k A[m, k] * B[k, n]``
    with int8/uint8 inputs and an int32 result, optionally subtracting per-tensor (or
    per-row/col) zero-points supplied as the 3rd and 4th inputs.

    For Stream's purposes the iteration space, operand mapping and operand dtypes are
    identical to ``Gemm``'s ``A @ B``: the first two inputs are the matrices, and the
    optional zero-point inputs do not change the loop nest or the memory access being
    scheduled. We therefore reuse :class:`GemmParser`, which takes the first two inputs
    as the operands and ignores any trailing ones.
    """
