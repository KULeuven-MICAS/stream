from xdsl.ir.affine import AffineExpr, AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class MatMulParser(OnnxOperatorParser):
    """Parses an ONNX MatMul operator into an affine ``ComputationNode``.

    MatMul is the workhorse of attention (Q@K^T scores, P@V context) and of the linear
    projections in every transformer/MLP block. Its access relation is fully affine, so it is
    represented directly with ``operand_mapping`` -- no FusionEdge needed.

    Semantics follow ``numpy.matmul``: the last two axes are the matrix axes ``(m, k) @ (k, n)``
    and any leading axes are batch axes that broadcast between the two inputs (numpy rules,
    right-aligned). The iteration space is ``(batch..., m, n, k)`` with ``k`` the single
    contraction (REDUCTION) dimension. A batch axis that an operand broadcasts over (size 1 or
    absent) indexes that operand at constant 0, which keeps the map affine.
    """

    EXPECTED_NB_OF_INPUTS = 2  # A and B

    def _batch_exprs(self, operand_batch: tuple[int, ...], out_batch: tuple[int, ...]) -> list[AffineExpr]:
        """Index expressions for an operand's batch axes: the iteration dim when the axis matches
        the (broadcast) output extent, constant 0 when the operand broadcasts over it."""
        num_batch = len(out_batch)
        offset = num_batch - len(operand_batch)  # right-align operand batch axes to the output's
        exprs: list[AffineExpr] = []
        for j, size in enumerate(operand_batch):
            out_pos = offset + j
            if size == out_batch[out_pos] and size != 1:
                exprs.append(AffineExpr.dimension(out_pos))
            else:
                exprs.append(AffineExpr.constant(0))
        return exprs

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        assert len(inputs) == self.EXPECTED_NB_OF_INPUTS, f"MatMul expects 2 inputs, got {len(inputs)}."
        a, b = inputs
        if len(a.shape) < 2 or len(b.shape) < 2:  # noqa: PLR2004
            raise NotImplementedError("1D MatMul (vector) operands are not supported; provide rank >= 2 tensors.")
        assert a.shape[-1] == b.shape[-2], f"MatMul contraction mismatch: A[..,{a.shape[-1]}] vs B[{b.shape[-2]},..]."

        outputs = self.get_output_tensors()
        assert len(outputs) == 1, "MatMul must have exactly 1 output."
        out_batch = outputs[0].shape[:-2]
        num_batch = len(out_batch)

        # Iteration dims: batch_0..batch_{p-1}, then m, n, k.
        pos_m, pos_n, pos_k = num_batch, num_batch + 1, num_batch + 2
        num_dims = num_batch + 3

        a_map = AffineMap(
            num_dims,
            0,
            tuple(self._batch_exprs(a.shape[:-2], out_batch))
            + (AffineExpr.dimension(pos_m), AffineExpr.dimension(pos_k)),
        )
        b_map = AffineMap(
            num_dims,
            0,
            tuple(self._batch_exprs(b.shape[:-2], out_batch))
            + (AffineExpr.dimension(pos_k), AffineExpr.dimension(pos_n)),
        )
        out_map = AffineMap(
            num_dims,
            0,
            tuple(AffineExpr.dimension(i) for i in range(num_batch))
            + (AffineExpr.dimension(pos_m), AffineExpr.dimension(pos_n)),
        )

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=outputs,
            operand_mapping=(a_map, b_map, out_map),
        )
