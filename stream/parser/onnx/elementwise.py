from xdsl.ir.affine import AffineExpr, AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class ElementwiseParser(OnnxOperatorParser):
    """Parses any elementwise ONNX op (Add, Sub, Mul, Div, Relu, Silu, Gelu, Sigmoid, Tanh, ...)
    into an affine ``ComputationNode``.

    Every operand shares one identity iteration space at the output's rank: each output index is
    read from the same index of every input. NumPy-style broadcasting is affine -- an input axis of
    size 1 (or a missing leading axis) indexes that operand at constant 0. This single parser
    replaces the former per-op Add/Mul/Relu/Simd parsers, which were byte-for-byte identity maps
    differing only in hard-coded rank. Rank is inferred from the output tensor, so the same class
    serves 2D (SwiGLU) and 4D (ResNet) tensors and the scalar/vector broadcasts of attention.
    """

    def _operand_map(self, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> AffineMap:
        out_rank = len(out_shape)
        offset = out_rank - len(in_shape)  # right-align input axes to the output's
        results: list[AffineExpr] = []
        for j, size in enumerate(in_shape):
            out_pos = offset + j
            if size == out_shape[out_pos]:
                results.append(AffineExpr.dimension(out_pos))
            elif size == 1:
                results.append(AffineExpr.constant(0))  # broadcast this axis
            else:
                raise NotImplementedError(
                    f"Elementwise broadcast mismatch on {self.node.op_type}: operand axis {j} "
                    f"size {size} is incompatible with output size {out_shape[out_pos]}."
                )
        return AffineMap(out_rank, 0, tuple(results))

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        outputs = self.get_output_tensors()
        assert len(outputs) == 1, f"{self.node.op_type} must have exactly 1 output."
        out_shape = outputs[0].shape
        in_maps = tuple(self._operand_map(inp.shape, out_shape) for inp in inputs)
        out_map = AffineMap.identity(len(out_shape))

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=outputs,
            operand_mapping=(*in_maps, out_map),
        )
