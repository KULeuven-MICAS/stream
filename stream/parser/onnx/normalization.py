from xdsl.ir.affine import AffineExpr, AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.node import NormalizationNode
from stream.workload.workload import Tensor


class NormalizationParser(OnnxOperatorParser):
    """Parse a normalization (Softmax, LpNormalization, LayerNormalization) into a NormalizationNode,
    storing the normalized axis/axes from the ONNX ``axis`` attribute as ``reduction_axes``."""

    # Softmax/LpNormalization reduce one axis; LayerNormalization reduces axis..rank-1.
    SPANS_TO_END = {"LayerNormalization"}

    def _reduction_axes(self, rank: int) -> tuple[int, ...]:
        axis = self.get_node_attribute_int("axis")
        raw = -1 if axis is None else axis  # ONNX default is -1 for all three ops
        pos = raw if raw >= 0 else rank + raw
        if self.node.op_type in self.SPANS_TO_END:
            return tuple(range(pos, rank))
        return (pos,)

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> NormalizationNode:
        data = name_to_tensor_dict[self.node.input[0]]
        outputs = self.get_output_tensors()
        assert len(outputs) == 1, f"{self.node.op_type} must have exactly 1 output."
        rank = len(outputs[0].shape)
        identity = AffineMap.identity(rank)
        axes = self._reduction_axes(rank)
        # A layer normalization's scale and shift run along the normalized axes.
        params = tuple(name_to_tensor_dict[name] for name in self.node.input[1:] if name)
        over_axes = AffineMap(rank, 0, tuple(AffineExpr.dimension(a) for a in axes))

        return NormalizationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=(data, *params),
            outputs=outputs,
            operand_mapping=(identity, *(over_axes for _ in params), identity),
            reduction_axes=axes,
        )
