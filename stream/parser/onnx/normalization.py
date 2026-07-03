from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.node import NormalizationNode
from stream.workload.workload import Tensor


class NormalizationParser(OnnxOperatorParser):
    """Parses a normalization (Softmax, LpNormalization, LayerNormalization) into a NormalizationNode.

    The node is a single schedulable op (identity iteration space = the fused-kernel view); the axis
    (or axes) it normalizes over is read from the ONNX ``axis`` attribute and stored as
    ``reduction_axes`` -- the reduce-then-broadcast structure a single affine map cannot express, and
    what :func:`stream.workload.normalization.decompose_normalization` expands for fusion analysis.
    Only the primary data input participates (LayerNorm's scale/bias are per-channel params).
    """

    # Softmax/LpNormalization reduce one axis; LayerNormalization reduces axis..rank-1.
    SPANS_TO_END = {"LayerNormalization"}

    def _reduction_axes(self, rank: int) -> tuple[int, ...]:
        axis = self.get_node_attribute_ints("axis")
        raw = axis[0] if axis else -1  # ONNX default is -1 for all three ops
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

        return NormalizationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=(data,),
            outputs=outputs,
            operand_mapping=(identity, identity),
            reduction_axes=self._reduction_axes(rank),
        )
