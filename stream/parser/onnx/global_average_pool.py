from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class GlobalAveragePoolParser(OnnxOperatorParser):
    """Parses an ONNX GlobalAveragePool operator into a ComputationNode.

    Iteration space: (b, c, oh, ow, ih, iw) — 6 dimensions.
    The kernel covers the entire spatial input, and the output is 1×1
    (oh and ow are size-1 dimensions derived from the output tensor shape).
    """

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        mappings = (
            AffineMap.from_callable(lambda b, c, oh, ow, ih, iw: (b, c, ih, iw)),
            AffineMap.from_callable(lambda b, c, oh, ow, ih, iw: (b, c, oh, ow)),
        )

        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        assert len(inputs) == 1, "GlobalAveragePool must have exactly 1 input."

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
