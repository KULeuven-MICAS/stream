from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class AddParser(OnnxOperatorParser):
    """Parses an ONNX Add operator into a ComputationNode.

    Separate from MulParser per D-14 (cleaner extension point for future broadcast support).
    Uses 4D identity maps matching ResNet18 tensor shape [batch, channels, height, width].
    """

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        mappings = (
            AffineMap.from_callable(lambda b, c, h, w: (b, c, h, w)),  # input 0
            AffineMap.from_callable(lambda b, c, h, w: (b, c, h, w)),  # input 1
            AffineMap.from_callable(lambda b, c, h, w: (b, c, h, w)),  # output
        )

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=tuple(name_to_tensor_dict[inp] for inp in self.node.input),
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
