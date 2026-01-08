from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.workload import ComputationNode, HasOutput


class MulParser(OnnxComputeOperatorParser):
    """Parses an ONNX operator representing an elementwise operation (Mul) into a ComputationNode."""

    def generate_node(self, name_to_node_dict: dict[str, HasOutput]) -> ComputationNode:
        mappings = (
            AffineMap.from_callable(lambda m, n: (m, n)),
            AffineMap.from_callable(lambda m, n: (m, n)),
            AffineMap.from_callable(lambda m, n: (m, n)),
        )

        return ComputationNode(
            name=self.node.name,
            inputs=tuple(name_to_node_dict[input] for input in self.node.input),
            output=self.get_output_tensor(),
            operand_mapping=mappings,
        )
