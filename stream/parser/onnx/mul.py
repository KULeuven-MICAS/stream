from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class MulParser(OnnxComputeOperatorParser):
    """Parses an ONNX operator representing an elementwise operation (Mul) into a ComputationNode."""

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        mappings = (
            AffineMap.from_callable(lambda m, n: (m, n)),
            AffineMap.from_callable(lambda m, n: (m, n)),
            AffineMap.from_callable(lambda m, n: (m, n)),
        )

        return ComputationNode(
            name=self.node.name,
            inputs=tuple(name_to_tensor_dict[input] for input in self.node.input),
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
