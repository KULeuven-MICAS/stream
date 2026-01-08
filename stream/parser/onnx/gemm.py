import logging

from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.workload import ComputationNode, HasOutput

logger = logging.getLogger(__name__)


class GemmParser(OnnxComputeOperatorParser):
    """Parses an ONNX Gemm operator into a ComputationNode"""

    def generate_node(self, name_to_node_dict: dict[str, HasOutput]) -> ComputationNode:
        mappings = (
            AffineMap.from_callable(lambda m, n, k: (m, k)),
            AffineMap.from_callable(lambda m, n, k: (k, n)),
            AffineMap.from_callable(lambda m, n, k: (m, n)),
        )

        return ComputationNode(
            name=self.node.name,
            inputs=tuple(name_to_node_dict[input] for input in self.node.input),
            output=self.get_output_tensor(),
            operand_mapping=mappings,
        )
