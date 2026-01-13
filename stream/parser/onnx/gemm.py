import logging

from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.workload import ComputationNode, HasOutputs

logger = logging.getLogger(__name__)


class GemmParser(OnnxComputeOperatorParser):
    """Parses an ONNX Gemm operator into a ComputationNode"""

    def generate_node(self, name_to_node_dict: dict[str, HasOutputs]) -> ComputationNode:
        mappings = (
            AffineMap.from_callable(lambda m, k, n: (m, k)),
            AffineMap.from_callable(lambda m, k, n: (k, n)),
            AffineMap.from_callable(lambda m, k, n: (m, n)),
        )

        return ComputationNode(
            name=self.node.name,
            inputs=tuple(name_to_node_dict[input] for input in self.node.input),
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
