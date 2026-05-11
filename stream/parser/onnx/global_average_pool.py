from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class GlobalAveragePoolParser(OnnxOperatorParser):
    """Parses an ONNX GlobalAveragePool operator into a ComputationNode.

    Uses 2 AffineMaps (input activation + output) per D-16.
    The kernel covers the entire spatial dimension, so the iteration space
    is (b, c, ih, iw) and the output spatial indices are constant 0.
    """

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        # Try constant output map; fall back to multiplication by 0
        try:
            mappings = (
                AffineMap.from_callable(lambda b, c, ih, iw: (b, c, ih, iw)),
                AffineMap.from_callable(lambda b, c, ih, iw: (b, c, 0, 0)),
            )
        except (TypeError, ValueError):
            mappings = (
                AffineMap.from_callable(lambda b, c, ih, iw: (b, c, ih, iw)),
                AffineMap.from_callable(lambda b, c, ih, iw: (b, c, ih * 0, iw * 0)),
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
