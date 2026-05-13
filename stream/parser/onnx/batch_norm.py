from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class BatchNormParser(OnnxOperatorParser):
    """Parses an ONNX BatchNormalization operator into a ComputationNode.

    Per D-13: all 5 operands modeled -- input activation (4D), scale (1D),
    bias (1D), mean (1D), variance (1D), plus output (4D). 6 AffineMaps total.

    The iteration space is (b, c, h, w). The 1D per-channel operands
    project from the channel dimension only.
    """

    EXPECTED_NB_OF_INPUTS = 5

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        # 6 AffineMaps: input (4D) + scale (1D) + bias (1D) + mean (1D) + var (1D) + output (4D)
        mappings = (
            AffineMap.from_callable(lambda b, c, h, w: (b, c, h, w)),  # input activation (4D)
            AffineMap.from_callable(lambda b, c, h, w: (c,)),  # scale (1D, per-channel)
            AffineMap.from_callable(lambda b, c, h, w: (c,)),  # bias (1D, per-channel)
            AffineMap.from_callable(lambda b, c, h, w: (c,)),  # mean (1D, per-channel)
            AffineMap.from_callable(lambda b, c, h, w: (c,)),  # variance (1D, per-channel)
            AffineMap.from_callable(lambda b, c, h, w: (b, c, h, w)),  # output (4D)
        )

        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        assert len(inputs) == self.EXPECTED_NB_OF_INPUTS, (
            f"BatchNormalization must have exactly {self.EXPECTED_NB_OF_INPUTS} inputs (X, scale, B, mean, var),"
            f" got {len(inputs)}."
        )

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
