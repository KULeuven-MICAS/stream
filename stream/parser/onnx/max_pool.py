from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class MaxPoolParser(OnnxOperatorParser):
    """Parses an ONNX MaxPool operator into a ComputationNode.

    Uses 2 AffineMaps (input activation + output) per D-16 -- no fake weight operand.
    """

    EXPECTED_NB_OF_INPUTS = 1
    EXPECTED_NB_OF_STRIDES = 2
    EXPECTED_NB_OF_PADS = 4

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        strides = self.get_node_attribute_ints("strides")
        if not strides:
            strides = [1, 1]
        assert len(strides) == self.EXPECTED_NB_OF_STRIDES, (
            f"MaxPool strides must be {self.EXPECTED_NB_OF_STRIDES}D, got {len(strides)}."
        )
        sx, sy = strides

        pads = self.get_node_attribute_ints("pads")
        if not pads:
            pads = [0, 0, 0, 0]
        assert len(pads) == self.EXPECTED_NB_OF_PADS, (
            f"MaxPool pads must be {self.EXPECTED_NB_OF_PADS}D, got {len(pads)}."
        )
        if not all(p == pads[0] for p in pads):
            raise NotImplementedError("Asymmetric padding not supported for MaxPool.")
        p = pads[0]

        # 2 AffineMaps: input activation + output (D-16)
        mappings = (
            AffineMap.from_callable(lambda b, k, oy, ox, fy, fx: (b, k, sy * oy + fy - p, sx * ox + fx - p)),
            AffineMap.from_callable(lambda b, k, oy, ox, fy, fx: (b, k, oy, ox)),
        )

        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        assert len(inputs) == self.EXPECTED_NB_OF_INPUTS, (
            f"MaxPool must have exactly {self.EXPECTED_NB_OF_INPUTS} input."
        )

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
