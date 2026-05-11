from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class MaxPoolParser(OnnxOperatorParser):
    """Parses an ONNX MaxPool operator into a ComputationNode.

    Uses 2 AffineMaps (input activation + output) per D-16 -- no fake weight operand.
    """

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        strides = self.get_node_attribute_ints("strides")
        if not strides:
            strides = [1, 1]
        assert len(strides) == 2, "MaxPool strides must be 2D."
        sx, sy = strides

        pads = self.get_node_attribute_ints("pads")
        if not pads:
            pads = [0, 0, 0, 0]
        if not all(p == pads[0] for p in pads):
            raise NotImplementedError("Asymmetric padding not supported for MaxPool.")
        p = pads[0]

        # 2 AffineMaps: input activation + output (D-16)
        mappings = (
            AffineMap.from_callable(lambda b, k, oy, ox, fy, fx: (b, k, sy * oy + fy - p, sx * ox + fx - p)),
            AffineMap.from_callable(lambda b, k, oy, ox, fy, fx: (b, k, oy, ox)),
        )

        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        assert len(inputs) == 1, "MaxPool must have exactly 1 input."

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
