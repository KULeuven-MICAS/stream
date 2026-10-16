from xdsl.ir.affine import AffineDimExpr, AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.utils import window_index
from stream.workload.workload import ComputationNode, Tensor


class MaxPoolParser(OnnxOperatorParser):
    """Parses an ONNX MaxPool into a ComputationNode over (b, k, oy, ox, fy, fx): the sliding window of a Conv, read
    from its one input, with no weight operand."""

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        (activation,) = (name_to_tensor_dict[inp] for inp in self.node.input)
        b, k, oy, ox, fy, fx = (AffineDimExpr(i) for i in range(6))
        kernel = tuple(self.get_node_attribute_ints("kernel_shape") or ())
        window = self.get_window(activation.shape[2:], kernel)
        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=(activation,),
            outputs=self.get_output_tensors(),
            operand_mapping=(
                AffineMap(6, 0, (b, k, *window_index((oy, ox), (fy, fx), *window))),
                AffineMap(6, 0, (b, k, oy, ox)),
            ),
            window_extents=((4, kernel[0]), (5, kernel[1])),
        )
