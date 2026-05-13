from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.node import FusionEdge
from stream.workload.workload import Tensor


class FusionEdgeParser(OnnxOperatorParser):
    """Parses shape-only ONNX ops (Flatten, Reshape) into FusionEdge nodes.

    FusionEdge is not a ComputationNode -- it has no iteration space or
    operand_mapping. Output shapes come from ONNX shape inference (D-06).
    """

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> FusionEdge:
        inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input if inp in name_to_tensor_dict)
        outputs = self.get_output_tensors()

        return FusionEdge(
            name=self.node.name,
            inputs=inputs,
            outputs=outputs,
            op_type=self.node.op_type,
        )
