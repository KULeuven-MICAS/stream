from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.node import FusionEdge
from stream.workload.workload import Tensor


class FusionEdgeParser(OnnxOperatorParser):
    """Parses shape-only ONNX ops (Flatten, Reshape) into FusionEdge nodes.

    FusionEdge is not a ComputationNode -- it has no iteration space or
    operand_mapping. Output shapes come from ONNX shape inference (D-06).
    """

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> FusionEdge:
        # FusionEdge is a shape-only boundary node. Only the first input (data tensor)
        # is a real data-flow edge. Additional inputs (e.g., Reshape's shape tensor)
        # are metadata and must not become graph edges — split_fusion_groups()
        # asserts len(fe.inputs) == 1.
        data_input_name = self.node.input[0]
        inputs = (name_to_tensor_dict[data_input_name],)
        outputs = self.get_output_tensors()

        return FusionEdge(
            name=self.node.name,
            inputs=inputs,
            outputs=outputs,
            op_type=self.node.op_type,
        )
