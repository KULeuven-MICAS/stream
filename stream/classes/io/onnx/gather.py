from stream.classes.workload.gather_node import GatherNode
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser
from zigzag.parser.onnx.utils import get_onnx_tensor_type


class GatherParser(ONNXOperatorParser):
    """Parses an onnx gather operator into a GatherNode."""

    def run(self):
        return self.generate_node()

    def generate_node(self):
        predecessors = self.get_node_predecessors()

        # `axis` is an attribute of the node
        try:
            axis_attr = next(filter(lambda x: x.name == "axis", self.node.attribute))
            axis = axis_attr.i
        except StopIteration:
            axis = 0

        # `indices` is the second input to the node
        indices_tensor_name = self.node.input[1]
        indices = get_onnx_tensor_type(indices_tensor_name, self.onnx_model).shape

        input_names = [self.node.input[0]]
        output_names = [self.node.output[0]]

        return GatherNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            gather_axis=axis,
            gather_indices=indices,
            input_names=input_names,
            output_names=output_names,
        )
