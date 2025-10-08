from onnx import numpy_helper

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.dependency_propagation.pad_node import PadNode


class PadParser(OnnxOperatorParser):
    """Parses an onnx pad operator into a PadNode."""

    def generate_node(self):
        predecessors = self.get_node_predecessors()
        padding = self.get_padding()
        input_names = list(self.node.input)
        return PadNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessors,
            input_names=input_names,
            padding=padding,
        )

    def get_padding(self):
        """Find the value of the padding tensor associated with this pad node in ONNX"""
        DEFAULT = 0

        # `indices` is the second input to the node
        padding_tensor_name = self.node.input[1]
        # Try to find the padding in the graph nodes first
        try:
            padding_tensor = next(
                filter(
                    lambda x: x.output[0] == padding_tensor_name and x.op_type in "Constant", self.onnx_model.graph.node
                )
            )
            padding_attr = next(filter(lambda x: x.name == "value", padding_tensor.attribute))
            padding_array = numpy_helper.to_array(padding_attr.t)  # type: ignore
            padding = list(padding_array) if len(padding_array.shape) > 0 else DEFAULT  # type: ignore
            # The ONNX padding format is (x1_start, x2_start,...x1_end, x2_end, ...) and
            # numpy expects (x1_start, x1_end, x2_start, x2_end, ...)
            padding2 = []
            for i in range(len(padding) // 2):
                padding2.append((padding[i].item(), padding[len(padding) // 2 + i].item()))
            padding = padding2
        except StopIteration:
            # Try to find the padding in the initializers
            try:
                padding_tensor = next(
                    filter(lambda x: x.name == padding_tensor_name, self.onnx_model.graph.initializer)
                )
                padding_array = numpy_helper.to_array(padding_tensor)
                padding = list(padding_array) if len(padding_array.shape) > 0 else DEFAULT  # type: ignore
                # The ONNX padding format is (x1_start, x2_start,...x1_end, x2_end, ...) and
                # numpy expects (x1_start, x1_end, x2_start, x2_end, ...)
                padding2 = []
                for i in range(len(padding) // 2):
                    padding2.append((padding[i].item(), padding[len(padding) // 2 + i].item()))
                padding = padding2
            except StopIteration:
                padding = DEFAULT

        return padding
