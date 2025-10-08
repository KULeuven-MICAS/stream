from zigzag.parser.onnx.utils import get_onnx_tensor_type
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.parser.onnx.simd import SimdParser
from stream.workload.computation.computation_node import ComputationNode


class InPlaceAccumulatorParser(SimdParser):
    def generate_node(self):
        # Get the input and output activation shapes
        accumulator_shape, grad_shape = inplaceaccumulator_get_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

        # From the ONNX node
        mapping = self.get_mapping_this_node()
        node_data = self.get_layer_node_user_format(grad_shape, accumulator_shape, mapping)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()

        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type="accumulate",
            node_attr=node_attrs,
            mapping_attr=mapping,
        )


def inplaceaccumulator_get_node_input_output_dimension_shapes(node, model):
    # assumed it is the first input, don't see a way to otherwise know

    accumulator_name = node.input[0]
    accumulator_shape = get_onnx_tensor_type(accumulator_name, model).shape
    grad_name = node.input[1]
    grad_shape = get_onnx_tensor_type(grad_name, model).shape

    # output_name = node.output[0]
    # output_shape = get_onnx_tensor_type(output_name, model).shape

    return accumulator_shape, grad_shape
