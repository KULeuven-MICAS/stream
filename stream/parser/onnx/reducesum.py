from typing import Any

from zigzag.parser.onnx.utils import get_attribute_ints_with_name, get_onnx_tensor_type
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.utils import get_value
from stream.workload.computation.computation_node import ComputationNode


class ReduceSumParser(OnnxComputeOperatorParser):
    """Parses an operator that reduces the data in multiple dimensions.
    e.g. sum over multiple row or max of a single row
    """

    def get_layer_node_user_format(self, input_shape: list[int], axes: list[int], output_shape: list[int]):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        # TODO check the output shape as well?
        assert len(self.get_node_predecessors()) == 1

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = input_shape

        attrs = self.node.attribute
        keep_dims = get_attribute_ints_with_name("keepdims", attrs, default=1)
        match len(input_shape):
            case 2:
                data["loop_dims"] = ["K", "C"]
                dims = ["K", "C"]
                if not keep_dims:
                    for element in axes:
                        dims.remove(data["loop_dims"][element])
                    output_indices = "".join(f"[{dim.lower()}]" for dim in dims)
                data["equation"] = f"O{output_indices}+=I[k][c]*W[]"
            case 3:
                data["loop_dims"] = ["B", "K", "C"]
                dims = ["B", "K", "C"]
                if not keep_dims:
                    for element in axes:
                        dims.remove(data["loop_dims"][element])
                output_indices = "".join(f"[{dim.lower()}]" for dim in dims)
                data["equation"] = f"O{output_indices}+=I[b][k][c]*W[]"
            case 4:
                # data["equation"] = "O[h] += I[b][h][k][c] * W[]"
                data["loop_dims"] = ["B", "H", "K", "C"]
                dims = ["B", "H", "K", "C"]
                if not keep_dims:
                    for element in axes:
                        dims.remove(data["loop_dims"][element])
                output_indices = "".join(f"[{dim.lower()}]" for dim in dims)
                data["equation"] = f"O{output_indices}+=I[b][h][k][c]*W[]"
            case _:
                raise NotImplementedError
        return data

    def generate_node(self):
        # Get the input and output activation shapes
        data_shape, axes_shape, output_shape = get_reduce_sum_input_dimension_shapes(self.node, self.onnx_model)
        axes = get_value(self.node.input[1], self.onnx_model)

        # From the ONNX node
        node_data = self.get_layer_node_user_format(data_shape, axes, output_shape)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()

        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type=self.node.op_type,
            node_attr=node_attrs,
            mapping_attr=mapping,
        )


def get_reduce_sum_input_dimension_shapes(node, model):
    """Get the input and output dimension shapes for the ReduceSumParser."""

    data_name = node.input[0]
    data_shape = get_onnx_tensor_type(data_name, model).shape
    axes_name = node.input[1]
    axes_shape = get_onnx_tensor_type(axes_name, model).shape
    output_name = node.output[0]
    output_shape = get_onnx_tensor_type(output_name, model).shape

    return data_shape, axes_shape, output_shape
