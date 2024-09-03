from typing import Any, Iterator

from zigzag.parser.onnx.utils import (
    get_node_input_output_dimension_shapes,
)
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.classes.io.onnx.operator_parser import OnnxOperatorParser
from stream.classes.workload.computation_node import ComputationNode
from stream.classes.workload.simd_node import SimdNode


class SoftmaxParser(OnnxOperatorParser):
    """Parses the Softmax operator"""

    def run(self) -> Iterator[ComputationNode]:
        return self.generate_node()

    def get_layer_node_input_format(self, ia_shape: list[int], oa_shape: list[int]):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        assert ia_shape == oa_shape, "Input and output of simd operation should be identical."
        predecessors = self.get_node_predecessors()
        assert len(predecessors) > 0, "Undefined behavior for Simd node with no inputs"
        # Nodes with only 1 input (e.g. Relu, Max, add/mul with constant, etc) have an empty `W` part in equation
        has_single_input = len(predecessors) == 1

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["loop_sizes"] = oa_shape
        data["dimension_relations"] = []

        match len(oa_shape):
            case 1:
                data["equation"] = f"O[k]+=I[k]*W{'[]' if has_single_input else '[k]'}"
                data["loop_dims"] = ["K"]
            case 2:
                data["equation"] = f"O[d][k]+=I[d][k]*W{'[]' if has_single_input else '[d][k]'}"
                data["loop_dims"] = ["D", "K"]
            case 3:
                data["equation"] = f"O[b][d][k]+=I[b][d][k]*W{'[]' if has_single_input else '[b][d][k]'}"
                data["loop_dims"] = ["B", "D", "k"]
            case 4:
                data["equation"] = f"O[b][h][d][k]+=I[b][h][d][k]*W{'[]' if has_single_input else '[b][h][d][k]'}"
                data["loop_dims"] = ["B", "H", "D", "k"]
            case _:
                raise NotImplementedError

        act_precision = self.get_activation_precision()
        weight_precision = self.get_weight_precision()
        intermediate_output_precision = self.get_intermediate_output_precision()
        match len(predecessors):
            case 1:
                # One source operand, one constant
                data["operand_source"] = {"W": self.node_id, "I": predecessors[0]}
                data["operand_precision"] = {
                    "W": weight_precision,
                    "I": act_precision,
                    "O_final": act_precision,
                    "O": intermediate_output_precision,
                }
            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                data["operand_source"] = {"W": predecessors[0], "I": predecessors[1]}
                data["operand_precision"] = {
                    "W": act_precision,
                    "I": act_precision,
                    "O_final": act_precision,
                    "O": intermediate_output_precision,
                }

            case _:
                raise ValueError("No more than 2 layer predecessors expected")

        return data

    def generate_node(self):
        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        node_data = self.get_layer_node_input_format(ia_dimension_shape, oa_dimension_shape)
        node_factory = LayerNodeFactory(node_data, self.mapping_data)
        node_attrs = node_factory.create_node_attr()

        # Override spatial mapping by the one defined in the core's dataflows
        core_allocation = node_attrs.core_allocation
        spatial_mapping = self.accelerator.get_spatial_mapping_from_core(core_allocation)
        node_attrs.spatial_mapping = spatial_mapping

        node_input_names = list(self.node.input)
        node_output_names = list(self.node.output)

        return SimdNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            input_names=node_input_names,
            output_names=node_output_names,
            op_type=self.node.op_type,
        )
