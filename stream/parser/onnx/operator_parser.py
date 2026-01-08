from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Any

from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.onnx_operator_parser import ONNXOperatorParser as ONNXOperatorParserZigZag
from zigzag.parser.onnx.utils import (
    get_attribute_ints_with_name,
    get_onnx_tensor_type,
)

from stream.onnx_utils import get_axis_attribute
from stream.parser.onnx.utils import onnx_tensor_to_tensor
from stream.workload.workload import HasOutput, Node, Tensor


class OnnxOperatorParser(ONNXOperatorParserZigZag, metaclass=ABCMeta):
    def __init__(
        self,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
    ) -> None:
        """'overloads' the ONNXOperatorParserZigZag init method with the correct `accelerator` type"""
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.onnx_model = onnx_model

    def run(self, name_to_node_dict: dict[str, HasOutput]) -> Generator[Node, None, None]:  # type: ignore
        yield self.generate_node(name_to_node_dict)

    @abstractmethod
    def generate_node(self, name_to_node_dict: dict[str, HasOutput]) -> Node: ...

    def get_operand_source_input_format(self):
        predecessors = self.get_node_predecessors()
        match len(predecessors):
            case 0:
                # e.g. first node of graph
                return {"W": self.node_id, "I": self.node_id}
            case 1:
                # One source operand, one constant
                return {"W": self.node_id, "I": predecessors[0]}

            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                return {"W": predecessors[0], "I": predecessors[1]}
            case _:
                raise ValueError("No more than 2 layer predecessors expected")

    def get_axis_attribute(self):
        return get_axis_attribute(self.node)


class OnnxComputeOperatorParser(OnnxOperatorParser, metaclass=ABCMeta):
    def run(self, name_to_node_dict) -> Generator[Node, None, None]:
        yield self.generate_node(name_to_node_dict)

    def get_output_activation_precision(self):
        """Return the output activation precision for this node.
        The output activation precision of ONNX nodes can be customized by manually adding the attribute
         `CUSTOM_OUTPUT_SIZE_ATTR` to the node."""
        default = 8
        try:
            return get_attribute_ints_with_name(
                name=self.CUSTOM_OUTPUT_SIZE_ATTR, attrs=self.node.attribute, default=default
            )
        except NotImplementedError as exc:
            raise ValueError("Custom activation size attribute must be an integer.") from exc

    def get_operand_precision_user_format(self) -> dict[str, int]:
        act_precision: int = self.get_activation_precision()
        weight_precision: int = self.get_weight_precision()
        intermediate_output_precision: int = self.get_intermediate_output_precision()
        output_act_precision: int = self.get_output_activation_precision()
        predecessors = self.get_node_predecessors()
        match len(predecessors):
            case 0:
                # e.g. the first node in the network -> assume only one variable input
                return {
                    "W": weight_precision,
                    "I": act_precision,
                    "O_final": output_act_precision,
                    "O": intermediate_output_precision,
                }
            case 1:
                # One source operand, one constant
                return {
                    "W": weight_precision,
                    "I": act_precision,
                    "O_final": output_act_precision,
                    "O": intermediate_output_precision,
                }
            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                return {
                    "W": act_precision,
                    "I": act_precision,
                    "O_final": output_act_precision,
                    "O": intermediate_output_precision,
                }
            case _:
                raise ValueError("No more than 2 layer predecessors expected")

    def get_mapping_this_node(self):
        """Get the mapping that corresponds to this node's operator. Replace the spatial mapping with the corresponding
        core's dataflows.
        NOTE The core's dataflow always precedes the mapping's spatial mapping
        TODO Mapping based on node name instead of note operator is not yet supported
        """
        default_mapping = self.all_mappings["default"]
        if self.node.name in self.all_mappings:
            mapping = self.all_mappings[self.node.name]
        elif self.node.op_type in self.all_mappings:
            mapping = self.all_mappings[self.node.op_type]
        else:
            mapping = default_mapping

        # Override spatial mapping by the one defined in the core's dataflows
        try:
            core_dataflow = self.accelerator.get_spatial_mapping_from_core(mapping.core_allocation)
            mapping.spatial_mapping = core_dataflow
        except ValueError:
            pass

        # If no inter/intra mapping is given: use default one
        if not mapping.intra_core_tiling:
            mapping.intra_core_tiling = default_mapping.intra_core_tiling
        if not mapping.inter_core_tiling:
            mapping.inter_core_tiling = default_mapping.inter_core_tiling

        return mapping

    def get_output_tensor(self) -> Tensor:
        # Get the input and output activation shapes
        assert len(self.node.output) == 1
        onnx_tensor = get_onnx_tensor_type(self.node.output[0], self.onnx_model)
        return onnx_tensor_to_tensor(onnx_tensor)
