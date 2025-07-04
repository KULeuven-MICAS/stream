from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Any

from onnx import ModelProto, NodeProto
from zigzag.datatypes import Constants
from zigzag.parser.onnx.onnx_operator_parser import ONNXOperatorParser as ONNXOperatorParserZigZag
from zigzag.parser.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.hardware.architecture.accelerator import Accelerator
from stream.onnx_utils import get_axis_attribute
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.mapping import InterCoreMappingAttributes
from stream.workload.node import Node


class OnnxOperatorParser(ONNXOperatorParserZigZag, metaclass=ABCMeta):
    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
        all_mappings: dict[str, InterCoreMappingAttributes],
        accelerator: Accelerator,
    ) -> None:
        """'overloads' the ONNXOperatorParserZigZag init method with the correct `accelerator` type"""
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.onnx_model = onnx_model
        self.all_mappings = all_mappings
        self.accelerator = accelerator

    def run(self) -> Generator[Node, None, None]:  # type: ignore
        yield self.generate_node()

    @abstractmethod
    def generate_node(self) -> Node: ...

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
    def run(self) -> Generator[ComputationNode, None, None]:
        yield self.generate_node()

    @abstractmethod
    def get_layer_node_user_format(
        self, input_shape: list[int], output_shape: list[int], mapping: InterCoreMappingAttributes | None
    ) -> dict[str, Any]: ...

    def get_operand_precision_user_format(self) -> dict[str, int]:
        act_precision: int = self.get_activation_precision()
        weight_precision: int = self.get_weight_precision()
        intermediate_output_precision: int = self.get_intermediate_output_precision()
        predecessors = self.get_node_predecessors()
        match len(predecessors):
            case 0:
                # e.g. the first node in the network -> assume only one variable input
                return {
                    "W": weight_precision,
                    "I": act_precision,
                    "O_final": act_precision,
                    "O": intermediate_output_precision,
                }
            case 1:
                # One source operand, one constant
                return {
                    "W": weight_precision,
                    "I": act_precision,
                    "O_final": act_precision,
                    "O": intermediate_output_precision,
                }
            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                return {
                    "W": act_precision,
                    "I": act_precision,
                    "O_final": act_precision,
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

    def generate_node(self):
        # Get the input and output activation shapes
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # From the ONNX node
        mapping = self.get_mapping_this_node()
        node_data = self.get_layer_node_user_format(input_shape, output_shape, mapping)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()
        input_names = list(self.node.input)

        # ! Messy patchwork for KV caches. Assumes that operators that take in the cache have a special name
        if "_cache" in self.node.name:
            partially_constant_operands = [Constants.LAYER_OP_W]
        else:
            partially_constant_operands = []

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type=self.node.op_type,
            node_attr=node_attrs,
            mapping_attr=mapping,
            input_names=input_names,
            partially_constant_operands=partially_constant_operands,
        )
