from abc import ABCMeta, abstractmethod
from typing import Any, Generator

from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.onnx_operator_parser import ONNXOperatorParser as ONNXOperatorParserZigZag
from zigzag.parser.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.hardware.architecture.accelerator import Accelerator
from stream.workload.computation_node import ComputationNode
from stream.workload.node import Node


class OnnxOperatorParser(ONNXOperatorParserZigZag, metaclass=ABCMeta):
    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
        mapping_data: list[dict[str, Any]],
        accelerator: Accelerator,
    ) -> None:
        """'overloads' the ONNXOperatorParserZigZag init method with the correct `accelerator` type"""
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.onnx_model = onnx_model
        self.mapping_data = mapping_data
        self.accelerator = accelerator

    def run(self) -> Generator[Node, None, None]:  # type: ignore
        yield self.generate_node()

    @abstractmethod
    def generate_node(self) -> Node: ...

    def get_operand_source_input_format(self):
        predecessors = self.get_node_predecessors()
        match len(predecessors):
            case 1:
                # One source operand, one constant
                return {"W": self.node_id, "I": predecessors[0]}

            case 2:
                # Two source operands, none are constant (W and I can be swapped)
                return {"W": predecessors[0], "I": predecessors[1]}
            case _:
                raise ValueError("No more than 2 layer predecessors expected")


class OnnxComputeOperatorParser(OnnxOperatorParser, metaclass=ABCMeta):

    def run(self) -> Generator[ComputationNode, None, None]:
        yield self.generate_node()

    @abstractmethod
    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]) -> dict[str, Any]: ...

    def get_operand_precision_input_format(self):
        act_precision = self.get_activation_precision()
        weight_precision = self.get_weight_precision()
        intermediate_output_precision = self.get_intermediate_output_precision()
        predecessors = self.get_node_predecessors()
        match len(predecessors):
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

    def generate_node(self):
        # Get the input and output activation shapes
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        node_data = self.get_layer_node_user_format(input_shape, output_shape)
        node_factory = LayerNodeFactory(node_data, self.mapping_data)
        node_attrs = node_factory.create_node_attr()

        # Override spatial mapping by the one defined in the core's dataflows
        core_allocation = node_attrs.core_allocation
        spatial_mapping = self.accelerator.get_spatial_mapping_from_core(core_allocation)
        node_attrs.spatial_mapping = spatial_mapping

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            node_attr=node_attrs,
            op_type=self.node.op_type,
        )
