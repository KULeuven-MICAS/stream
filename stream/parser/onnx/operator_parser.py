from typing import Any, Iterator

from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.ONNXOperatorParser import ONNXOperatorParser as ONNXOperatorParserZigZag

from stream.hardware.architecture.accelerator import Accelerator
from stream.workload.computation_node import ComputationNode
from stream.workload.node import Node


class OnnxOperatorParser(ONNXOperatorParserZigZag):
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

    def run(self) -> Node | Iterator[ComputationNode]: ...  # type: ignore
