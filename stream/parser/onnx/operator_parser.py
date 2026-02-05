from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Any

from onnx import ModelProto, NodeProto
from zigzag.parser.onnx.utils import (
    get_onnx_tensor_type,
)

from stream.parser.onnx.utils import onnx_tensor_to_tensor
from stream.workload.workload import HasOutputs, Tensor


class OnnxOperatorParser(metaclass=ABCMeta):
    def __init__(
        self,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
    ) -> None:
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.onnx_model = onnx_model

    def run(self, name_to_tensor_dict: dict[str, Tensor]) -> Generator[HasOutputs]:  # type: ignore
        yield self.generate_node(name_to_tensor_dict)

    @abstractmethod
    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> HasOutputs: ...

    def get_output_tensors(self) -> tuple[Tensor, ...]:
        # Get the input and output activation shapes
        onnx_tensors = [get_onnx_tensor_type(output, self.onnx_model) for output in self.node.output]
        return tuple(
            onnx_tensor_to_tensor(onnx_tensor, name=output)
            for onnx_tensor, output in zip(onnx_tensors, self.node.output, strict=False)
        )

    def get_node_attribute_ints(self, attribute_name: str) -> list[int] | None:
        for attribute in self.node.attribute:
            if attribute.name == attribute_name:
                return list(attribute.ints)
        return None
