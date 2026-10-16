from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from typing import Any

from onnx import ModelProto, NodeProto, helper
from zigzag.parser.onnx.utils import (
    get_onnx_tensor_type,
)

from stream.parser.onnx.utils import onnx_tensor_to_tensor
from stream.workload.utils import sliding_window
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
        return tuple(self.output_tensor(output, self.onnx_model) for output in self.node.output)

    @staticmethod
    def output_tensor(name: str, onnx_model: ModelProto) -> Tensor:
        """The tensor ``name`` a node of ``onnx_model`` outputs, its shape and element type from shape inference."""
        return onnx_tensor_to_tensor(get_onnx_tensor_type(name, onnx_model), name=name)

    def get_node_attribute_int(self, attribute_name: str) -> int | None:
        """Read a scalar INT attribute; ``get_node_attribute_ints`` reads the unrelated INTS field."""
        for attribute in self.node.attribute:
            if attribute.name == attribute_name:
                return attribute.i
        return None

    def get_node_attribute_ints(self, attribute_name: str) -> list[int] | None:
        for attribute in self.node.attribute:
            if attribute.name == attribute_name:
                return list(attribute.ints)
        return None

    def get_window(self, sizes: tuple[int, ...], kernel: tuple[int, ...]) -> tuple[list[int], ...]:
        """The node's sliding window per spatial axis, in ONNX's axis order, as :func:`sliding_window` reads it."""
        auto_pad = next((helper.get_attribute_value(a) for a in self.node.attribute if a.name == "auto_pad"), b"")
        return sliding_window(
            sizes,
            kernel,
            self.get_node_attribute_ints("strides"),
            self.get_node_attribute_ints("dilations"),
            self.get_node_attribute_ints("pads"),
            auto_pad.decode() or "NOTSET",
        )
