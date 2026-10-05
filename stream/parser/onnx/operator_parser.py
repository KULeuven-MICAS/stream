from abc import ABCMeta, abstractmethod
from collections.abc import Generator
from math import ceil
from typing import Any

from onnx import ModelProto, NodeProto, helper
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

    def get_window(self, sizes: tuple[int, ...], kernel: tuple[int, ...]) -> tuple[list[int], list[int], list[int]]:
        """A sliding window's strides, dilations and leading padding per spatial axis, in ONNX's axis order: the
        ``pads`` attribute, or what ``auto_pad`` derives from the input ``sizes`` and the ``kernel``."""
        strides = self.get_node_attribute_ints("strides") or [1] * len(sizes)
        dilations = self.get_node_attribute_ints("dilations") or [1] * len(sizes)
        auto_pad = next((helper.get_attribute_value(a) for a in self.node.attribute if a.name == "auto_pad"), b"")
        if auto_pad in (b"", b"NOTSET"):
            return strides, dilations, (self.get_node_attribute_ints("pads") or [0] * len(sizes))[: len(sizes)]
        totals = [
            max(0, (ceil(n / s) - 1) * s + (f - 1) * d + 1 - n) if auto_pad != b"VALID" else 0
            for n, f, s, d in zip(sizes, kernel, strides, dilations, strict=True)
        ]
        return strides, dilations, [t // 2 if auto_pad == b"SAME_UPPER" else t - t // 2 for t in totals]
