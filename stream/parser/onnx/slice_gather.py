from onnx import numpy_helper

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.data_movement import gather_access_maps, slice_access_maps
from stream.workload.node import ComputationNode
from stream.workload.workload import Tensor


class _IndexReadingParser(OnnxOperatorParser):
    """Shared helper: read the int64 index tensors (starts/ends/axes/indices) that ONNX passes as
    initializers -- the model parser skips those (they are int), so they are read here directly."""

    def _read_ints(self, tensor_name: str) -> list[int] | None:
        for init in self.onnx_model.graph.initializer:
            if init.name == tensor_name:
                return [int(v) for v in numpy_helper.to_array(init).reshape(-1)]
        return None


class SliceParser(_IndexReadingParser):
    """Parses an ONNX Slice into an affine offset ``ComputationNode`` (the exact KV-cache read).

    Slice indexes each sliced axis as ``start + step*p`` -- an affine offset -- so the derived
    dependency recovers the precise source region moved. starts/ends/axes/steps arrive as int64
    initializer inputs; negatives and out-of-range ends are clamped to the data shape."""

    # ONNX Slice input positions: data, starts, ends, axes(opt), steps(opt).
    _STARTS, _ENDS, _AXES, _STEPS = 1, 2, 3, 4

    def _optional_ints(self, index: int) -> list[int] | None:
        inputs = self.node.input
        if len(inputs) > index and inputs[index]:
            return self._read_ints(inputs[index])
        return None

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        data = name_to_tensor_dict[self.node.input[0]]
        rank = len(data.shape)
        starts = self._read_ints(self.node.input[self._STARTS]) or []
        ends = self._read_ints(self.node.input[self._ENDS]) or []
        axes = self._optional_ints(self._AXES) or list(range(len(starts)))
        steps = self._optional_ints(self._STEPS) or [1] * len(starts)

        offsets: dict[int, int] = {}
        step_map: dict[int, int] = {}
        for start, _end, axis, step in zip(starts, ends, axes, steps, strict=False):
            norm_axis = axis % rank
            offsets[norm_axis] = start + data.shape[norm_axis] if start < 0 else start
            step_map[norm_axis] = step

        in_map, out_map = slice_access_maps(rank, offsets, step_map)
        return ComputationNode(
            type="Slice",
            name=self.node.name,
            inputs=(data,),
            outputs=self.get_output_tensors(),
            operand_mapping=(in_map, out_map),
        )


class GatherParser(_IndexReadingParser):
    """Parses an ONNX Gather into a conservative data-movement ``ComputationNode``.

    The indices are data-dependent, so the source ``axis`` is modelled as read in full (a safe
    over-approximation for paged/sparse KV-cache gathers). ONNX Gather with a scalar index removes the
    axis; with a 1-D index it keeps it at the index length -- either way ``gather_access_maps`` reads
    the whole source axis."""

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        data = name_to_tensor_dict[self.node.input[0]]
        rank = len(data.shape)
        axis_attr = self.get_node_attribute_ints("axis")
        axis = (axis_attr[0] if axis_attr else 0) % rank
        outputs = self.get_output_tensors()
        if len(outputs[0].shape) != rank:
            raise NotImplementedError(
                "Gather with a scalar index (axis-removing) is not supported yet; use a 1-D index "
                "(the paged/sparse KV-cache case, which keeps the gathered axis)."
            )
        in_map, out_map = gather_access_maps(rank, axis)
        return ComputationNode(
            type="Gather", name=self.node.name, inputs=(data,), outputs=outputs, operand_mapping=(in_map, out_map)
        )
