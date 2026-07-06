"""torch.export frontend (plan/08 spike -> tier-1) -- a moat-clean, optional plugin.

``torch.export`` preserves op provenance (unlike ONNX's flattening), which is what M04's rewrite
matching and M05's block detection want. This frontend converts an ``ExportedProgram``'s ATen graph
into the internal affine :class:`~stream.workload.workload.Workload`.

Design for soundness + the moat boundary:

- **torch is optional.** Importing this module never imports torch; only :meth:`TorchExportFrontend.load`
  does. Without torch the frontend registers but declines every source (``can_load`` is False), exactly
  like the islpy exact path.
- **The conversion is torch-free and testable.** The torch glue only lowers an ``ExportedProgram`` to a
  list of :class:`AtenCall` (a tiny torch-independent op record); :func:`convert_aten_calls` maps those
  to affine nodes via the declarative :data:`ATEN_OP_TABLE`. So the whole mapping is unit-tested with
  hand-built calls, no torch required.
- **The op table is the extension point.** A private overlay adds SOTA ATen coverage (or production
  block converters) via :func:`register_aten_op` -- no fork. The public table is a starter subset; ops
  with no builder are reported in an :class:`UnsupportedOpReport`, never crashed on (plan/08).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import FixedBitwidthType, bf16
from xdsl.ir.affine import AffineExpr, AffineMap

from stream.frontends import FrontendConfig, register_frontend
from stream.workload.node import ComputationNode, InEdge, Node, NormalizationNode, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

__all__ = [
    "AtenTensor",
    "AtenCall",
    "UnsupportedOpReport",
    "AtenNodeBuilder",
    "register_aten_op",
    "convert_aten_calls",
    "TorchExportFrontend",
]


@dataclass(frozen=True)
class AtenTensor:
    """A tensor in the exported graph -- torch-independent, so conversion is testable without torch."""

    name: str
    shape: tuple[int, ...]
    dtype: FixedBitwidthType = bf16


@dataclass(frozen=True)
class AtenCall:
    """One ATen op: its target (e.g. ``"aten::linear"``), input tensor names, output tensor, and attrs."""

    target: str
    inputs: tuple[str, ...]
    output: AtenTensor
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnsupportedOpReport:
    """ATen targets with no registered builder, as counts -- the per-release coverage frontier."""

    counts: Counter[str] = field(default_factory=Counter)

    def add(self, target: str) -> None:
        self.counts[target] += 1

    @property
    def is_complete(self) -> bool:
        return not self.counts


# A builder turns one call (its resolved input Tensors + output Tensor) into an affine node.
AtenNodeBuilder = Callable[[tuple[Tensor, ...], Tensor, AtenCall], ComputationNode]

ATEN_OP_TABLE: dict[str, AtenNodeBuilder] = {}


def register_aten_op(target: str, builder: AtenNodeBuilder) -> None:
    """Register (or override) the affine-node builder for an ATen ``target`` -- the overlay's seam."""
    ATEN_OP_TABLE[target] = builder


def _identity(rank: int) -> AffineMap:
    return AffineMap.identity(rank)


def _elementwise(inputs: tuple[Tensor, ...], output: Tensor, call: AtenCall) -> ComputationNode:
    """Unary/binary elementwise op: identity access, the op semantics carried as the node ``type``."""
    rank = len(output.shape)
    maps = tuple(_identity(rank) for _ in inputs) + (_identity(rank),)
    return ComputationNode(
        type=call.target, name=call.output.name, inputs=inputs, outputs=(output,), operand_mapping=maps
    )


def _matmul_2d(inputs: tuple[Tensor, ...], output: Tensor, call: AtenCall) -> ComputationNode:
    """``aten::mm``/``matmul`` on 2-D operands: ``O[m,n] = sum_k A[m,k] B[k,n]`` (dims m, n, k)."""
    dim = AffineExpr.dimension
    a_map = AffineMap(3, 0, (dim(0), dim(2)))
    b_map = AffineMap(3, 0, (dim(2), dim(1)))
    o_map = AffineMap(3, 0, (dim(0), dim(1)))
    return ComputationNode(
        type="MatMul",
        name=call.output.name,
        inputs=inputs[:2],
        outputs=(output,),
        operand_mapping=(a_map, b_map, o_map),
    )


def _linear(inputs: tuple[Tensor, ...], output: Tensor, call: AtenCall) -> ComputationNode:
    """``aten::linear``: ``y[m,n] = sum_k x[m,k] W[n,k]`` (weight is transposed; bias is a cost-negligible
    add, omitted from the structural spike). Contracts k = REDUCTION."""
    dim = AffineExpr.dimension
    x_map = AffineMap(3, 0, (dim(0), dim(2)))
    w_map = AffineMap(3, 0, (dim(1), dim(2)))
    o_map = AffineMap(3, 0, (dim(0), dim(1)))
    return ComputationNode(
        type="MatMul",
        name=call.output.name,
        inputs=inputs[:2],
        outputs=(output,),
        operand_mapping=(x_map, w_map, o_map),
    )


def _softmax(inputs: tuple[Tensor, ...], output: Tensor, call: AtenCall) -> ComputationNode:
    """``aten::softmax``: a NormalizationNode reducing over ``attrs['dim']`` (default the last axis)."""
    rank = len(output.shape)
    axis = int(call.attrs.get("dim", -1)) % rank
    return NormalizationNode(
        type="Softmax",
        name=call.output.name,
        inputs=inputs,
        outputs=(output,),
        operand_mapping=(_identity(rank), _identity(rank)),
        reduction_axes=(axis,),
    )


# Starter subset -- the public reference coverage. The overlay extends it via register_aten_op.
for _target in (
    "aten::mul",
    "aten::add",
    "aten::sub",
    "aten::div",
    "aten::relu",
    "aten::silu",
    "aten::gelu",
    "aten::sigmoid",
    "aten::tanh",
):
    register_aten_op(_target, _elementwise)
register_aten_op("aten::mm", _matmul_2d)
register_aten_op("aten::matmul", _matmul_2d)
register_aten_op("aten::linear", _linear)
register_aten_op("aten::softmax", _softmax)


def convert_aten_calls(
    graph_inputs: list[AtenTensor], calls: list[AtenCall], output_name: str | None = None
) -> tuple[Workload, UnsupportedOpReport]:
    """Build a :class:`~stream.workload.workload.Workload` from graph inputs + ATen calls, plus a coverage
    report. Unmapped ops are recorded and skipped (their outputs become graph inputs downstream), never
    crashed on. Pure -- no torch."""
    tensors: dict[str, Tensor] = {}
    nodes: list[Node] = []
    report = UnsupportedOpReport()

    for graph_input in graph_inputs:
        tensor = Tensor.create(graph_input.name, graph_input.dtype, graph_input.shape)
        tensors[graph_input.name] = tensor
        nodes.append(InEdge(name=graph_input.name, outputs=(tensor,)))

    for call in calls:
        builder = ATEN_OP_TABLE.get(call.target)
        output = Tensor.create(call.output.name, call.output.dtype, call.output.shape)
        if builder is None:
            report.add(call.target)  # coverage gap: reported, not fatal
            tensors[call.output.name] = output
            nodes.append(InEdge(name=call.output.name, outputs=(output,)))
            continue
        inputs = tuple(tensors[name] for name in call.inputs if name in tensors)
        nodes.append(builder(inputs, output, call))
        tensors[call.output.name] = output

    sink = output_name or (calls[-1].output.name if calls else None)
    if sink is not None and sink in tensors:
        nodes.append(OutEdge(name=f"{sink}_out", inputs=(tensors[sink],)))
    return Workload(nodes), report


class TorchExportFrontend:
    """Loads a torch ``ExportedProgram`` into a workload. torch is optional; without it, this frontend
    declines every source, so the public package never depends on torch."""

    name = "torch_export"

    def can_load(self, source: Any) -> bool:
        try:
            from torch.export import ExportedProgram  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            return False
        return isinstance(source, ExportedProgram)

    def load(self, source: Any, config: FrontendConfig | None = None) -> Workload:
        graph_inputs, calls, output_name = _lower_exported_program(source)
        workload, report = convert_aten_calls(graph_inputs, calls, output_name)
        if config is not None:
            config.options["unsupported_ops"] = report
        return workload


def _lower_exported_program(exported_program: Any) -> tuple[list[AtenTensor], list[AtenCall], str | None]:
    """Torch glue (guarded): lower an ``ExportedProgram`` to torch-free AtenCalls. Raises if torch is
    absent. Kept thin so the conversion (:func:`convert_aten_calls`) stays testable without torch."""
    try:
        import torch  # type: ignore[import]  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only in the torch-absent job
        raise RuntimeError("the torch.export frontend requires torch; install the 'stream[torch]' extra") from exc

    def as_tensor(node: Any) -> AtenTensor:
        val = node.meta.get("val")
        shape = tuple(int(s) for s in val.shape) if val is not None else ()
        return AtenTensor(name=str(node.name), shape=shape)

    graph_inputs: list[AtenTensor] = []
    calls: list[AtenCall] = []
    output_name: str | None = None
    graph = exported_program.graph_module.graph
    for node in graph.nodes:
        if node.op == "placeholder":
            graph_inputs.append(as_tensor(node))
        elif node.op == "call_function":
            target = f"aten::{getattr(node.target, '_opname', str(node.target))}"
            inputs = tuple(str(a.name) for a in node.args if hasattr(a, "name"))
            calls.append(AtenCall(target=target, inputs=inputs, output=as_tensor(node), attrs=dict(node.kwargs)))
        elif node.op == "output":
            args = node.args[0] if node.args else ()
            first = args[0] if isinstance(args, (tuple, list)) and args else None
            output_name = str(first.name) if first is not None and hasattr(first, "name") else None
    return graph_inputs, calls, output_name


# Register the built-in so `import stream.frontends.torch_export` makes it available (torch stays lazy).
register_frontend(TorchExportFrontend())
