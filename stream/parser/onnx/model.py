import logging
from typing import Any

import onnx
from onnx import NodeProto, TensorProto
from zigzag.parser.onnx.utils import parse_onnx_model_from_path

from stream.parser.onnx.batch_norm import BatchNormParser
from stream.parser.onnx.conv import ConvParser
from stream.parser.onnx.elementwise import CastParser, ElementwiseParser
from stream.parser.onnx.fusion_edge import FusionEdgeParser
from stream.parser.onnx.gemm import GemmParser
from stream.parser.onnx.global_average_pool import GlobalAveragePoolParser
from stream.parser.onnx.matmul import MatMulParser
from stream.parser.onnx.max_pool import MaxPoolParser
from stream.parser.onnx.normalization import NormalizationParser
from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.parser.onnx.slice_gather import GatherParser, SliceParser
from stream.parser.onnx.utils import onnx_tensor_to_tensor
from stream.workload.workload import InEdge, Node, OutEdge, Tensor, Workload

logger = logging.getLogger(__name__)

# Out-of-tree parsers, added or overridden via the ``stream.onnx_parsers`` entry-point group; kept
# separate from the built-in table.
_REGISTERED_PARSERS: dict[str, type[OnnxOperatorParser]] = {}
_PARSER_PLUGINS_LOADED = {"done": False}


def register_onnx_parser(op_type: str, parser_class: type[OnnxOperatorParser]) -> None:
    """Register (or override) the ONNX parser for ``op_type`` -- the seam for out-of-tree parsers."""
    _REGISTERED_PARSERS[op_type] = parser_class


def _load_parser_plugins() -> None:
    """Discover out-of-tree parsers under the ``stream.onnx_parsers`` entry-point group (name = op_type,
    object = parser class)."""
    if _PARSER_PLUGINS_LOADED["done"]:
        return
    _PARSER_PLUGINS_LOADED["done"] = True
    from stream.plugins import load_group  # noqa: PLC0415

    for plugin in load_group("stream.onnx_parsers"):
        register_onnx_parser(plugin.name, plugin.obj)


def onnx_parser_for(op_type: str) -> type[OnnxOperatorParser] | None:
    """The parser for ``op_type``: a registered parser takes precedence over the built-in table."""
    _load_parser_plugins()
    return _REGISTERED_PARSERS.get(op_type) or ONNXModelParser.OP_TYPE_TO_PARSER.get(op_type)


class ONNXModelParser:
    """Parse the ONNX model into a workload."""

    # Layout-only ops (pure re-indexing, no compute) -> FusionEdgeParser: a fusion-graph boundary, not
    # an affine ComputationNode. (Normalizations are schedulable NormalizationNodes, not here.)
    FUSION_EDGE_OPS: set[str] = {
        "Flatten",
        "Reshape",
        "Transpose",
        "Squeeze",
        "Unsqueeze",
    }

    # op_type -> affine-ComputationNode parser (elementwise ops share one; MatMul/Gemm/Conv carry their own).
    OP_TYPE_TO_PARSER: dict[str, type[OnnxOperatorParser]] = {
        "Conv": ConvParser,
        "Gemm": GemmParser,
        # The score GEMM with its online softmax folded in: a Gemm to the tiling
        # machinery, since the softmax is elementwise over the block it produces.
        "MatmulSoftmax": GemmParser,
        "MatMul": MatMulParser,
        "MaxPool": MaxPoolParser,
        "GlobalAveragePool": GlobalAveragePoolParser,
        "BatchNormalization": BatchNormParser,
        # Normalizations (reduce-then-broadcast) -> a single schedulable NormalizationNode
        "Softmax": NormalizationParser,
        "LpNormalization": NormalizationParser,
        "LayerNormalization": NormalizationParser,
        # Data-movement / indexing (KV cache) -> access ComputationNodes carrying the moved region
        "Slice": SliceParser,
        "Gather": GatherParser,
        # Elementwise (unary and binary, NumPy broadcast) -> ElementwiseParser
        "Add": ElementwiseParser,
        "Sub": ElementwiseParser,
        "Mul": ElementwiseParser,
        "Div": ElementwiseParser,
        "Pow": ElementwiseParser,
        "Relu": ElementwiseParser,
        "Silu": ElementwiseParser,
        "PartialSoftmax": ElementwiseParser,
        "Gelu": ElementwiseParser,
        "Sigmoid": ElementwiseParser,
        "Tanh": ElementwiseParser,
        "Cast": CastParser,
        "QuantizeLinear": CastParser,
        "DequantizeLinear": CastParser,
    }

    def __init__(self, onnx_model_path: str) -> None:
        self.onnx_model_path = onnx_model_path

    def run(self):
        """Parse the ONNX model at ``onnx_model_path`` into a ``Workload``."""
        self.onnx_model = parse_onnx_model_from_path(self.onnx_model_path)
        self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        fold_quantize_dequantize(self.onnx_model)
        self.workload = self.parse_workload()

    def get_parser_class(self, node: NodeProto):
        if node.op_type in ONNXModelParser.FUSION_EDGE_OPS:
            return FusionEdgeParser
        parser_class = onnx_parser_for(node.op_type)
        if not parser_class:
            raise NotImplementedError(f"No parser registered for ONNX op type '{node.op_type}'.")
        return parser_class

    def parse_workload(self):
        """Convert the ONNX model into a ``Workload`` graph."""
        assert self.onnx_model is not None

        nodes_outputs: dict[int, Any] = {}

        unnamed_id = 0
        name_to_tensor_dict: dict[str, Tensor] = {}
        workload_nodes: list[Node] = []

        # Add InEdges, for the graph inputs a node reads
        read = {name for node in self.onnx_model.graph.node for name in node.input}
        for input in self.onnx_model.graph.input:
            if input.name not in read:
                continue
            tensor = onnx_tensor_to_tensor(input)
            workload_nodes.append(InEdge(name=input.name, outputs=(tensor,)))
            name_to_tensor_dict[input.name] = tensor
        # Shapes and axes are int64 tensors: an int32 initializer is data, such as a quantized bias
        for initializer in self.onnx_model.graph.initializer:
            if initializer.data_type == TensorProto.INT64 or initializer.name not in read:
                continue
            tensor = onnx_tensor_to_tensor(initializer)
            workload_nodes.append(InEdge(name=initializer.name, outputs=(tensor,)))
            name_to_tensor_dict[initializer.name] = tensor

        # Add ComputationNodes
        for node in self.onnx_model.graph.node:
            # If this node has no inputs, don't take it into consideration (e.g. Constant operator has no inputs)
            if not node.input:
                raise NotImplementedError()

            if not node.name:
                # Generate a unique name for an unnamed node.
                node.name = f"Op{unnamed_id}"
                unnamed_id += 1

            parser_class = self.get_parser_class(node)
            parser = parser_class(
                node=node,
                nodes_outputs=nodes_outputs,
                onnx_model=self.onnx_model,
            )

            logger.info("Parsed %s node %s.", node.op_type, node.name)
            for node_obj in parser.run(name_to_tensor_dict):
                for output in node_obj.outputs:
                    name_to_tensor_dict[output.name] = output
                workload_nodes.append(node_obj)

        # Drop the InEdges no node reads as a tensor, such as an index or a bias its parser takes from the graph
        consumed = {tensor.name for node in workload_nodes for tensor in getattr(node, "inputs", ())}
        workload_nodes = [n for n in workload_nodes if not isinstance(n, InEdge) or n.outputs[0].name in consumed]

        # Add OutEdge
        workload_nodes.append(
            OutEdge(
                name=self.onnx_model.graph.output[0].name,
                inputs=(name_to_tensor_dict[self.onnx_model.graph.output[0].name],),
            )
        )

        workload = Workload(workload_nodes)
        logger.info(
            "Created ONNXWorkload graph with %i nodes and %i edges.",
            workload.number_of_nodes(),
            workload.number_of_edges(),  # type: ignore
        )
        return workload


def fold_quantize_dequantize(model: onnx.ModelProto) -> None:
    """Fold QuantizeLinear and DequantizeLinear into the nodes next to them, as the backends running a quantized
    (QDQ) model do: they mark the element type a tensor has, not a computation. A DequantizeLinear's readers read its
    quantized input instead, and a QuantizeLinear's output is written by the one node producing its input, in place
    of that input, when nothing else reads it: the requantization a node does as it writes its result. One that
    cannot fold, quantizing a graph input or a layout-only node's output, dequantizing a graph output or a tensor
    others read too, stays a cast,
    its scale and zero point part of the conversion rather than tensors it reads."""
    graph = model.graph
    outputs = {output.name for output in graph.output}
    for node in [n for n in graph.node if n.op_type == "DequantizeLinear" and n.output[0] not in outputs]:
        for reader in graph.node:
            for i, name in enumerate(reader.input):
                if name == node.output[0]:
                    reader.input[i] = node.input[0]
        graph.node.remove(node)
    for node in [n for n in graph.node if n.op_type == "QuantizeLinear"]:
        source = node.input[0]
        producers = [n for n in graph.node if source in n.output]
        readers = [n for n in graph.node if source in n.input]
        if (
            source in outputs
            or len(producers) != 1
            or readers != [node]
            or producers[0].op_type in ONNXModelParser.FUSION_EDGE_OPS
        ):
            continue
        producer = producers[0]
        producer.output[list(producer.output).index(source)] = node.output[0]
        graph.node.remove(node)
    for node in graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            del node.input[1:]
