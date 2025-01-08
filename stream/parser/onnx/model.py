import logging
from typing import Any, Type

from onnx import NodeProto
from zigzag.parser.onnx.utils import parse_onnx_model_from_path

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.onnx.asymmetric_simd import AsymmetricSimdParser
from stream.parser.onnx.concat import ConcatParser
from stream.parser.onnx.conv import ConvParser
from stream.parser.onnx.default import DefaultNodeParser
from stream.parser.onnx.flatten import FlattenParser
from stream.parser.onnx.gather import GatherParser
from stream.parser.onnx.gemm import GemmParser
from stream.parser.onnx.lpnormalization import LpNormalizationParser
from stream.parser.onnx.matmul import MatMulParser
from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.parser.onnx.pooling import PoolingParser
from stream.parser.onnx.reshape import ReshapeParser
from stream.parser.onnx.simd import SimdParser
from stream.parser.onnx.softmax import SoftmaxParser
from stream.parser.onnx.transpose import TransposeParser
from stream.utils import get_onnx_input_shapes, has_asymmetric_input_data
from stream.workload.mapping import InterCoreMappingAttributes
from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)


class ONNXModelParser:
    """Parse the ONNX model into a workload."""

    # Map the node's op_type to the corresponding Parser class
    OP_TYPE_TO_PARSER: dict[str, Type[OnnxOperatorParser]] = {
        "QLinearConv": ConvParser,
        "Conv": ConvParser,
        "MatMul": MatMulParser,
        "Gemm": GemmParser,
        "MaxPool": PoolingParser,
        "AveragePool": PoolingParser,
        "GlobalMaxPool": PoolingParser,
        "GlobalAveragePool": PoolingParser,
        "Add": SimdParser,
        "Mul": SimdParser,
        "Softmax": SoftmaxParser,
        "Relu": SimdParser,
        "Gelu": SimdParser,
        "Silu": SimdParser,
        "LpNormalization": LpNormalizationParser,
        "Gather": GatherParser,
        "Transpose": TransposeParser,
        "Reshape": ReshapeParser,
        "Flatten": FlattenParser,
        "Concat": ConcatParser,
    }

    def __init__(
        self, onnx_model_path: str, all_mappings: dict[str, InterCoreMappingAttributes], accelerator: Accelerator
    ) -> None:
        self.onnx_model_path = onnx_model_path
        self.all_mappings = all_mappings
        self.accelerator = accelerator

    def run(self):
        """Run the parser:
        - parse the onnx_model_path into an onnx model
        - parse the mapping_path into a mapping dict
        - iterate through the onnx model and generate the workload consisting of LayerNodes and DummyNodes
        """
        self.onnx_model = parse_onnx_model_from_path(self.onnx_model_path)
        self.workload = self.parse_workload()

    def get_parser_class(self, node: NodeProto):
        # A temporary fix an element-wise Add or Mul which has asymmetric input data -> treat it as a  DummyNode.
        # TODO support node with asymmetric input data.
        if node.op_type in ["Add", "Mul"] and has_asymmetric_input_data(node, self.onnx_model):
            in_shape_1, in_shape_2 = get_onnx_input_shapes(node, self.onnx_model)
            # In case only the batch dimension is missing. Other cases are not supported for now
            if abs(len(in_shape_1) - len(in_shape_2)) == 1:
                return AsymmetricSimdParser
            else:
                return DefaultNodeParser

        parser_class = ONNXModelParser.OP_TYPE_TO_PARSER.get(node.op_type)
        if not parser_class:
            return DefaultNodeParser
        return parser_class

    def parse_workload(self):
        """
        Converts an onnx model into a workload object.
        We scan the model for all convolutional layers, and setup a Layer object for each of those using the mapping.
        Then we combine the layers into a workload graph.

        If the model isn't in the format with external data, it will be slow to manipulate it, so better to work with
        raw models with external data # The line below accomplishes this.
        onnx.save_model(model, 'model_external.onnx', save_as_external_data=True, all_tensors_to_one_file=True,
        location='model_external_raw_data', size_threshold=1024, convert_attribute=False)

        In the future, assume we will have a model saved with external data, then we have to execute the code below
        if the model isn't inferred yet
        This approach is faster for large models because the raw model is used (w/o the external data)
        if model is not inferred:
          onnx.shape_inference.infer_shapes_path('path/to/the/model.onnx')  # This will save the inferred model to the
        same file
          model = onnx.load('path/to/the/model.onnx')  # reload the inferred model
        """
        assert self.onnx_model is not None

        # Saves for each node_id the inputs and outputs tensor names
        nodes_inputs: dict[int, Any] = {}
        nodes_outputs: dict[int, Any] = {}

        # Workload Graph
        workload = ONNXWorkload()
        node_id = 0
        for node in self.onnx_model.graph.node:
            # If this node has no inputs, don't take it into consideration (e.g. Constant operator has no inputs)
            if not node.input:
                continue

            nodes_inputs[node_id] = node.input

            parser_class = self.get_parser_class(node)
            parser = parser_class(
                node_id=node_id,
                node=node,
                nodes_outputs=nodes_outputs,
                onnx_model=self.onnx_model,
                all_mappings=self.all_mappings,
                accelerator=self.accelerator,
            )

            logger.info("Parsed %s node %s.", node.op_type, node.name)
            for node_obj in parser.run():
                # Parsers that yield multiple nodes increment the node id internally, so we must keep count here.
                workload.add(node_id, node_obj)
                node_id += 1

            nodes_outputs[node_id - 1] = node.output

        logger.info(
            "Created ONNXWorkload graph with %i nodes and %i edges.",
            workload.number_of_nodes(),
            workload.number_of_edges(),  # type: ignore
        )

        return workload
