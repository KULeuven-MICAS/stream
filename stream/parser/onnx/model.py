import logging
from typing import Any

from onnx import NodeProto
from zigzag.parser.onnx.utils import parse_onnx_model_from_path

# from stream.hardware.architecture.accelerator import Accelerator
# from stream.parser.onnx.concat import ConcatParser
# from stream.parser.onnx.conv import ConvParser
# from stream.parser.onnx.default import DefaultNodeParser
# from stream.parser.onnx.einsum import EinsumParser
# from stream.parser.onnx.exp import ExpParser
# from stream.parser.onnx.flatten import FlattenParser
# from stream.parser.onnx.gather import GatherParser
# from stream.parser.onnx.gelu import GeluParser
from stream.parser.onnx.gemm import GemmParser

# from stream.parser.onnx.lpnormalization import LpNormalizationParser
# from stream.parser.onnx.matmul import MatMulParser
from stream.parser.onnx.mul import MulParser
from stream.parser.onnx.operator_parser import OnnxOperatorParser

# from stream.parser.onnx.pooling import PoolingParser
# from stream.parser.onnx.reciprocal import ReciprocalParser
# from stream.parser.onnx.reduce_1d import Reduce1DParser
# from stream.parser.onnx.relu import ReluParser
# from stream.parser.onnx.reshape import ReshapeParser
# from stream.parser.onnx.sigmoid import SigmoidParser
from stream.parser.onnx.simd import SimdParser

# from stream.parser.onnx.slice import SliceParser
# from stream.parser.onnx.softmax import SoftmaxParser
# from stream.parser.onnx.split import SplitParser
# from stream.parser.onnx.sqrt import SqrtParser
# from stream.parser.onnx.ssm import SSMParser
# from stream.parser.onnx.transpose import TransposeParser
from stream.parser.onnx.utils import onnx_tensor_to_tensor
from stream.workload.workload import HasOutput, InEdge, Node, OutEdge, Workload

logger = logging.getLogger(__name__)


class ONNXModelParser:
    """Parse the ONNX model into a workload."""

    # Map the node's op_type to the corresponding Parser class
    OP_TYPE_TO_PARSER: dict[str, type[OnnxOperatorParser]] = {
        # General
        # "QLinearConv": ConvParser,
        # "Conv": ConvParser,
        # "MatMul": MatMulParser,
        "Gemm": GemmParser,
        # "Einsum": EinsumParser,
        # "MaxPool": PoolingParser,
        # "AveragePool": PoolingParser,
        # "GlobalMaxPool": PoolingParser,
        # "GlobalAveragePool": PoolingParser,
        # "Add": MulParser,
        "Mul": MulParser,
        # Special operators
        # "SSM": SSMParser,
        # "Softmax": SoftmaxParser,
        # Single-input element-wise
        # "Exp": ExpParser,
        # "ReduceMean": Reduce1DParser,
        # "Relu": ReluParser,
        # "Gelu": GeluParser,
        "Silu": SimdParser,
        # "Sigmoid": SigmoidParser,
        # "Sqrt": SqrtParser,
        # "Div": SimdParser,
        # "Pow": SimdParser,
        # "Reciprocal": ReciprocalParser,  # Div with 1 as numerator
        # # Dependency propagation
        # "LpNormalization": LpNormalizationParser,
        # "Gather": GatherParser,
        # "Transpose": TransposeParser,
        # "Reshape": ReshapeParser,
        # "Flatten": FlattenParser,
        # "Concat": ConcatParser,
        # "Split": SplitParser,
        # "Slice": SliceParser,
    }

    def __init__(self, onnx_model_path: str) -> None:
        self.onnx_model_path = onnx_model_path

    def run(self):
        """Run the parser:
        - parse the onnx_model_path into an onnx model
        - parse the mapping_path into a mapping dict
        - iterate through the onnx model and generate the workload consisting of LayerNodes and DummyNodes
        """
        self.onnx_model = parse_onnx_model_from_path(self.onnx_model_path)
        self.workload = self.parse_workload()

    def get_parser_class(self, node: NodeProto):
        parser_class = ONNXModelParser.OP_TYPE_TO_PARSER.get(node.op_type)
        if not parser_class:
            raise NotImplementedError()
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

        unnamed_id = 0
        # Workload Graph
        name_to_node_dict: dict[str, HasOutput] = {}
        workload_nodes: list[Node] = []

        # Add InEdges
        for input in self.onnx_model.graph.input:
            workload_nodes.append(in_edge := InEdge(input.name, onnx_tensor_to_tensor(input)))
            name_to_node_dict[input.name] = in_edge
        for initializer in self.onnx_model.graph.initializer:
            workload_nodes.append(in_edge := InEdge(initializer.name, onnx_tensor_to_tensor(initializer)))
            name_to_node_dict[initializer.name] = in_edge

        # Add ComputationNodes
        for node in self.onnx_model.graph.node:
            # If this node has no inputs, don't take it into consideration (e.g. Constant operator has no inputs)
            if not node.input:
                raise NotImplementedError()

            if not node.name:
                # If the node has no name, we generate a unique name based on the node id.
                # This is useful for debugging and visualization purposes.
                node.name = f"Op{unnamed_id}"
                unnamed_id += 1

            parser_class = self.get_parser_class(node)
            parser = parser_class(
                node=node,
                nodes_outputs=nodes_outputs,
                onnx_model=self.onnx_model,
            )

            logger.info("Parsed %s node %s.", node.op_type, node.name)
            for node_obj in parser.run(name_to_node_dict):
                if len(node.output) == 1:
                    assert isinstance(node_obj, HasOutput)
                    name_to_node_dict[node.output[0]] = node_obj
                # Parsers that yield multiple nodes increment the node id internally, so we must keep count here.
                workload_nodes.append(node_obj)

        # Add OutEdge
        workload_nodes.append(
            OutEdge(
                self.onnx_model.graph.output[0].name,
                (name_to_node_dict[self.onnx_model.graph.output[0].name],),
            )
        )

        workload = Workload(workload_nodes)
        logger.info(
            "Created ONNXWorkload graph with %i nodes and %i edges.",
            workload.number_of_nodes(),
            workload.number_of_edges(),  # type: ignore
        )
        return workload
