from typing import Any

from onnx import ModelProto, NodeProto
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.io.onnx.flatten import FlattenParser
from stream.classes.io.onnx.pooling import PoolingParser
from stream.classes.io.onnx.reshape import ReshapeParser
from stream.classes.io.onnx.simd import SimdParser
from stream.classes.io.onnx.transpose import TransposeParser
from stream.classes.io.onnx.lpnormalization import LpNormalizationParser
from stream.classes.io.onnx.default import DefaultNodeParser
from zigzag.parser.onnx.utils import parse_onnx_model_from_path, get_onnx_tensor_type
from zigzag.stages.WorkloadParserStage import WorkloadParserStage
from stream.classes.io.onnx.gemm import GemmParser
from stream.classes.io.onnx.matmul import MatMulParser
from stream.classes.io.onnx.conv import ConvParser
from stream.classes.workload.onnx_workload import ONNXWorkload


import logging


logger = logging.getLogger(__name__)


class ONNXModelParser:
    """Parse the ONNX model into a workload."""

    def __init__(self, onnx_model_path: str, mapping_yaml_path: str, accelerator: Accelerator) -> None:
        self.onnx_model_path = onnx_model_path
        self.mapping_yaml_path_data = mapping_yaml_path
        self.accelerator = accelerator

        self.onnx_model = None
        self.workload = None
        self.mapping_data = None

    def run(self):
        """Run the parser:
        - parse the onnx_model_path into an onnx model
        - parse the mapping_path into a mapping dict
        - iterate through the onnx model and generate the workload consisting of LayerNodes and DummyNodes
        """
        self.onnx_model = parse_onnx_model_from_path(self.onnx_model_path)
        self.mapping_data = WorkloadParserStage.parse_mapping_data(self.mapping_yaml_path_data)
        self.workload = self.parse_workload_from_onnx_model_and_mapping()

    def parse_workload_from_onnx_model_and_mapping(self):
        """
        Converts an onnx model into a workload object.
        We scan the model for all convolutional layers, and setup a Layer object for each of those using the mapping.
        Then we combine the layers into a workload graph.
        """
        assert self.mapping_data is not None
        assert self.onnx_model is not None
        # If the model isn't in the format with external data, it will be slow to manipulate it, so better to work with
        # raw models with external data # The line below accomplishes this.
        # onnx.save_model(model, 'model_external.onnx', save_as_external_data=True, all_tensors_to_one_file=True,
        # location='model_external_raw_data', size_threshold=1024, convert_attribute=False)

        # In the future, assume we will have a model saved with external data, then we have to execute the code below
        # if the model isn't inferred yet
        # This approach is faster for large models because the raw model is used (w/o the external data)
        # if model is not inferred:
        #   onnx.shape_inference.infer_shapes_path('path/to/the/model.onnx')  # This will save the inferred model to the
        # same file
        #   model = onnx.load('path/to/the/model.onnx')  # reload the inferred model

        # Saves for each node_id the inputs and outputs tensor names
        nodes_inputs: dict[int, Any] = {}
        nodes_outputs: dict[int, Any] = {}

        # Workload Graph
        workload = ONNXWorkload()

        for node_id, node in enumerate(self.onnx_model.graph.node):
            node_id_tuple = (node_id,)
            # If this node has no inputs, don't take it into consideration (e.g. Constant operator has no inputs)
            if not node.input:
                continue
            nodes_inputs[node_id] = node.input
            nodes_outputs[node_id] = node.output

            if node.op_type in ["QLinearConv", "Conv"]:
                parser = ConvParser(
                    node_id=node_id,
                    node=node,
                    nodes_outputs=nodes_outputs,
                    mapping_data=self.mapping_data,
                    onnx_model=self.onnx_model,
                    accelerator=self.accelerator,
                )
                logger.info("Parsed Conv node %s.", node.name)
            elif node.op_type in ["MatMul"]:
                parser = MatMulParser(
                    node_id=node_id,
                    node=node,
                    nodes_outputs=nodes_outputs,
                    mapping_data=self.mapping_data,
                    onnx_model=self.onnx_model,
                    accelerator=self.accelerator,
                )
                logger.info("Parsed MatMul node %s.", node.name)
            elif node.op_type in ["Gemm"]:
                parser = GemmParser(
                    node_id=node_id,
                    node=node,
                    nodes_outputs=nodes_outputs,
                    mapping_data=self.mapping_data,
                    onnx_model=self.onnx_model,
                    accelerator=self.accelerator,
                )
                logger.info("Parsed Gemm node %s.", node.name)
            elif node.op_type in [
                "MaxPool",
                "AveragePool",
                "GlobalMaxPool",
                "GlobalAveragePool",
            ]:
                parser = PoolingParser(
                    node_id=node_id,
                    node=node,
                    nodes_outputs=nodes_outputs,
                    mapping_data=self.mapping_data,
                    onnx_model=self.onnx_model,
                    accelerator=self.accelerator,
                )
                logger.info("Parsed Pooling node %s.", node.name)
            elif node.op_type in ["Reshape"]:
                parser = ReshapeParser(
                    node_id=node_id,
                    node=node,
                    nodes_outputs=nodes_outputs,
                    onnx_model=self.onnx_model,
                )
            elif node.op_type in ["Flatten"]:
                parser = FlattenParser(
                    node_id=node_id,
                    node=node,
                    nodes_outputs=nodes_outputs,
                    onnx_model=self.onnx_model,
                )
                logger.info("Parsed Flatten node %s.", node.name)
            elif node.op_type in ["Add", "Mul"]:
                # TODO: a temporary fix an element-wise Add or Mul which has asymmetric input data -> treat it as a DummyNode.
                #  Future to support node with asymmetric input data.
                if has_asymmetric_input_data(node, self.onnx_model):
                    parser = DefaultNodeParser(
                        node_id=node_id,
                        node=node,
                        nodes_outputs=nodes_outputs,
                        onnx_model=self.onnx_model,
                    )
                    logger.info(
                        "Parsed asymmetric %s node %s as a DummyNode",
                        node.op_type,
                        node.name,
                    )
                else:
                    parser = SimdParser(
                        node_id=node_id,
                        node=node,
                        nodes_outputs=nodes_outputs,
                        mapping_data=self.mapping_data,
                        onnx_model=self.onnx_model,
                        accelerator=self.accelerator,
                    )
                    logger.info(
                        "Parsed %s node %s.",
                        node.op_type,
                        node.name,
                    )
            elif node.op_type in ["Transpose"]:
                parser = TransposeParser(node_id, node, nodes_outputs, self.mapping_data, self.onnx_model)
                logger.info("Parsed Transpose node %s.", node.name)
            elif node.op_type in ["LpNormalization"]:
                parser = LpNormalizationParser(node_id, node, nodes_outputs, self.mapping_data, self.onnx_model)
                logger.info("Parsed LpNormalization node %s.", node.name)
            # it is not any of the above, so create a DummyNode
            else:
                parser = DefaultNodeParser(node_id, node, nodes_outputs, self.onnx_model)
                logger.info(
                    "Parsed %s node %s as a DummyNode",
                    node.op_type,
                    node.name,
                )
            node_obj = parser.run()
            # Add the node_obj to the ONNXWorkload
            workload.add(node_id, node_obj)

        logger.info(
            "Created ONNXWorkload graph with %i  nodes and %i  edges.",
            workload.number_of_nodes(),
            workload.number_of_edges(),
        )

        return workload

    def get_onnx_model(self):
        return self.onnx_model

    def get_mapping(self):
        return self.mapping_data

    def get_workload(self):
        return self.workload


def has_asymmetric_input_data(node: NodeProto, onnx_model: ModelProto):

    input_name1 = node.input[0]
    input_name2 = node.input[1]
    input_shape1 = get_onnx_tensor_type(input_name1, onnx_model).shape
    input_shape2 = get_onnx_tensor_type(input_name2, onnx_model).shape
    return input_shape1 != input_shape2
