from typing import Any

from zigzag.datatypes import Constants

from stream.workload.computation.computation_node import ComputationNode
from zigzag.parser.workload_factory import LayerNodeFactory
from zigzag.parser.onnx.utils import get_onnx_tensor_type
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.parser.onnx.softmax import SoftmaxExpParser, SoftmaxDivParser
from stream.parser.onnx.simd import SimdParser
from stream.parser.onnx.reduce_1d import Reduce1DParser

class SoftmaxCrossEntropyParser(OnnxComputeOperatorParser):
    """
    Parses the SoftmaxCrossEntropy Operator
    attributes :
        - ignore_index:int
        - reduction:string (default mean), "none", "sum", "mean"
    inputs: 
        - scores
        - labels
        - weights (optional)
    outputs :
        - output
        - log_prob(optional)
    """

    """Parses the Softmax operator. Softmax works on full rows and can be computed as follows:
    (1) m <- max(row[0:L])
    (2) e[0:L] <- exp(row[0:L] - m)
    (3) s <- sum(e[0:L])
    (4) r[0:L] <- e[0:L] / s
    It is split up in four distinct computation nodes.
    """
    # NODE_TYPES = ["max", "exp", "sum", "div", "log", "sum", "reduce"]
    NODE_TYPES = ["max", "exp", "sum", "div", "log"]

    def run(self):
        for node in self.get_nodes():
            yield node

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]) -> dict[str, Any]:
        """Not used for this class, but abstract base class requires instantiation anyway"""
        ...

    def get_nodes(self):
        # Parse initial CNs
        self.parse_into_subnodes()
        # Give correct op type and name
        self.set_nodes_name_and_type()
        # Override dependencies
        self.correct_nodes_operand_source()

        return self.nodes
    
    def parse_into_subnodes(self):
        """Prase the base ONNX node multiple times into the different Computation Nodes.
        The CNs that result from this operation have some incorrect properties regarding the graph structure
        """
        # parser_classes: list[type] = [Reduce1DParser, SoftmaxExpParser, Reduce1DParser, SoftmaxDivParser, SoftmaxCrossEntropySIMDParser, Reduce1DParser, SoftmaxCrossEntropyReduceParser]
        parser_classes: list[type] = [Reduce1DParser, SoftmaxExpParser, Reduce1DParser, SoftmaxDivParser, SoftmaxCrossEntropySIMDParser]
        node_ids = [self.node_id + i for i in range(7)]
        parsers: list[OnnxComputeOperatorParser] = [
            parser(
                node_id=node_id,
                node=self.node,
                nodes_outputs=self.nodes_outputs,
                onnx_model=self.onnx_model,
                all_mappings=self.all_mappings,
                accelerator=self.accelerator,
            )
            for parser, node_id in zip(parser_classes, node_ids)
        ]
        self.nodes = []
        for parser in parsers :
            for node in parser.run() :
                self.nodes.append(node)
        self.nodes = tuple(self.nodes)

    def set_nodes_name_and_type(self):
        """Set the name and operator type of all Computation Nodes that stem from the base ONNX node"""
        for node, node_type in zip(self.nodes, SoftmaxCrossEntropyParser.NODE_TYPES):
            node.type = node_type
            node.name += f"-{node_type}/"

    def correct_nodes_operand_source(self):
        """Correct the `input_operand_source` and `constant_operands` of all Computation Nodes that stem from the base
        ONNX node"""
        op_I = Constants.LAYER_OP_I
        op_W = Constants.LAYER_OP_W
        node_sfm_max, node_sfm_exp, node_sfm_sum, node_sfm_div, node_log = self.nodes
        id_sfm_max, id_sfm_exp, id_sfm_sum, id_sfm_div, id_log = [node.id for node in self.nodes]
        prev_node_id = node_sfm_max.input_operand_source[op_I]  # Node before Softmax

        # Default after generation: input_operand_source = {op_I: prev_node_id} and constant_operands = [W]
        node_sfm_exp.input_operand_source = {op_I: prev_node_id, op_W: id_sfm_max}
        node_sfm_exp.constant_operands = []
        node_sfm_sum.input_operand_source = {op_I: id_sfm_exp}
        node_sfm_div.input_operand_source = {op_I: id_sfm_exp, op_W: id_sfm_sum}
        node_sfm_div.constant_operands = []

        # TODO: fix constant operands for added nodes
        node_log.input_operand_source = {op_I: id_sfm_div}
        node_log.constant_operands = []
        
class SoftmaxCrossEntropySIMDParser(SimdParser) :
    def generate_node(self):
        # Get the input and output activation shapes
        scores_shape, labels_shape, output_shape, logprob_shape = softmaxcrossentropy_get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # From the ONNX node
        node_data = self.get_layer_node_user_format(scores_shape, scores_shape)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()

        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type=self.node.op_type,
            node_attr=node_attrs,
            mapping_attr=mapping,
        )

class SoftmaxCrossEntropyReduceParser(Reduce1DParser) :
    def generate_node(self):
        # Get the input and output activation shapes
        scores_shape, labels_shape, output_shape, logprob_shape = softmaxcrossentropy_get_node_input_output_dimension_shapes(self.node, self.onnx_model)

        # From the ONNX node
        node_data = self.get_layer_node_user_format(scores_shape, scores_shape)
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()

        mapping = self.get_mapping_this_node()

        return ComputationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            op_type=self.node.op_type,
            node_attr=node_attrs,
            mapping_attr=mapping,
        )

def softmaxcrossentropy_get_node_input_output_dimension_shapes(node, model) :
    # assumed it is the first input, don't see a way to otherwise know

    scores_name = node.input[0]
    scores_shape = get_onnx_tensor_type(scores_name, model).shape
    labels_name = node.input[1]
    labels_shape = get_onnx_tensor_type(labels_name, model).shape

    output_name = node.output[0]
    output_shape = get_onnx_tensor_type(output_name, model).shape
    logprob_name = node.output[1]
    logprob_shape = get_onnx_tensor_type(logprob_name, model).shape
    return scores_shape, labels_shape, output_shape, logprob_shape
