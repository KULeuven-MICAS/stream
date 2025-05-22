from typing import Any

from zigzag.datatypes import Constants

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.parser.onnx.reduce_1d import Reduce1DParser

from zigzag.parser.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.parser.workload_factory import LayerNodeFactory
from stream.workload.computation.computation_node import ComputationNode

class ReduceSumParser(OnnxComputeOperatorParser):
    # For now reduce over 4 axis to one

    NODE_TYPES = ["sum", "sum", "sum"]
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
    
    def parse_into_subnodes(self) :
        """Prase the base ONNX node multiple times into the different Computation Nodes.
        The CNs that result from this operation have some incorrect properties regarding the graph structure
        """
        # parser_classes: list[type] = [Reduce1DParser, SoftmaxExpParser, Reduce1DParser, SoftmaxDivParser, SoftmaxCrossEntropySIMDParser, Reduce1DParser, SoftmaxCrossEntropyReduceParser]
        parser_classes: list[type] = [Reduce4D3DParser, Reduce3D2DParser, Reduce2D1DParser]
        node_ids = [self.node_id + i for i in range(3)]
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

    def set_nodes_name_and_type(self) :
        """Set the name and operator type of all Computation Nodes that stem from the base ONNX node"""
        for node, node_type in zip(self.nodes, ReduceSumParser.NODE_TYPES):
            node.type = node_type
            node.name += f"-{node_type}/"
    def correct_nodes_operand_source(self) :
        """Correct the `input_operand_source` and `constant_operands` of all Computation Nodes that stem from the base
        ONNX node"""
        op_I = Constants.LAYER_OP_I
        op_W = Constants.LAYER_OP_W
        node_sum1, node_sum2, node_sum3 = self.nodes
        id_sum1, id_sum2, id_sum3 = [node.id for node in self.nodes]
        prev_node_id = node_sum1.input_operand_source[op_I]  # Node before Softmax

        # Default after generation: input_operand_source = {op_I: prev_node_id} and constant_operands = [W]
        node_sum1.input_operand_source = {op_I:prev_node_id}
        node_sum2.input_operand_source = {op_I:id_sum1}
        node_sum3.input_operand_source = {op_I:id_sum2}
        
class Reduce4D3DParser(Reduce1DParser) :
    def generate_node(self):
        # Get the input and output activation shapes
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)
        reduced_input_shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]]

        # From the ONNX node
        node_data = self.get_layer_node_user_format(reduced_input_shape, output_shape)
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

class Reduce3D2DParser(Reduce1DParser) :
    def generate_node(self):
        # Get the input and output activation shapes
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)
        reduced_input_shape = [input_shape[0], input_shape[1], input_shape[2]]

        # From the ONNX node
        node_data = self.get_layer_node_user_format(reduced_input_shape, output_shape)
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

class Reduce2D1DParser(Reduce1DParser) :
    def generate_node(self):
        # Get the input and output activation shapes
        input_shape, output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)
        reduced_input_shape = [input_shape[0], input_shape[1]]

        # From the ONNX node
        node_data = self.get_layer_node_user_format(reduced_input_shape, output_shape)
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
