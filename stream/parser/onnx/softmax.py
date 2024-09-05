from typing import Any

from zigzag.datatypes import Constants

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.parser.onnx.reduce_1d import Reduce1DParser


class SoftmaxParser(OnnxComputeOperatorParser):
    """Parses the Softmax operator. Softmax works on full rows and can be computed as follows:
    (1) m <- max(row[0:L])
    (2) e[0:L] <- exp(row[0:L] - m)
    (3) s <- sum(e[0:L])
    (4) r[0:L] <- e[0:L] / s
    It is split up in four distinct computation nodes.
    """

    NODE_TYPES = ["max", "exp", "sum", "div"]

    def run(self):
        for node in self.get_nodes():
            yield node

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]) -> dict[str, Any]:
        """Not used for this class, but abstract base class requires instantiation anyway"""
        ...

    def parse_into_subnodes(self):
        """Prase the base ONNX node multiple times into the different Computation Nodes.
        The CNs that result from this operation have some incorrect properties regarding the graph structure
        """
        parser_classes = [Reduce1DParser, SoftmaxExpParser, Reduce1DParser, SoftmaxDivParser]

        node_ids = [self.node_id + i for i in range(4)]
        parsers = [
            parser(
                node_id=node_id,
                node=self.node,
                nodes_outputs=self.nodes_outputs,  # TODO now, the node_outputs does not contain the current id
                onnx_model=self.onnx_model,
                mapping_data=self.mapping_data,
                accelerator=self.accelerator,
            )
            for parser, node_id in zip(parser_classes, node_ids)
        ]
        self.nodes = tuple(next(parser.run()) for parser in parsers)

    def get_nodes(self):
        # Parse initial CNs
        self.parse_into_subnodes()
        # Give correct op type and name
        self.set_nodes_name_and_type()
        # Override dependencies
        self.correct_nodes_operand_source()
        # self.correct_nodes_inputs_outputs()

        return self.nodes

    def set_nodes_name_and_type(self):
        """Set the name and operator type of all Computation Nodes that stem from the base ONNX node"""
        for node, node_type in zip(self.nodes, SoftmaxParser.NODE_TYPES):
            node.type = node_type
            node.name += f"-{node_type}/"

    def correct_nodes_operand_source(self):
        """Correct the `input_operand_source` and `constant_operands` of all Computation Nodes that stem from the base
        ONNX node"""
        op_I = Constants.LAYER_OP_I
        op_W = Constants.LAYER_OP_W
        node_max, node_exp, node_sum, node_div = self.nodes
        id_max, id_exp, id_sum, _ = [node.id for node in self.nodes]
        prev_node_id = node_max.input_operand_source[op_I]  # Node before Softmax

        # Default after generation: input_operand_source = {op_I: prev_node_id} and constant_operands = [W]
        node_exp.input_operand_source = {op_I: prev_node_id, op_W: id_max}
        node_exp.constant_operands = []
        node_sum.input_operand_source = {op_I: id_exp}
        node_div.input_operand_source = {op_I: id_exp, op_W: id_sum}
        node_div.constant_operands = []

    # def correct_nodes_inputs_outputs(self):
    #     """Correct the `node_inputs` and `node_outputs` of all Computation Nodes that stem from the base
    #     ONNX node"""
    #     node_max, node_exp, node_sum, node_div = self.nodes
    #     prev_node_name = node_max.input_names[0]  # Node before Softmax
    #     next_node_name = node_max.output_names[0]  # Node after Softmax

    #     node_max.output_names = [node_exp.name]
    #     node_exp.input_names = [node_max.name, prev_node_name]
    #     node_exp.output_names = [node_div.name, node_sum.name]
    #     node_sum.input_names = [node_exp.name]
    #     node_sum.output_names = [node_div.name]
    #     node_div.input_names = [node_exp.name, node_sum.name]
    #     node_div.output_names = [next_node_name]


class SoftmaxExpParser(OnnxComputeOperatorParser):
    """Parses a softmax node into a ComputationNode for the element-wise operation exp(row-m) where m is the max value
    of the row.
    """

    def get_layer_node_user_format(self, input_shape: list[int], output_shape: list[int]):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_input_format()
        data["dimension_relations"] = []
        data["loop_sizes"] = input_shape

        # C is the row dimension, so W should not have this as there is only 1 max value for each row
        match len(input_shape):
            case 2:
                data["equation"] = "O[k][c]+=I[k][c]+W[k]"
                data["loop_dims"] = ["K", "C"]
            case 3:
                data["equation"] = "O[b][k][c]+=I[b][k][c]+W[b][k]"
                data["loop_dims"] = ["B", "K", "C"]
            case 4:
                data["equation"] = "O[b][h][k][c]+=I[b][h][k][c]+W[b][h][k]"
                data["loop_dims"] = ["B", "H", "K", "C"]
            case _:
                raise NotImplementedError

        return data


class SoftmaxDivParser(SoftmaxExpParser):
    """Parses a softmax node into a ComputationNode for the element-wise operation div(row, s) where s is the sum value
    of the row.
    The equation is identical to the one from SoftmaxExpParser
    """
