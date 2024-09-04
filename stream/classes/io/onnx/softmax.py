from typing import Any

from zigzag.datatypes import Constants

from stream.classes.io.onnx.operator_parser import OnnxComputeOperatorParser
from stream.classes.io.onnx.reduce_1d import Reduce1DParser


class SoftmaxParser(OnnxComputeOperatorParser):
    """Parses the Softmax operator. Softmax works on full rows and can be computed as follows:
    (1) m <- max(row[0:L])
    (2) e[0:L] <- exp(row[0:L] - m)
    (3) s <- sum(e[0:L])
    (4) r[0:L] <- e[0:L] / s
    It is split up in four distinct computation nodes.
    """

    def run(self):
        for node in self.get_nodes():
            yield node

    def get_nodes(self):
        node_types = ["max", "exp", "sum", "div"]
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
        nodes = [next(parser.run()) for parser in parsers]

        # Give correct op type and name
        for node, node_type in zip(nodes, node_types):
            node.type = node_type
            node.name += f"-{node_type}"

        # Override dependencies
        op_I = Constants.LAYER_OP_I
        op_W = Constants.LAYER_OP_W
        id_max, id_exp, id_sum, _ = node_ids
        prev_node_id = nodes[0].input_operand_source[op_I]  # Node before max
        nodes[1].input_operand_source = {op_I: prev_node_id, op_W: id_max}  # Exp
        nodes[2].input_operand_source = {op_I: id_exp, op_W: id_sum}  # Sum
        nodes[3].input_operand_source = {op_I: id_exp, op_W: id_sum}  # Div

        return nodes


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
                data["loop_dims"] = ["B", "C", "k"]
            case 4:
                data["equation"] = "O[b][h][k][c]+=I[b][h][k][c]+W[b][h][k]"
                data["loop_dims"] = ["B", "H", "C", "k"]
            case _:
                raise NotImplementedError

        return data


class SoftmaxDivParser(SoftmaxExpParser):
    """Parses a softmax node into a ComputationNode for the element-wise operation div(row, s) where s is the sum value
    of the row.
    The equation is identical to the one from SoftmaxExpParser
    """
