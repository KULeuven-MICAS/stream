from typing import Any

from zigzag.datatypes import Constants

from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.parser.onnx.reduce_1d import Reduce1DParser
from stream.parser.onnx.simd import SimdParser
from stream.workload.mapping import InterCoreMappingAttributes


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
        yield from self.get_nodes()

    def get_layer_node_user_format(
        self, input_shape: list[int], output_shape: list[int], mapping: InterCoreMappingAttributes | None
    ) -> dict[str, Any]:
        """Not used for this class, but abstract base class requires instantiation anyway"""
        ...

    def parse_into_subnodes(self):
        """Prase the base ONNX node multiple times into the different Computation Nodes.
        The CNs that result from this operation have some incorrect properties regarding the graph structure
        """
        parser_classes: list[type] = [Reduce1DParser, SoftmaxExpParser, Reduce1DParser, SoftmaxDivParser]

        node_ids = [self.node_id + i for i in range(4)]
        parsers: list[OnnxComputeOperatorParser] = [
            parser(
                node_id=node_id,
                node=self.node,
                nodes_outputs=self.nodes_outputs,
                onnx_model=self.onnx_model,
                all_mappings=self.all_mappings,
                accelerator=self.accelerator,
            )
            for parser, node_id in zip(parser_classes, node_ids, strict=False)
        ]
        self.nodes = tuple(next(parser.run()) for parser in parsers)

    def get_nodes(self):
        # Parse initial CNs
        self.parse_into_subnodes()
        # Give correct op type and name
        self.set_nodes_name_and_type()
        # Override dependencies
        self.correct_nodes_operand_source()

        return self.nodes

    def set_nodes_name_and_type(self):
        """Set the name and operator type of all Computation Nodes that stem from the base ONNX node"""
        for node, node_type in zip(self.nodes, SoftmaxParser.NODE_TYPES, strict=False):
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


class SoftmaxExpParser(SimdParser):
    """Parses a softmax node into a ComputationNode for the element-wise operation exp(row-m) where m is the max value
    of the row.
    The 'weight' input always has one rank less than the activations input
    """

    DEFAULT_LAYER_DIMENSIONS = ["B", "H", "D", "K"]

    def get_layer_node_user_format(
        self, input_shape: list[int], output_shape: list[int], mapping: InterCoreMappingAttributes
    ):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """

        data: dict[str, Any] = {}
        data["id"] = self.node_id
        data["name"] = self.node.name
        data["operator_type"] = self.node.op_type
        data["operand_source"] = self.get_operand_source_input_format()
        data["operand_precision"] = self.get_operand_precision_user_format()
        data["dimension_relations"] = []

        MINIMUM_INPUT_LENGTH = 1
        if len(input_shape) > len(SimdParser.DEFAULT_LAYER_DIMENSIONS) or len(input_shape) <= MINIMUM_INPUT_LENGTH:
            raise NotImplementedError

        possible_loop_dims = (
            mapping.layer_dimension_names
            if len(mapping.layer_dimension_names) == len(output_shape)
            else SimdParser.DEFAULT_LAYER_DIMENSIONS
        )

        loop_dims = possible_loop_dims[0 : len(output_shape)]

        # The output of MaxParser will have a arbitrary dim of size 1
        reduced_dim_output = "R"  # C reduced to 1
        assert reduced_dim_output not in possible_loop_dims, "Layer dimension `R` is reserved for the reduction axis"

        # W should have one dimension less because the same value is used for a ful row of I
        loop_dims_W = loop_dims[:-1] + [reduced_dim_output]

        equation_dims = "".join([f"[{dim.lower()}]" for dim in loop_dims])
        equation_dims_W = "".join([f"[{dim.lower()}]" for dim in loop_dims_W])

        equation = f"O{equation_dims}+=I{equation_dims}*W{equation_dims_W}"

        data["equation"] = equation
        data["loop_dims"] = loop_dims + [reduced_dim_output]
        data["loop_sizes"] = input_shape + [1]

        return data


class SoftmaxDivParser(SoftmaxExpParser):
    """Parses a softmax node into a ComputationNode for the element-wise operation div(row, s) where s is the sum value
    of the row.
    The equation is identical to the one from SoftmaxExpParser
    """
