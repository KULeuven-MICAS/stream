import logging
from enum import StrEnum
from typing import Any

from onnx import ModelProto, NodeProto
from zigzag.datatypes import LayerDim
from zigzag.parser.workload_factory import LayerNodeFactory

from stream.hardware.architecture.accelerator import Accelerator
from stream.onnx_utils import get_onnx_output_shapes
from stream.parser.onnx.operator_parser import OnnxComputeOperatorParser
from stream.workload.computation.computation_node import GeneratedComputationNode
from stream.workload.dependency_propagation.concat_node import ConcatNode
from stream.workload.dependency_propagation.split_node import SplitNode
from stream.workload.mapping import InterCoreMappingAttributes

logger = logging.getLogger(__name__)


class SSMNode(StrEnum):
    Mul_h_dA = "mul_h_dA"
    Add_dBx = "add_dBx"
    Mul_h_C = "mul_h_C"


class SSMInput(StrEnum):
    da = "dA"
    dbx = "dBx"
    C = "C"


class SSMParser(OnnxComputeOperatorParser):
    def __init__(
        self,
        node_id: int,
        node: NodeProto,
        nodes_outputs: dict[int, Any],
        onnx_model: ModelProto,
        all_mappings: dict[str, InterCoreMappingAttributes],
        accelerator: Accelerator,
    ):
        super().__init__(
            node_id=node_id,
            node=node,
            nodes_outputs=nodes_outputs,
            onnx_model=onnx_model,
            all_mappings=all_mappings,
            accelerator=accelerator,
        )

        self.__node_id_tracker = node_id
        self.set_shape_info()
        self._set_mapping()
        self.base_ids: dict[SSMNode, int] = {}

    def run(self):
        yield from self.get_nodes()

    def get_nodes(self):
        # Nodes that will split dA, dBx and C into L pieces
        split_node_dA = self.create_split_node(SSMInput.da)
        yield split_node_dA
        split_node_dBx = self.create_split_node(SSMInput.dbx)
        yield split_node_dBx
        split_node_C = self.create_split_node(SSMInput.C)
        yield split_node_C

        # Iterate over sequence length
        prev_h_id = None
        all_y_ids: list[int] = []
        for i in range(self.L):
            mul_h_dA_node = self.create_compute_node(
                SSMNode.Mul_h_dA, idx=i, source_1=prev_h_id, source_2=split_node_dA.id
            )
            yield mul_h_dA_node
            mul_add_dBx_node = self.create_compute_node(
                SSMNode.Add_dBx, idx=i, source_1=mul_h_dA_node.id, source_2=split_node_dBx.id
            )
            yield mul_add_dBx_node
            mul_h_C_node = self.create_compute_node(
                SSMNode.Mul_h_C, idx=i, source_1=mul_add_dBx_node.id, source_2=split_node_C.id
            )
            yield mul_h_C_node
            all_y_ids.append(mul_h_C_node.id)
            prev_h_id = mul_add_dBx_node.id

        final_concat_node = self.create_concat_node(all_y_ids)
        yield final_concat_node

    def create_compute_node(self, ssm_node_type: SSMNode, idx: int, source_1: int | None, source_2: int):
        node_id = self.get_and_increment_id()
        op_type = "add" if ssm_node_type == SSMNode.Add_dBx else "mul"
        node_name = str(ssm_node_type)

        # First time this node_type is created, log it's base id
        if idx == 0:
            self.base_ids[ssm_node_type] = node_id
        base_id = self.base_ids[ssm_node_type]

        if source_1 is None:
            # source 1 will be considered as constant input
            source_1 = node_id

        node_data = self.get_node_user_format(
            ssm_node_type, node_id=node_id, op_type=op_type, node_name=node_name, source_1=source_1, source_2=source_2
        )
        node_factory = LayerNodeFactory(node_data, mapping_data=[])
        node_attrs = node_factory.create_node_attr()

        node = GeneratedComputationNode(
            node_id=node_id,
            gen_id=idx,
            gen_split_layer_dim=LayerDim("L"),
            base_id=base_id,
            node_name=node_name,
            op_type=op_type,
            node_attr=node_attrs,
            mapping_attr=self.mapping,
            input_names=[],
        )

        return node

    def create_split_node(self, ssm_input_type: SSMInput):
        node_id = self.get_and_increment_id()
        node_name = f"split_{str(ssm_input_type)}"
        predecessor_id = self.get_predecessor_id(ssm_input_type)

        # Shape is (B, L, D) or (B, L, N) -> split on L
        axis = 1

        return SplitNode(
            node_id=node_id,
            node_name=node_name,
            predecessor=predecessor_id,
            axis=axis,
            splits=self.L * [1],
            output_names=[],
            input_names=list(self.node.input),
        )

    def create_concat_node(self, input_ids: list[int]):
        node_id = self.get_and_increment_id()

        return ConcatNode(
            node_id=node_id,
            node_name="concat_y",
            predecessors=input_ids,
            output_shape=(self.B, self.L, self.D),
            axis=1,
            axis_exists_in_input=True,  # Because the smaller y nodes have size-1 L dimension at axis 1
        )

    def get_predecessor_id(self, ssm_input_type: SSMInput):
        """Get the id for the input node from the ONNX graph
        # TODO input order is assumed
        """
        match ssm_input_type:
            case SSMInput.da:
                relevant_input_name = self.node.input[0]
            case SSMInput.dbx:
                relevant_input_name = self.node.input[1]
            case SSMInput.C:
                relevant_input_name = self.node.input[2]

        # Find id through previously seen output names
        try:
            return next(
                node_id
                for node_id, outputs_of_node in self.nodes_outputs.items()
                if relevant_input_name in outputs_of_node
            )
        except StopIteration as exc:
            raise ValueError(
                f"{ssm_input_type} not seen before! Assuming corresponding input name is {relevant_input_name}"
            ) from exc

    def get_node_user_format(
        self, ssm_node_type: SSMNode, node_id: int, node_name: str, source_1: int, source_2: int, op_type: str
    ):
        """
        Generate the necessary dictionary items required for the LayerNode creation.
        """
        act_precision = self.get_activation_precision()
        data: dict[str, Any] = {}
        data["id"] = node_id
        data["name"] = node_name
        data["operator_type"] = op_type
        data["operand_source"] = {"I": source_1, "W": source_2}
        data["operand_precision"] = {
            "W": act_precision,
            "I": act_precision,
            "O_final": act_precision,
            "O": act_precision,
        }
        data["dimension_relations"] = []
        # Smaller nodes will have size-1 dimension in L
        data["loop_sizes"] = [self.B, 1, self.D, self.N]
        data["loop_dims"] = ["B", "L", "D", "N"]

        match ssm_node_type:
            case SSMNode.Mul_h_dA:
                data["equation"] = "O[b][l][d][n]+=I[b][l][d][n]*W[b][l][d][n]"
            case SSMNode.Add_dBx:
                data["equation"] = "O[b][l][d][n]+=I[b][l][d][n]*W[b][l][d][n]"
            case SSMNode.Mul_h_C:
                data["equation"] = "O[b][l][d]+=I[b][l][d][n]*W[b][l][n]"
        return data

    def get_and_increment_id(self):
        """Keeps track of how many nodes have been created. Returns a new id that has not been used before"""
        curr_id = self.__node_id_tracker
        self.__node_id_tracker += 1
        return curr_id

    def set_shape_info(self):
        """Set the SSM dimensions based on the output shapes.
        The output shapes will be (B, L, D) and (B, D, N)"""
        output_shapes = get_onnx_output_shapes(self.node, self.onnx_model)
        EXPECTED_NUMBER_OF_OUTPUTS = 2
        assert len(output_shapes) == EXPECTED_NUMBER_OF_OUTPUTS, "SSM node should have 2 outputs: Y and state"

        # The ordering is an assumption. Sanity check that B and D dimension are at least the same
        shape_y, shape_h = output_shapes
        B, L, D = shape_y
        _, _, N = shape_h
        assert B == shape_h[0] and D == shape_h[1]
        self.L = L
        self.D = D
        self.N = N
        self.B = B

    def _set_mapping(self):
        self.mapping = self.get_mapping_this_node()

    def get_layer_node_user_format(
        self, input_shape: list[int], output_shape: list[int], mapping: InterCoreMappingAttributes | None
    ) -> dict[str, Any]:
        """Not used for this class, but abstract base class requires instantiation anyway"""
        ...
