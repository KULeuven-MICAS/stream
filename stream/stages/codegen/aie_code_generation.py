from collections import defaultdict
from typing import Any, cast

from snaxc.dialects.tsl import TSL
from xdsl.context import MLContext
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, MemRefType, ModuleOp
from xdsl.printer import Printer
from zigzag.datatypes import Constants, LayerOperand

from stream.compiler.dialects.stream import ComputationNodeOp, EdgeOp, EmptySSAValue, Stream, TransferOp
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_stream_to_aie import ConvertStreamToAIEPass
from stream.cost_model.communication_manager import CommunicationLinkEvent
from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.tensor import SubviewTensor


class AIECodeGenerationStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)

        # set up the correct xDSL context
        self.context: MLContext = MLContext()

        # add custom dialects and passes
        self.context.load_dialect(Stream)
        self.context.load_dialect(TSL)

        self.output_path: str = kwargs["codegen_path"]

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in sub_stage.run():
            if cme:
                self.codegen_main(cme)
            yield cme, extra_info

    def generate_stream_workload(self, cme: StreamCostModelEvaluation) -> ModuleOp:

        # (precision, [loop_range_0, .. loop_range_n-1])
        edges_ranges: dict[tuple[int, LayerOperand], tuple[int, list[int]]] = {}
        edge_ops: dict[tuple[int, LayerOperand], EdgeOp] = {}

        for node in cme.workload.node_list:
            for operand in node.layer_operands:
                key = (node.id, operand)
                tensor = node.operand_tensors[operand]

                if tensor.layer_operand == Constants.OUTPUT_LAYER_OP:
                    precision = tensor.cn_source.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
                else:
                    precision = tensor.cn_source.operand_precision[tensor.layer_operand]

                edges_ranges[key] = (precision, tensor.original_shape)

        edge_ops = {
            key: EdgeOp(MemRefType(IntegerType(precision), loop_ranges), str(key))
            for key, (precision, loop_ranges) in edges_ranges.items()
        }

        # create computation nodes for all computation nodes
        nodes: dict[ComputationNode, ComputationNodeOp] = {}

        for node in cme.workload.node_list:

            # build operand types:
            operand_types = []

            for operand in (*node.constant_operands, node.output_operand):
                size = cast(int, node.operand_size_elem[operand])
                if operand == node.output_operand:
                    operand = Constants.FINAL_OUTPUT_LAYER_OP
                precision = cast(int, node.operand_precision[operand])
                typ = MemRefType(IntegerType(precision), [size])
                operand_types.append(typ)

            input_operands, output_operand = operand_types[:-1], operand_types[-1]

            # create computation node op with the needed information
            op = ComputationNodeOp(
                [EmptySSAValue(typ) for typ in input_operands],
                EmptySSAValue(output_operand),
                kernel=node.kernel.name,
                core_allocation=str(node.chosen_core_allocation),
                repeat=1,
            )

            nodes[node] = op

        # gather all transfers
        transfer_list: list[CommunicationLinkEvent] = []

        for communication_event in cme.accelerator.communication_manager.events:
            for cle in communication_event.tasks:
                transfer_list.append(cle)

        transfer_list.sort(key=lambda x: x.start)

        # transfers are unique per Layer Operand and Steady state stuff
        transfers: dict[SubviewTensor, list[tuple[CommunicationLinkEvent, TransferOp]]] = defaultdict(list)
        transfer_ops: list[TransferOp] = []

        for cle in transfer_list:

            tensor = cle.tensors[0]

            layer_id_and_op = edge_ops[(tensor.id[0], tensor.id[2])]

            # Get source and dest and convert to string for TransferOp creation which uses string
            source = str(cle.source)
            dest = str(cle.destinations[0])  # TODO: Support broadcasting to multiple destinations

            # size = cast(int, tensor.origin.operand_size_elem[tensor.layer_operand])
            if tensor.layer_operand == Constants.OUTPUT_LAYER_OP:
                precision = tensor.cn_source.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
            else:
                precision = tensor.cn_source.operand_precision[tensor.layer_operand]

            offsets = [x[0] for x in tensor.loop_ranges]
            sizes = [x[1] - x[0] for x in tensor.loop_ranges]
            strides = [1 for x in tensor.loop_ranges]

            result_type = MemRefType(IntegerType(precision), sizes)

            op = TransferOp(layer_id_and_op, [result_type], source, dest, str(tensor), 1, offsets, sizes, strides)

            transfers[tensor].append((cle, op))
            transfer_ops.append(op)

        # sort ops based on start
        for ops in transfers.values():
            ops.sort(key=lambda x: x[0].start)

        # Connect the operands of transfers to computation nodes
        for node, node_op in nodes.items():
            operands = node.layer_operands
            operand_order = ["I", "W", "O"]
            for operand in operands:
                tensor = node.operand_tensors[operand]
                # find last transfer for this tensor
                for cle, transfer_op in transfers[tensor]:
                    assert node.start
                    if not operand.is_output():
                        if cle.end > node.start:
                            break
                        if cle.destinations[0].id != node.chosen_core_allocation:
                            continue
                    else:
                        # TODO
                        pass
                    node_op.operands[operand_order.index(operand.name)] = transfer_op.results[0]

        # add all nodes and transfers to the module
        node_ops = tuple(nodes.values())
        all_ops = tuple(edge_ops.values()) + tuple(transfer_ops) + node_ops
        module = ModuleOp(list(all_ops))

        return module

    def generate_stream_workload_steady_state(self, cme: StreamCostModelEvaluation) -> ModuleOp:

        # steady state nodes: need some representation here
        # this is currently a mapping id -> list[node]
        # assuming that a layer is entirely steady state and can
        # be differentiated based on its id.
        # The values of this dict are sequential lists of the computation
        # nodes that are `equal` (<-> can be represented with a for loop)
        nodes_steady_state: dict[int, list[ComputationNode]] = defaultdict(list)

        # assuming node_list is ordered in order of execution
        # (tends to be correct, but I guess this is not necessarily correct to assume)
        for node in cme.workload.node_list:
            nodes_steady_state[node.id].append(node)

        # create computation nodes for all computation nodes
        nodes: dict[ComputationNode, ComputationNodeOp] = {}

        for node_list in nodes_steady_state.values():

            # take first node as the reference
            node = node_list[0]

            # build operand types:
            operand_types = []

            for operand in (*node.constant_operands, node.output_operand):
                size = cast(int, node.operand_size_elem[operand])
                if operand == node.output_operand:
                    operand = Constants.FINAL_OUTPUT_LAYER_OP
                precision = cast(int, node.operand_precision[operand])
                typ = MemRefType(IntegerType(precision), [size])
                operand_types.append(typ)

            input_operands, output_operand = operand_types[:-1], operand_types[-1]

            # create computation node op with the needed information
            op = ComputationNodeOp(
                [EmptySSAValue(typ) for typ in input_operands],
                EmptySSAValue(output_operand),
                kernel=node.kernel.name,
                core_allocation=str(node.core_allocation[0]),
                repeat=len(node_list),
            )

            # complete mapping from node -> ComputationNodeOp
            for node in node_list:
                nodes[node] = op

        # gather all transfers
        transfer_list: list[tuple[CommunicationLinkEvent, CommunicationLink]] = []

        for _, link_pair in cme.accelerator.communication_manager.pair_links.items():
            if link_pair:
                for link in link_pair:
                    for event in link.events:
                        transfer_list.append((event, link))

        transfer_list.sort(key=lambda x: x[0].start)

        # create transfer ops for every transfer
        # transfers: dict[Tensor, TransferOp] = {}

        # transfers are unique per Layer Operand and Steady state stuff
        transfers: dict[tuple[int, LayerOperand], TransferOp] = {}

        for transfer, link in transfer_list:

            tensor = transfer.tensors[0]

            if (tensor.id[0], tensor.id[2]) in transfers:
                # increse repeat field of existing transfer op
                transfer_op = transfers[(tensor.id[0], tensor.id[2])]
                transfer_op.repeat = IntegerAttr(transfer_op.repeat.value.data + 1, IndexType())

            else:
                # create transfer op

                # TODO: why is this backwards?
                dest = str(link.sender)
                source = str(link.receiver)

                size = cast(int, tensor.cn_source.operand_size_elem[tensor.layer_operand])
                if tensor.layer_operand == Constants.OUTPUT_LAYER_OP:
                    precision = tensor.cn_source.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
                else:
                    precision = tensor.cn_source.operand_precision[tensor.layer_operand]

                result_type = MemRefType(IntegerType(precision), [size])

                op = TransferOp(None, [result_type], source, dest, str(tensor), repeat=1)

                transfers[(tensor.id[0], tensor.id[2])] = op

        # Connect the operands of transfers to computation nodes
        for node_id, nodes_list in nodes_steady_state.items():
            node = nodes_list[0]  # first node as reference
            operands = node.layer_operands
            operand_order = ["I", "W", "O"]
            for operand in operands:
                tensor = node.operand_tensors[operand]
                nodes[node].operands[operand_order.index(operand.name)] = transfers[
                    (tensor.id[0], tensor.id[2])
                ].results[0]

        # add all nodes and transfers to the module
        transfer_ops = tuple(transfers.values())
        node_ops = tuple(nodes[node_list[0]] for node_list in nodes_steady_state.values())
        all_ops = transfer_ops + node_ops
        module = ModuleOp(list(all_ops))

        return module

    def codegen_main(self, cme: StreamCostModelEvaluation):

        # generate workload based on cme:
        module = self.generate_stream_workload(cme)

        # StreamLoopRollerPass().apply(self.context, module)

        # Convert to AIE
        ConvertStreamToAIEPass().apply(self.context, module)

        # Remove custom layout attributes
        ClearMemorySpace().apply(self.context, module)

        # print output to codegen path
        file = open(self.output_path, "w")
        printer = Printer(file)
        printer.print(module)

    def is_leaf(self) -> bool:
        return False
