from typing import Any, cast
from stream.hardware.architecture.noc.communication_link import CommunicationLink

from xdsl.context import MLContext
from xdsl.dialects.builtin import IntegerType, MemRefType, ModuleOp
from xdsl.printer import Printer
from xdsl.xdsl_opt_main import xDSLOptMain

from stream.compiler.dialects.stream import ComputationNodeOp, EmptySSAValue, Stream, TransferOp
from stream.compiler.transforms.convert_stream_to_aie import ConvertStreamToAIEPass
from stream.cost_model.communication_manager import CommunicationLinkEvent
from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode
from stream.workload.tensor import Tensor

from zigzag.datatypes import Constants


class AIECodeGenerationStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)

        # set up the correct xDSL context
        self.context: MLContext = xDSLOptMain().ctx.clone()

        # add custom dialects and passes
        self.context.load_dialect(Stream)

        self.output_path: str = kwargs["codegen_path"]

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in sub_stage.run():
            if cme:
                self.codegen_main(cme)
            yield cme, extra_info

    def codegen_main(self, cme: StreamCostModelEvaluation):

        # gather all nodes
        nodes: dict[ComputationNode, ComputationNodeOp] = {}

        for node in cme.workload.nodes:

            operand_types = []

            # build operand types:

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
                node.kernel.name,
                node.core_allocation[0],
            )

            nodes[node] = op

            # only consider 4
            if len(nodes) >= 2:
                break


        # gather all transfers
        transfer_list: list[tuple[CommunicationLinkEvent, CommunicationLink]] = []

        for _, link_pair in cme.accelerator.communication_manager.pair_links.items():
            if link_pair:
                for link in link_pair:
                    for event in link.events:
                        transfer_list.append((event, link))

        # create transfer ops for every transfer
        transfers: dict[Tensor, TransferOp] = {}

        for transfer, link in transfer_list:

            # TODO: why is this backwards?
            dest = str(link.sender)
            source = str(link.receiver)
            tensor = transfer.tensors[0]

            # TODO: should not be necessary anymore:
            # only consider the one with included node
            if tensor.origin not in nodes:
                continue

            size = cast(int, tensor.origin.operand_size_elem[tensor.layer_operand])
            if tensor.layer_operand == Constants.OUTPUT_LAYER_OP:
                precision = tensor.origin.operand_precision[Constants.FINAL_OUTPUT_LAYER_OP]
            else:
                precision = tensor.origin.operand_precision[tensor.layer_operand]

            result_type = MemRefType(IntegerType(precision), [size])

            # transfer.tensors[0].origin.operand_size_elem[transfer.tensors[0].layer_operand]

            op = TransferOp(None, [result_type], source, dest, str(tensor))

            transfers[transfer.tensors[0]] = op

            # make sure the operation uses the result of this transfer
            # order = ["I", "W", "O"]
            # nodes[tensor.origin].operands[order.index(tensor.layer_operand.name)] = op.results[0]

        for node, node_op in nodes.items():
            operands = node.layer_operands
            for i, operand in enumerate(reversed(operands)):
                tensor = node.operand_tensors[operand]
                node_op.operands[i] = transfers[tensor].results[0]

        # add all nodes and transfers to the module
        transfer_ops = tuple(transfers.values())
        node_ops = tuple(nodes.values())
        all_ops = transfer_ops + node_ops
        module = ModuleOp(list(all_ops))

        # Convert to AIE
        ConvertStreamToAIEPass().apply(self.context, module)

        # print output to codegen path
        file = open(self.output_path, "w")
        printer = Printer(file)
        printer.print(module)

    def is_leaf(self) -> bool:
        return False
