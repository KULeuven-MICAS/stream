from typing import Any, cast

from xdsl.context import MLContext
from xdsl.dialects.builtin import IntegerType, MemRefType, ModuleOp
from xdsl.xdsl_opt_main import xDSLOptMain

from stream.compiler.dialects.zigzag import ComputationNodeOp, EmptySSAValue, TransferOp, ZigZag
from stream.compiler.transforms.convert_zigzag_to_aie import ConvertZigZagToAIEPass
from stream.cost_model.communication_manager import CommunicationLinkEvent
from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.stages.stage import Stage, StageCallable
from stream.workload.computation.computation_node import ComputationNode


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
        self.context.load_dialect(ZigZag)

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in sub_stage.run():
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
                precision = cast(int, node.operand_precision[operand])
                typ = MemRefType(IntegerType(precision), [size])
                operand_types.append(typ)

            input_operands, output_operand = operand_types[:-1], operand_types[-1]

            # create computation node op with the needed information

            op = ComputationNodeOp(
                [EmptySSAValue(typ) for typ in input_operands],
                EmptySSAValue(output_operand),
                "conv2dk1_i8",
                node.core_allocation[0],
            )

            nodes[node] = op

        # gather all transfers
        transfer_list = []

        for _, link_pair in cme.accelerator.communication_manager.pair_links.items():
            if link_pair:
                for link in link_pair:
                    for event in link.events:
                        transfer_list.append((event, link))

        # create transfer ops for every transfer
        transfers: dict[CommunicationLinkEvent, TransferOp] = {}

        for transfer, link in transfer_list:

            source = str(link.sender)
            dest = str(link.receiver)
            tensor = transfer.tensors[0]

            size = cast(int, tensor.origin.operand_size_elem[tensor.layer_operand])
            precision = tensor.origin.operand_precision[tensor.layer_operand]

            result_type = MemRefType(IntegerType(precision), [size])

            transfer.tensors[0].origin.operand_size_elem[transfer.tensors[0].layer_operand]

            op = TransferOp(None, [result_type], source, dest, str(tensor))

            transfers[transfer] = op

            # make sure the operation uses the result of this transfer
            order = ["I", "W", "O"]
            nodes[tensor.origin].operands[order.index(tensor.layer_operand.name)] = op.results[0]

        # add all nodes and transfers to the module
        transfer_ops = tuple(transfers.values())
        node_ops = tuple(nodes.values())
        all_ops = transfer_ops + node_ops
        module = ModuleOp(list(all_ops))

        ConvertZigZagToAIEPass().apply(self.context, module)

    def is_leaf(self) -> bool:
        return False
