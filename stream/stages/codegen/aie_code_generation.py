from collections.abc import Sequence
from typing import Any

from snaxc.dialects.tsl import TSL
from xdsl.context import MLContext
from xdsl.dialects.builtin import MemRefType, ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import Operand
from xdsl.printer import Printer
from zigzag.utils import DiGraphWrapper

from stream.compiler.dialects.stream import ComputationNodeOp, EdgeOp, Stream, TransferOp
from stream.compiler.transforms.aie_add_tracing_script import AIEAddTracingScript
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_stream_to_aie import ConvertStreamToAIEPass
from stream.compiler.transforms.stream_split_transfers import StreamSplitTransfersPass
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.tensor import SteadyStateTensor
from stream.workload.steady_state.transfer import SteadyStateTransfer
from stream.workload.steady_state.workload import SteadyStateWorkload


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

        self.trace_size = kwargs.get("trace_size", 1048576)

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in sub_stage.run():
            if cme:
                self.codegen_main(cme, self.trace_size)
            yield cme, extra_info

    def create_edge_op(
        self,
        workload: DiGraphWrapper[SteadyStateNode],
        edge: SteadyStateTensor,
        transfer_ops: dict[SteadyStateTransfer, TransferOp],
    ) -> EdgeOp:
        """
        Create an EdgeOp for a given SteadyStateTensor.
        """
        assert edge.full_shape is not None
        memref_type = edge.subview.result.type
        assert isinstance(memref_type, MemRefType)
        # get rid of layout for now
        memref_type = MemRefType(memref_type.element_type, memref_type.shape)
        if workload.out_degree(edge) > 0:
            edge_op = EdgeOp(memref_type, edge.node_name)
        else:
            # find transfer:
            transfer_results = []
            for transfer in workload.predecessors(edge):
                assert isinstance(transfer, SteadyStateTransfer)
                print("\t", transfer, transfer.chosen_memory_core)
                transfer_op = transfer_ops[transfer]
                transfer_results.append(transfer_op.results[0])
            edge_op = EdgeOp(None, edge.node_name, transfer_results)
        return edge_op

    def create_transfer_op(
        self,
        workload: DiGraphWrapper[SteadyStateNode],
        transfer: SteadyStateTransfer,
        compute_ops: dict[SteadyStateComputation, ComputationNodeOp],
        edge_ops: dict[SteadyStateTensor, EdgeOp],
    ) -> TransferOp:
        """
        Create a TransferOp for a given SteadyStateTransfer.
        """
        # Get source and dest and convert to string for TransferOp creation which uses string
        source = transfer.srcs[0]
        dest = transfer.dsts[0]

        if isinstance(dest, SteadyStateComputation):
            pass

        dest_tensor = next(workload.successors(transfer))
        assert isinstance(dest_tensor, SteadyStateTensor)
        dest_type = dest_tensor.subview.result.type
        assert isinstance(dest_type, MemRefType)

        # get rid of layout for now
        assert isinstance(dest_type, MemRefType)
        dest_type = MemRefType(dest_type.element_type, dest_type.shape)

        source_vals: Sequence[SSAValue | Operation] = []
        for source in transfer.srcs:
            if source in edge_ops:
                source_val = edge_ops[source]
            else:
                # find source op
                source_source = next(workload.predecessors(source))
                assert isinstance(source_source, SteadyStateComputation)
                source_val = compute_ops[source_source]
            source_vals.append(source_val)

        if isinstance(source_vals[0], EdgeOp):
            tensor = transfer.dsts[0]
        else:
            tensor = transfer.srcs[0]
        offsets = [x[0] for x in tensor.loop_ranges]
        sizes = [x[1] - x[0] for x in tensor.loop_ranges]
        strides = [1 for x in tensor.loop_ranges]

        # determine spatial strides
        spatial_strides = []
        if len(source_vals) > 1:
            sorted_transfers = sorted(transfer.srcs, key=lambda x: x.loop_ranges)
        else:
            sorted_transfers = sorted(transfer.dsts, key=lambda x: x.loop_ranges)
        if len(sorted_transfers) > 1:
            for dim in tensor.loop_dimensions:
                spatial_strides.append(
                    sorted_transfers[1].loop_ranges_per_dim[dim][0] - sorted_transfers[0].loop_ranges_per_dim[dim][0]
                )
        else:
            for dim in tensor.loop_dimensions:
                spatial_strides.append(transfer.srcs[0].loop_ranges_per_dim[dim][0])

        if not any(spatial_strides):
            result_types = [dest_type]
        else:
            result_types = [dest_type] * len(transfer.dsts)

        op = TransferOp(
            source_vals,
            result_types,
            "source_todo",
            "dest_todo",
            "tensor_todo",
            transfer.steady_state_iteration_space,
            offsets,
            sizes,
            strides,
            spatial_strides,
            [str(dim) for dim in tensor.loop_dimensions],
            str(transfer.chosen_memory_core),
        )

        return op

    def create_computation_node_op(
        self,
        workload: DiGraphWrapper[SteadyStateNode],
        compute: SteadyStateComputation,
        transfer_ops: dict[SteadyStateTransfer, TransferOp],
    ) -> ComputationNodeOp:
        # get inputs
        inputs = list(workload.predecessors(compute))
        transfers: list[TransferOp] = []
        operands: list[Operand] = []
        for input in inputs:
            transfer = next(workload.predecessors(input))
            assert isinstance(transfer, SteadyStateTransfer)
            transfers.append(transfer_op := transfer_ops[transfer])
            if len(transfer_op.results) == 1:
                operands.append(transfer_op.results[0])
            else:
                sorted_transfers = sorted(transfer.dsts, key=lambda x: x.loop_ranges)
                assert isinstance(input, SteadyStateTensor)
                index = sorted_transfers.index(input)
                operands.append(transfer_op.results[index])

        # get output type:
        output_type = next(
            tensor
            for tensor in workload.successors(compute)
            if isinstance(tensor, SteadyStateTensor) and tensor.operand == compute.output_operand
        ).subview.result.type
        # get rid of layout for now
        assert isinstance(output_type, MemRefType)
        output_type = MemRefType(output_type.element_type, output_type.shape)

        # create computation node op with the needed information
        op = ComputationNodeOp(
            operands,
            None,
            kernel=compute.kernel.name,
            core_allocation=str(compute.chosen_resource_allocation),
            ssis=compute.steady_state_iteration_space,
            result_types=[output_type],
        )

        return op

    def generate_steady_state_workload(self, workload: SteadyStateWorkload) -> ModuleOp:
        edge_ops: dict[SteadyStateTensor, EdgeOp] = {}
        transfer_ops: dict[SteadyStateTransfer, TransferOp] = {}
        compute_ops: dict[SteadyStateComputation, ComputationNodeOp] = {}

        all_ops: list[Operation] = []

        for node in workload.topological_sort():
            # create edge
            if isinstance(node, SteadyStateTensor) and (
                workload.in_degree(node) == 0 or workload.out_degree(node) == 0
            ):
                # Create edge op for the tensor
                edge_op = self.create_edge_op(workload, node, transfer_ops)
                edge_ops[node] = edge_op
                all_ops.append(edge_op)

            # create transfer
            if isinstance(node, SteadyStateTransfer):
                transfer_op = self.create_transfer_op(workload, node, compute_ops, edge_ops)
                transfer_ops[node] = transfer_op
                all_ops.append(transfer_op)

            if isinstance(node, SteadyStateComputation):
                computation_node_op = self.create_computation_node_op(workload, node, transfer_ops)
                compute_ops[node] = computation_node_op
                all_ops.append(computation_node_op)

        module = ModuleOp(list(all_ops))

        return module

    def codegen_main(self, cme: SteadyStateScheduler, trace_size: int) -> None:
        workload = cme.steady_state_workload
        assert workload is not None

        module = self.generate_steady_state_workload(workload)

        # SetNoReusePass().apply(self.context, module)
        # Split transfers in push and pull
        StreamSplitTransfersPass().apply(self.context, module)

        # Convert to AIE
        ConvertStreamToAIEPass().apply(self.context, module)

        # Remove custom layout attributes
        ClearMemorySpace().apply(self.context, module)

        # Optionally, Add Tracing Script
        AIEAddTracingScript(trace_size=trace_size).apply(self.context, module)

        # print output to codegen path
        file = open(self.output_path, "w")
        printer = Printer(file)
        printer.print(module)

    def is_leaf(self) -> bool:
        return False
