import warnings
from collections.abc import Sequence
from copy import copy
from math import prod
from typing import Any, cast

from snaxc.dialects.snax import NoneAttr
from snaxc.dialects.tsl import TSL
from xdsl.context import MLContext
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, MemRefType, ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import Operand
from xdsl_aie.dialects.aie import AIEDeviceEnum
from zigzag.datatypes import LayerOperand
from zigzag.utils import DiGraphWrapper

from stream.compiler.dialects.stream import ComputationNodeOp, EdgeOp, Stream, TransferOp

# from stream.compiler.transforms.aie_add_tracing_script import AIEAddTracingScript
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_stream_to_aie import ConvertStreamToAIEPass
from stream.compiler.transforms.stream_split_transfers import StreamSplitTransfersPass
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.iteration_space import SteadyStateIterationSpace
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
        self.npu = kwargs.get("npu", "npu2")
        self.runtime_args = kwargs.get("runtime_args", [])
        self.module = None

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)

        for cme, extra_info in sub_stage.run():
            if cme:
                self.codegen_main(cme, self.trace_size, self.npu)
                assert self.module is not None
            yield self.module, extra_info

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

    def create_transfer_op(  # noqa: PLR0912, PLR0915
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

        # construct destionation steady state iteration space
        dest_op = next(workload.successors(next(workload.successors(transfer))), None)
        is_output = transfer.tensor.origin.output_operand == transfer.tensor.operand
        if isinstance(dest_op, SteadyStateComputation) and is_output:
            current_operands = transfer.tensor.loop_dimensions
            next_layer_operand = next(
                key for key, val in dest_op.input_operand_source.items() if val == transfer.tensor.origin.id
            )
            next_operand_dims = dest_op.operand_dimensionality_order[next_layer_operand]
            operand_mapping = {prev: cur for (prev, cur) in zip(current_operands, next_operand_dims, strict=True)}
            dest_ssis_vars = []
            for ssis_var in transfer.steady_state_iteration_space:
                copyed_ssis_var = copy(ssis_var)
                if copyed_ssis_var.dimension not in operand_mapping:
                    assert not copyed_ssis_var.relevant
                else:
                    copyed_ssis_var.dimension = operand_mapping[ssis_var.dimension]
                    dest_ssis_vars.append(copyed_ssis_var)
            dest_ssis = SteadyStateIterationSpace(dest_ssis_vars)
        else:
            dest_ssis = transfer.steady_state_iteration_space

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
                if source not in workload:
                    warnings.warn(f"Source {source} not in workload, applying a little hack :)")  # noqa: B028
                    source_source = next(workload.predecessors(transfer))
                    source_source = next(workload.predecessors(source_source))
                else:
                    source_source = next(workload.predecessors(source))
                assert isinstance(source_source, SteadyStateComputation)
                source_val = compute_ops[source_source]
            source_vals.append(source_val)

        if isinstance(source_vals[0], EdgeOp):
            tensor = sorted(transfer.dsts, key=lambda x: x.loop_ranges)[0]
        else:
            tensor = sorted(transfer.srcs, key=lambda x: x.loop_ranges)[0]

        offsets = [x[0] for x in tensor.loop_ranges]
        sizes = [x[1] - x[0] for x in tensor.loop_ranges]
        strides = []
        for loop_dim in tensor.loop_dimensions:
            stride = prod(
                iv.size
                for iv in transfer.steady_state_iteration_space.variables
                if iv.spatial and iv.dimension == loop_dim
            )
            strides.append(stride)

        # determine spatial strides
        spatial_strides = []
        if len(source_vals) > 1:
            sorted_transfers = sorted(transfer.srcs, key=lambda x: x.loop_ranges)
        else:
            sorted_transfers = sorted(transfer.dsts, key=lambda x: x.loop_ranges)
        unique_transfers = {x.loop_ranges: x for x in sorted_transfers}
        if len(unique_transfers) > 1:
            for dim in tensor.loop_dimensions:
                spatial_strides.append(
                    list(unique_transfers.values())[1].loop_ranges_per_dim[dim][0]
                    - list(unique_transfers.values())[0].loop_ranges_per_dim[dim][0]
                )
        else:
            for dim in tensor.loop_dimensions:
                spatial_strides.append(transfer.srcs[0].loop_ranges_per_dim[dim][0])

        result_types = [dest_type] * len(unique_transfers)

        memtile = transfer.chosen_memory_core
        if memtile is None:
            memtile = NoneAttr()
        else:
            row, col = memtile.row_id, memtile.col_id
            assert row is not None
            assert col is not None
            memtile = ArrayAttr([IntegerAttr.from_index_int_value(x) for x in (col, row)])

        op = TransferOp(
            source_vals,
            result_types,
            "source_todo",
            "dest_todo",
            "tensor_todo",
            transfer.steady_state_iteration_space,
            dest_ssis,
            offsets,
            sizes,
            strides,
            spatial_strides,
            [str(dim) for dim in tensor.loop_dimensions],
            memtile,
        )

        return op

    def create_computation_node_op(
        self,
        workload: DiGraphWrapper[SteadyStateNode],
        compute: SteadyStateComputation,
        transfer_ops: dict[SteadyStateTransfer, TransferOp],
    ) -> ComputationNodeOp:
        # get inputs
        inputs = [(x[0], x[2].get("operand")) for x in workload.in_edges(compute, data=True)]
        ordered_inputs = map(
            lambda x: x[0], sorted(inputs, key=lambda x: compute.input_operands.index(cast(LayerOperand, x[1])))
        )
        transfers: list[TransferOp] = []
        operands: list[Operand] = []
        for input in ordered_inputs:
            transfer = next(workload.predecessors(input))
            assert isinstance(transfer, SteadyStateTransfer)
            transfers.append(transfer_op := transfer_ops[transfer])
            if len(transfer_op.results) == 1:
                operands.append(transfer_op.results[0])
            else:
                sorted_transfers = sorted(transfer.dsts, key=lambda x: x.loop_ranges)
                unique_transfers = {x.loop_ranges: x for x in sorted_transfers}
                assert isinstance(input, SteadyStateTensor)
                index = list(unique_transfers.keys()).index(input.loop_ranges)
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

        assert (core := compute.chosen_resource_allocation) is not None
        row, col = core.row_id, core.col_id
        assert row is not None
        assert col is not None
        core_allocation = ArrayAttr([IntegerAttr.from_index_int_value(x) for x in (col, row)])
        # create computation node op with the needed information
        op = ComputationNodeOp(
            operands,
            None,
            kernel=compute.kernel.function_name,
            core_allocation=core_allocation,
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

    def codegen_main(self, sss: SteadyStateScheduler, trace_size: int, npu: AIEDeviceEnum) -> None:
        workload = sss.steady_state_workload
        assert workload is not None

        aie_kernels = {
            node.kernel.function_name: node.kernel
            for node in workload.node_list
            if isinstance(node, SteadyStateComputation)
        }

        module = self.generate_steady_state_workload(workload)

        # SetNoReusePass().apply(self.context, module)
        # Split transfers in push and pull
        StreamSplitTransfersPass().apply(self.context, module)

        # Arguments that will be supplied via runtime sequence, modify as needed
        # args = ["Op0.I_in", "Op0.W_in", "Op0.O_out"]  # gemm
        # args = ["Gemm_Right.I_in", "Gemm_Right.W_in", "Gemm_Left.W_in", "Elt_Mul.O_out"]  # swiglu
        args = self.runtime_args  # will be inferred automatically based on EdgeOps

        # Convert to AIE
        ConvertStreamToAIEPass(args, aie_kernels).apply(self.context, module, npu)

        # Remove custom layout attributes
        ClearMemorySpace().apply(self.context, module)

        # Optionally, Add Tracing Script
        # if False:
        # AIEAddTracingScript(trace_size=trace_size).apply(self.context, module)

        # print output to codegen path
        # file = open(self.output_path, "w")
        # printer = Printer(file)
        # printer.print(module)
        self.module = module

    def is_leaf(self) -> bool:
        return False
