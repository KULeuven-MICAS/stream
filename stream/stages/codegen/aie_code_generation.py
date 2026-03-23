import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from copy import deepcopy
from itertools import product
from math import prod
from typing import cast

from snaxc.dialects.snax import NoneAttr
from snaxc.dialects.tsl import TSL
from xdsl.context import MLContext
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, MemRefType, ModuleOp, ShapedType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineDimExpr
from xdsl.parser import AffineMap
from xdsl_aie.dialects.aie import AIEDeviceEnum

from stream.compiler.dialects.stream import ComputationNodeOp, InEdgeOp, LayerDimAttr, OutEdgeOp, Stream, TransferOp

# from stream.compiler.transforms.aie_add_tracing_script import AIEAddTracingScript
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_stream_to_aie import ConvertStreamToAIEPass, canonicalize_transformation
from stream.compiler.transforms.stream_split_transfers import StreamSplitTransfersPass
from stream.compiler.transforms.stream_split_unicasts import StreamSplitUnicastsPass
from stream.datatypes import LayerDim
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping, NodeMapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.iteration_space import (
    IterationVariable,
    IterationVariableType,
    SteadyStateIterationSpace,
)
from stream.workload.workload import (
    ComputationNode,
    HasInputs,
    HasIterationSpace,
    HasOutputs,
    InEdge,
    Node,
    OutEdge,
    TransferNode,
    Workload,
)


class AIECodeGenerationStage(Stage):
    REQUIRED_FIELDS = tuple()

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        super().__init__(list_of_callables, ctx)

        # set up the correct xDSL context
        self.context: MLContext = MLContext()

        # add custom dialects and passes
        self.context.load_dialect(Stream)
        self.context.load_dialect(TSL)

        self.trace_size = self.ctx.get("trace_size", 1048576)
        self.npu = self.ctx.get("npu", "npu2")
        self.runtime_args = self.ctx.get("runtime_args", [])
        self.module = None

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)

        for ctx in sub_stage.run():
            self.ctx = ctx
            start = time.time()
            self.codegen_main()
            end = time.time()
            print(f"code generation took {end - start} seconds")
            assert self.module is not None
            self.ctx.set(module=self.module)
            yield self.ctx

    def create_transfer_op(  # noqa: PLR0912, PLR0915
        self,
        node: TransferNode,
        mapping: NodeMapping,
        full_mapping: Mapping,
        inputs: Sequence[Operation | SSAValue],
        ssis_dict: dict[HasIterationSpace, SteadyStateIterationSpace],
        workload: Workload,
    ) -> TransferOp:
        """
        Create a TransferOp for a given SteadyStateTransfer.
        """
        ssis = ssis_dict[node]
        if next_compute := next((n for n in workload.successors(node) if isinstance(n, ComputationNode)), None):
            ssis_dest = ssis_dict[next_compute]
        else:
            ssis_dest = ssis
        # The final transfer (pointing to an OutEdgeOp) requires special treatment
        is_out_transfer = any(isinstance(user, OutEdge) for user in workload.successors(node))

        # Get source and dest and convert to string for TransferOp creation which uses string
        # source = transfer.srcs[0]
        workload_strides = workload.strides_for_tensor(node.outputs[0])

        # To determine transfer type, iterate over kernel ssis vars
        transfer_shape = [0] * len(node.outputs[0].shape)
        for kernel_var in ssis.get_kernel_variables():
            for i, stride in enumerate(workload_strides[kernel_var.dimension]):
                transfer_shape[i] += stride * kernel_var.size

        assert isinstance(node.outputs[0].subview.source.type, MemRefType)

        # Determine strides based on layout mapping:
        # ordering of indeces:
        # TODO: make this more robust
        num_dims = node.outputs[0].subview.source.type.get_num_dims()
        tensor_name = node.name[len("Transfer(") : -1]
        if tensor_name in full_mapping.runtime_args:
            layout_mapping = cast(AffineMap, full_mapping.runtime_args[tensor_name])
        else:
            layout_mapping = AffineMap.identity(num_dims)
        # index order is the order of dimensions in layout. The last element of this list
        # is unrolled first
        index_order = [layout_mapping.results.index(AffineDimExpr(i)) for i in range(num_dims)]
        source_shape = node.outputs[0].subview.source.type.get_shape()
        strides = ShapedType.strides_for_shape([source_shape[i] for i in index_order])
        shape_multiplier = [strides[::-1][i] for i in index_order[::-1]]

        # To determine the number of (spatially) parallel transfers, iterate over spatial ssis vars of destination
        # the spatial variables must not be a spatio-temporal variable of the desintation
        next_spatio_temporal_vars = [v.dimension for v in ssis_dest.get_spatio_temporal_variables()]
        relevant_spat_vars = [
            v for v in ssis.get_spatial_variables() if v.relevant and v.dimension not in next_spatio_temporal_vars
        ]
        num_spat_results = prod(v.size for v in relevant_spat_vars)

        # Move determination of sizes/strides for runtime compies over to lowering logic,
        # here, just set sizes and strides to the input layout such that correct calculations can be made.
        operand_index_dims = [
            x[0] for x in sorted(workload_strides.items(), key=lambda x: x[1], reverse=True) if any(x[1])
        ]
        operand_attr = ArrayAttr([LayerDimAttr(x) for x in operand_index_dims])

        # Determine the output type based on this:
        if is_out_transfer:
            result_type = node.outputs[0].subview.source.type
            result_types = (result_type,)
        else:
            result_type = MemRefType(node.outputs[0].operand_type, transfer_shape)
            # Unroll spatial results: [(s0, 4), (s1, 4)] should give a 4x4 array of results = 16 results
            result_types = (result_type,) * num_spat_results

        # Determine the spatial strides for this transfer:
        # Spatial strides are simply determined in a linear fashion, each time prgressing
        # the number of elements that one transfer type takes.
        transfer_elements = prod(transfer_shape)
        # spatio_temporal_elements = prod(v.size for v in ssis_dest.get_spatio_temporal_variables())
        # spatial_stride = transfer_elements * spatio_temporal_elements
        if is_out_transfer:
            st_factor = num_spat_results // len(inputs)
            spatial_strides = tuple(range(0, transfer_elements * num_spat_results, transfer_elements * st_factor))
        elif len(inputs) == 1:
            spatial_strides = tuple(range(0, transfer_elements * num_spat_results, transfer_elements))
        elif num_spat_results == 1:
            spatial_strides = tuple(range(0, transfer_elements * len(inputs), transfer_elements))
        else:
            # FIXME: spatial strides can apply to both input and output but it is not very clear right now
            spatial_strides = tuple(range(0, transfer_elements * num_spat_results, transfer_elements))
            # raise NotImplementedError("The case with multiple inputs and multiple spatial results is not implemented yet")

        # Determine the temporal strides for this transfer:
        seen_dims = defaultdict(lambda: 1)
        # this assumes that each resulting tile after tiling to be in row-major layout
        # this could probably benefit from a more holistic view of layout transformation
        #
        if is_out_transfer:
            ssis_prev = ssis_dict[next(workload.predecessors(node))]
            new_ssis = []
            new_ssis_tvars = []
            for cur, prev in zip(ssis.variables, ssis_prev.variables, strict=True):
                if cur.type == prev.type:
                    if cur.type == IterationVariableType.TEMPORAL:
                        new_ssis_tvars.append(deepcopy(cur))
                    else:
                        new_ssis.append(deepcopy(cur))
                else:
                    assert cur.type == IterationVariableType.SPATIAL
                    assert prev.type == IterationVariableType.SPATIOTEMPORAL
                    new_var = deepcopy(cur)
                    new_var.type = IterationVariableType.TEMPORAL
                    new_ssis_tvars.append(new_var)
            ssis = SteadyStateIterationSpace([*new_ssis, *new_ssis_tvars])

        all_vars: Sequence[IterationVariable] = []
        # First, iterate over kernel dimensions in row-major order:
        kernel_var_dict = {v.dimension: v for v in ssis.get_kernel_variables()}
        filtered_vars = [(dim, strides) for dim, strides in workload_strides.items() if any(strides)]
        ordered_strides = sorted(filtered_vars, key=lambda x: x[1])
        ordered_strides = [ordered_strides[i] for i in index_order]
        all_vars.extend(kernel_var_dict[dim] for dim, _ in ordered_strides)

        # First, we should go over the temporal vars that are kept local in mem tile:
        reuse_tvars = []
        for var in ssis.get_temporal_variables():
            if var.mem_tile_reuse == MemTileReuse.REUSE:
                reuse_tvars.append(var)
            else:
                break
        non_reuse_tvars = ssis.get_temporal_variables()[len(reuse_tvars) :]
        all_vars.extend(var for var in reuse_tvars if var.relevant)
        # Then, iterate over relevant spatial vars:
        all_vars.extend(var for var in ssis.get_spatial_variables() if var.relevant)
        # This is only relevant for first and last ops, which should not have spatio-temporal vars.
        # After that, go over the temporal strides (both relevant and irellevant)
        # that aren't kept local in memtiles.
        # Then, add all remaining temporal variables that aren't kept local in the memtile
        all_vars.extend(var for var in non_reuse_tvars if var.applicable)

        # I dont' think offsets are relevant anymore with the new representation
        offsets = [0]

        # Construct sizes and strides
        sizes = []
        strides = []
        for var in all_vars:
            sizes.insert(0, var.size)
            stride = workload_strides[var.dimension]
            # multiply the stride by previous iteration vars
            stride = tuple(seen_dims[var.dimension] * x for x in stride)
            seen_dims[var.dimension] *= var.size
            # convert stride to int (assumes dram row-major layout):
            stride = sum(x * y for (x, y) in zip(stride, shape_multiplier, strict=True))
            strides.insert(0, stride)

        sizes, strides = canonicalize_transformation(sizes, strides)

        if len(mapping.memory_allocation):
            row, col = mapping.memory_allocation[0].row_id, mapping.memory_allocation[0].col_id
            assert row is not None
            assert col is not None
            if is_out_transfer:
                row = 1
                col = 7
                memtile = ArrayAttr(
                    [
                        ArrayAttr([IntegerAttr.from_index_int_value(6), IntegerAttr.from_index_int_value(1)]),
                        ArrayAttr([IntegerAttr.from_index_int_value(7), IntegerAttr.from_index_int_value(1)]),
                    ]
                )
            else:
                memtile = ArrayAttr([ArrayAttr([IntegerAttr.from_index_int_value(x) for x in (col, row)])])
        else:
            memtile = NoneAttr()

        op = TransferOp(
            inputs,
            result_types,
            ssis,
            offsets,
            sizes,
            strides,
            spatial_strides,
            memtile,
            operand_attr,
        )

        return op

    def create_computation_node_op(
        self,
        node: ComputationNode,
        mapping: NodeMapping,
        full_mapping: Mapping,
        inputs: Sequence[Operation | SSAValue],
        ssis: SteadyStateIterationSpace,
        workload: Workload,
    ) -> Sequence[ComputationNodeOp]:
        ops: list[ComputationNodeOp] = []

        # add spatio-temporal dims to get only inner shape:
        st_vars = ssis.get_spatio_temporal_variables()
        st_var: tuple[tuple[LayerDim, int], ...]
        assert len(st_vars) <= 1
        if len(st_vars):
            st_var = ((st_vars[0].dimension, st_vars[0].size),)
        else:
            st_var = tuple()
        # determine new result type based on spatial mapping
        shape = workload.get_tensor_shape_with_tiling(
            node.output, tuple(workload.get_unique_dims_inter_core_tiling(node, full_mapping)) + st_var
        )

        result_type = MemRefType(node.outputs[0].operand_type, shape)
        result_type = MemRefType(node.output.operand_type, shape)
        workload.global_mapping(node, node.operand_mapping[-1])
        # Spatial: (m, 4), (n, 4)
        # step 1: get iterable over all combinations with [(LayerDim, value), (LayerDim, value)] as data
        ranges = [
            [(spat_var.dimension, x) for x in range(spat_var.size)]
            for spat_var in ssis.get_spatial_variables()
            if spat_var.applicable
        ]

        # step 2:
        combined_ranges = list(product(*reversed(ranges)))
        for core, comb_ran in zip(mapping.resource_allocation, combined_ranges, strict=True):
            selected_inputs = []
            for input in inputs:
                assert isinstance(input, TransferOp)
                selected_inputs.append(
                    input.get_relevant_output(comb_ran, [x.dimension for x in ssis.get_spatio_temporal_variables()])
                )

            assert isinstance(core, Core)
            row, col = core.row_id, core.col_id
            assert row is not None
            assert col is not None
            core_allocation = ArrayAttr([IntegerAttr.from_index_int_value(x) for x in (col, row)])
            assert mapping.kernel is not None
            op = ComputationNodeOp((result_type,), mapping.kernel.unique_name, selected_inputs, core_allocation, ssis)
            ops.append(op)

        return ops

    def generate_steady_state_workload(
        self, workload: Workload, mapping: Mapping, ssis_dict: dict[HasIterationSpace, SteadyStateIterationSpace]
    ) -> ModuleOp:
        ops: dict[Node, Sequence[Operation]] = {}
        # edge_ops: dict[SteadyStateTensor, InEdgeOp | OutEdgeOp] = {}
        # transfer_ops: dict[SteadyStateTransfer, TransferOp] = {}
        # compute_ops: dict[SteadyStateComputation, ComputationNodeOp] = {}

        # all_ops: list[Operation] = []
        def inputs(node: HasInputs) -> Iterator[Operation]:
            for inp in node.inputs:
                for other_node in workload.nodes:
                    if isinstance(other_node, HasOutputs) and inp in other_node.outputs:
                        yield from ops[other_node]

        for node in workload.topological_sort():
            if isinstance(node, InEdge):
                ops[node] = [InEdgeOp(node)]
            if isinstance(node, OutEdge):
                inps = tuple(inputs(node))
                assert len(inps) == 1
                inp = inps[0]
                ops[node] = [OutEdgeOp(node, inp.results)]
            if isinstance(node, TransferNode):
                ops[node] = [
                    self.create_transfer_op(
                        node,
                        mapping.get(node),
                        mapping,
                        tuple(inputs(node)),
                        ssis_dict,
                        workload,
                    )
                ]
            if isinstance(node, ComputationNode):
                ops[node] = self.create_computation_node_op(
                    node, mapping.get(node), mapping, tuple(inputs(node)), ssis_dict[node], workload
                )

        all_ops = []
        for opsx in ops.values():
            all_ops.extend(opsx)

        module = ModuleOp(all_ops)

        return module

    def codegen_main(self) -> None:
        workload: Workload = self.ctx.get("workload")
        trace_size: int = self.ctx.get("trace_size", 1048576)  # noqa: F841
        npu: AIEDeviceEnum = self.ctx.get("npu", "npu2")
        assert workload is not None

        mapping = self.ctx.get("scheduler").mapping
        aie_kernels = {nm.kernel.unique_name: nm.kernel for nm in mapping.values() if nm.kernel is not None}
        assert isinstance(mapping, Mapping)

        ssis_dict = self.ctx.get("scheduler").ssis

        module = self.generate_steady_state_workload(workload, mapping, ssis_dict)

        with open("test.mlir", "w") as f:
            f.write(str(module))

        StreamSplitUnicastsPass().apply(self.context, module)

        with open("test2.mlir", "w") as f:
            f.write(str(module))
        # SetNoReusePass().apply(self.context, module)
        # Split transfers in push and pull
        StreamSplitTransfersPass().apply(self.context, module)

        with open("test3.mlir", "w") as f:
            f.write(str(module))

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

        self.module = module

    def is_leaf(self) -> bool:
        return False
