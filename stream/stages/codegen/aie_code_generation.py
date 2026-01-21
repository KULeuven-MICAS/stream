from collections import defaultdict
from collections.abc import Iterator, Sequence
from itertools import accumulate, product
from operator import mul

from snaxc.dialects.snax import NoneAttr
from snaxc.dialects.tsl import TSL
from xdsl.context import MLContext
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, MemRefType, ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl_aie.dialects.aie import AIEDeviceEnum

from stream.compiler.dialects.stream import ComputationNodeOp, InEdgeOp, OutEdgeOp, Stream, TransferOp

# from stream.compiler.transforms.aie_add_tracing_script import AIEAddTracingScript
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_stream_to_aie import ConvertStreamToAIEPass, canonicalize_transformation
from stream.compiler.transforms.stream_split_transfers import StreamSplitTransfersPass
from stream.compiler.transforms.stream_split_unicasts import StreamSplitUnicastsPass
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping, NodeMapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.iteration_space import (
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
        self.mapping = self.ctx.get("mapping")
        self.module = None

    def run(self):
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)

        for ctx in sub_stage.run():
            self.ctx = ctx
            self.codegen_main()
            assert self.module is not None
            self.ctx.set(module=self.module)
            yield self.ctx

    def create_transfer_op(
        self,
        node: TransferNode,
        mapping: NodeMapping,
        full_mapping: Mapping,
        inputs: Sequence[Operation | SSAValue],
        ssis: SteadyStateIterationSpace,
        workload: Workload,
    ) -> TransferOp:
        """
        Create a TransferOp for a given SteadyStateTransfer.
        """
        # Get source and dest and convert to string for TransferOp creation which uses string
        # source = transfer.srcs[0]
        workload_strides = workload.strides_for_tensor(node.outputs[0])

        result_types = []
        spatial_strides = []
        shape = workload.get_tensor_shape_of_transfer_to_single_core(node.outputs[0], node, full_mapping)
        result_type = MemRefType(node.outputs[0].operand_type, shape)
        # TODO: this only works for max 1 spatial stride
        # copy something like for computation node
        # TODO: this should be ssis_dest, (which should be None for an edge op)
        full_size = None
        min_size = None

        for spat_stride in ssis.get_spatial_variables():
            if spat_stride.relevant:
                result_types.extend((result_type,) * spat_stride.size)
                spatial_strides.append(workload_strides[spat_stride.dimension])
        if len(result_types) == 0:
            result_types = (result_type,)
        if any(isinstance(user, OutEdge) for user in workload.successors(node)):
            result_types = node.outputs[0].subview.source.type
            assert isinstance(inputs[0], ComputationNodeOp)
            assert inputs[0].result is not None
            assert isinstance(inputs[0].result.type, MemRefType)
            shape = inputs[0].result.type.get_shape()

        offsets = []
        sizes = []
        strides = []

        def reverse_cumprod(t):
            return tuple(reversed(list(accumulate(reversed(t), mul, initial=1))[:-1]))

        assert isinstance(node.outputs[0].subview.source.type, MemRefType)
        shape_multiplier = reverse_cumprod(node.outputs[0].subview.source.type.get_shape())

        # start by adding the shape of the destination
        sizes.extend(shape)
        strides.extend(shape_multiplier)

        seen_dims = defaultdict(lambda: 1)
        for v in ssis.variables:
            if v.relevant and v.type is not IterationVariableType.KERNEL:
                sizes.insert(0, v.size)
                stride = workload_strides[v.dimension]
                # multiply the stride by previous iteration vars
                stride = tuple(seen_dims[v.dimension] * x for x in stride)
                seen_dims[v.dimension] *= v.size
                # convert stride to int:
                stride = sum(x * y for (x, y) in zip(stride, shape_multiplier, strict=True))
                strides.insert(0, stride)

        sizes, strides = canonicalize_transformation(sizes, strides)

        if isinstance(mapping.memory_allocation, Core):
            row, col = mapping.memory_allocation.row_id, mapping.memory_allocation.col_id
            assert row is not None
            assert col is not None
            memtile = ArrayAttr([IntegerAttr.from_index_int_value(x) for x in (col, row)])
        else:
            memtile = NoneAttr()

        # Convert strides to integers:
        def strides_to_int(strides: Sequence[int], shape: Sequence[int]):
            result = 0
            mult = 1
            for stride, size in zip(reversed(strides), reversed(shape), strict=True):
                result += stride * mult
                mult *= size
            return result

        # somehow, get the shape of the input for now, take shape of first output
        shape = node.outputs[0].shape
        if len(spatial_strides) > 0:
            spatial_strides = list(map(lambda x: strides_to_int(x, shape), spatial_strides))
        else:
            spatial_strides = [0]

        op = TransferOp(
            inputs,
            result_types,
            ssis,
            ssis,
            offsets,
            sizes,
            strides,
            spatial_strides,
            memtile,
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

        # determine new result type based on spatial mapping
        shape = workload.get_tensor_shape_with_tiling(
            node.output, workload.get_unique_dims_inter_core_tiling(node, full_mapping)
        )

        result_type = MemRefType(node.outputs[0].operand_type, shape)
        result_type = MemRefType(node.output.operand_type, shape)
        workload.global_mapping(node, node.operand_mapping[-1])
        # Spatial: (m, 4), (n, 4)
        # step 1: get iterable over all combinations with [(LayerDim, value), (LayerDim, value)] as data
        ranges = [[(spat_var.dimension, x) for x in range(spat_var.size)] for spat_var in ssis.get_spatial_variables()]
        # step 2:
        combined_ranges = list(product(*ranges))
        for core, comb_ran in zip(mapping.resource_allocation, combined_ranges, strict=True):
            selected_inputs = []
            for input in inputs:
                assert isinstance(input, TransferOp)
                selected_inputs.append(input.get_relevant_output(comb_ran))

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
                        node, mapping.get(node), mapping, tuple(inputs(node)), ssis_dict[node], workload
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

        mapping = self.ctx.get("mapping")
        aie_kernels = {nm.kernel.unique_name: nm.kernel for nm in mapping.values() if nm.kernel is not None}
        assert isinstance(mapping, Mapping)

        ssis_dict = self.ctx.get("scheduler").ssis

        module = self.generate_steady_state_workload(workload, mapping, ssis_dict)

        StreamSplitUnicastsPass().apply(self.context, module)

        with open("test.mlir", "w") as f:
            f.write(str(module))

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

        self.module = module

    def is_leaf(self) -> bool:
        return False
