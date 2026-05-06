import os
import time
from collections.abc import Iterable, Sequence
from typing import cast
from warnings import warn

from snaxc.dialects.tsl import TSL
from xdsl.dialects.builtin import (
    ModuleOp,
    ShapedType,
)
from xdsl.ir import Block, Operation, OpResult, Region, SSAValue
from xdsl.parser import StringAttr

from stream.compiler.context.aie_context import AIEContext
from stream.compiler.dialects.stream import (
    ComputationNodeOp,
    FusionGroupOp,
    InEdgeOp,
    OutEdgeOp,
    Stream,
    StrensorSpace,
    StrensorType,
    StrensorVar,
    StrensorVarType,
    TransferOp,
    YieldOp,
)
from stream.compiler.transforms.aie_convert_ofs import AIEConvertOfs

# from stream.compiler.transforms.aie_add_tracing_script import AIEAddTracingScript
from stream.compiler.transforms.aie_dispatch import AIEDispatchPass
from stream.compiler.transforms.aie_move_tile_ops_up import AIEMoveTileOpsUp
from stream.compiler.transforms.clear_memory_space import ClearMemorySpace
from stream.compiler.transforms.convert_stream_to_aie import (
    ConvertStreamToAIEPass,
)
from stream.compiler.transforms.iteration_space_to_for import IterationSpaceToFor
from stream.compiler.transforms.unroll import SpatialUnrollPass
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.datatypes import LayerDim
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping, NodeMapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.steady_state.iteration_space import (
    IterationVariableType,
    LoopEffect,
    Reuse,
    SteadyStateIterationSpace,
)
from stream.workload.tensor import Tensor
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
        self.context: AIEContext = AIEContext()

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
        inputs: Sequence[SSAValue],
        ssis_dict: dict[HasIterationSpace | Tensor, SteadyStateIterationSpace],
        workload: Workload,
        ss: StrensorSpace,
        reuse_index: int,
    ) -> TransferOp:
        """
        Create a TransferOp for a given SteadyStateTransfer.
        """

        input_type = SSAValue.get(inputs[0]).type
        assert isinstance(input_type, StrensorType)
        path_plan = mapping.resource_allocation[0]
        assert isinstance(path_plan, MulticastPathPlan)
        cores = []
        for target in path_plan.targets:
            assert target.row_id is not None
            assert target.col_id is not None
            cores.append(StringAttr(f"tile_{target.col_id}_{target.row_id}"))

        # determine reuse index
        # FIXME: this completely ignores constraint optimization because the reuse index there does not make sense

        if input_type.reuse_index.data > 0:
            # make destination stationary because we're not retransmitting
            if (
                isinstance(inputs[0], OpResult)
                and isinstance(inputs[0].op, ComputationNodeOp)
                and isinstance(next(workload.successors(node)), ComputationNode)
            ):
                relevant_dims = {var.dim for var in ss.get_kernel_variables()}

                # just make sure we are output stationary
                for i, var in enumerate(ss.vars):
                    reuse_index = len(ss.vars) - i
                    if var.type == StrensorVarType.TEMPORAL and var.dim not in relevant_dims:
                        break
                    if var.type == StrensorVarType.KERNEL:
                        break
            elif isinstance(inputs[0], OpResult) and isinstance(inputs[0].op, TransferOp):
                # remove unnecessary reuse
                relevant_dims = {var.dim for var in ss.get_kernel_variables()}

                # just make sure we are output stationary
                for i, var in enumerate(ss.vars[-input_type.reuse_index.data :]):
                    reuse_index = input_type.reuse_index.data - i
                    if var.type == StrensorVarType.TEMPORAL and var.dim not in relevant_dims:
                        break
                    if var.type == StrensorVarType.KERNEL:
                        break
            else:
                # keep reuse index of input if available
                reuse_index = input_type.reuse_index.data
        # else we are input transfer for memtile, cover spatial of destination
        elif isinstance((next_t := next(workload.successors(node))), TransferNode):
            next_ssis = ssis_dict[next_t]
            total_spatial = (
                next_ssis.variables.index(next_ssis.get_spatial_variables()[-1])
                - next_ssis.variables.index(next_ssis.get_spatial_variables()[0])
                + 1
            )
            count_spatial = 0
            new_reuse_index = 0
            for i, var in enumerate(reversed(ss.vars)):
                if var.type in (StrensorVarType.SPATIAL, StrensorVarType.TEMPORAL):
                    count_spatial += 1
                if count_spatial == total_spatial:
                    new_reuse_index = i + 1
                    break
            reuse_index = new_reuse_index

        # FIXME: end

        if len(cores) == 1:
            # FIXME: hardcoded fix:
            if any(x.type == StrensorVarType.SPATIAL for x in ss.vars):
                warn("hardcoding spatial loop to be absent because of wrong input", stacklevel=2)
                ss_new = StrensorSpace(
                    tuple(
                        StrensorVar(StrensorVarType.ABSENT, x.size, x.dim) if x.type == StrensorVarType.SPATIAL else x
                        for x in ss.vars
                    )
                )
                ss = ss_new

        if cores == [StringAttr("tile_0_0")]:
            shape = cast(ShapedType, node.output.subview.source.type).get_shape()
            layer_dims = (
                x[0]
                for x in sorted(workload.strides_for_tensor(node.outputs[0]).items(), key=lambda x: x[1], reverse=True)
                if any(x[1])
            )
            result_type = StrensorType(
                node.output.operand_type,
                StrensorSpace(
                    tuple(StrensorVar(StrensorVarType.CONSTANT, s, d) for (s, d) in zip(shape, layer_dims, strict=True))
                ),
                [StringAttr("tile_0_0")],
            )
            result_types = [result_type]
        elif len(node.outputs) > 1:
            # create equal split based on compute allocations
            cores_per_output = len(cores) // len(node.outputs)
            result_types = []
            for i in range(len(node.outputs)):
                result_types.append(
                    StrensorType(
                        input_type.element_type,
                        ss,
                        cores[i * cores_per_output : (i + 1) * cores_per_output],
                        reuse_index,
                    )
                )
        else:
            result_type = StrensorType(
                input_type.element_type,
                ss,
                cores,
                reuse_index,
            )
            result_types = [result_type]
        op = TransferOp(
            inputs,
            result_types,
            #     ssis,
            #     offsets,
            #     sizes,
            #     strides,
            #     spatial_strides,
            #     memtile,
            #     operand_attr,
        )

        return op

    def create_computation_node_op(  # noqa: PLR0913
        self,
        node: ComputationNode,
        mapping: NodeMapping,
        full_mapping: Mapping,
        inputs: Sequence[Operation | SSAValue],
        ssis: SteadyStateIterationSpace,
        ssis_dict,
        workload: Workload,
        ss: StrensorSpace,
        reuse_index: int,
    ) -> ComputationNodeOp:
        # FIXME: recomputes reuse index because constraint optimization seems broken
        relevant_dims = {var.dim for var in ss.get_kernel_variables()}

        # just make sure we are output stationary
        for i, var in enumerate(ss.vars):
            reuse_index = len(ss.vars) - i
            if var.type == StrensorVarType.TEMPORAL and var.dim not in relevant_dims:
                break
            if var.type == StrensorVarType.KERNEL:
                break

        # FIXME: end

        # # add spatio-temporal dims to get only inner shape:
        cores = mapping.resource_allocation[0]
        cores_attrs = []
        for core in cores:
            assert isinstance(core, Core)
            assert core.row_id is not None
            assert core.col_id is not None
            cores_attrs.append(StringAttr(f"tile_{core.col_id}_{core.row_id}"))

        result_type = StrensorType(
            node.output.operand_type,
            ss,
            cores_attrs,
            reuse_index,
        )
        # workload.global_mapping(node, node.operand_mapping[-1])
        # # Spatial: (m, 4), (n, 4)
        # # step 1: get iterable over all combinations with [(LayerDim, value), (LayerDim, value)] as data
        # ranges = [
        #     [(spat_var.dimension, x) for x in range(spat_var.size)]
        #     for spat_var in ssis.get_spatial_variables()
        #     if spat_var.applicable
        # ]
        #
        # # step 2:
        # combined_ranges = list(product(*reversed(ranges)))
        # for core, comb_ran in zip(mapping.resource_allocation, combined_ranges, strict=True):
        #     selected_inputs = []
        #     for input in inputs:
        #         assert isinstance(input, TransferOp)
        #         selected_inputs.append(
        #             input.get_relevant_output(comb_ran, [x.dimension for x in ssis.get_spatio_temporal_variables()])
        #         )
        #
        #     assert isinstance(core, Core)
        #     row, col = core.row_id, core.col_id
        #     assert row is not None
        #     assert col is not None
        #     core_allocation = ArrayAttr([IntegerAttr.from_index_int_value(x) for x in (col, row)])
        assert mapping.kernel is not None
        return ComputationNodeOp(inputs, (result_type,), mapping.kernel.unique_name)

    def generate_steady_state_workload(  # noqa: PLR0915
        self,
        workload: Workload,
        mapping: Mapping,
        ssis_dict: dict[HasIterationSpace | Tensor, SteadyStateIterationSpace],
    ) -> ModuleOp:
        ops: dict[Node, Operation] = {}

        def get_layer_dims(tensor: Tensor) -> Iterable[LayerDim]:
            strides = workload.strides_for_tensor(tensor)
            filtered = {y: x for x, y in strides.items() if any(y)}
            return (filtered[x] for x in sorted(filtered))

        def get_kernel_size(tensor: Tensor) -> Sequence[int]:
            ssis = ssis_dict[tensor]
            kernel_vars = {var.dimension: var for var in ssis.get_kernel_variables() if var.relevant}
            shape: list[int] = []
            for dim in get_layer_dims(tensor):
                shape.append(kernel_vars[dim].size)
            return shape

        def ssis_to_strensorspace(tensor: Tensor) -> tuple[int, StrensorSpace]:
            # kernel vars:
            vars: list[StrensorVar] = []
            for size, dim in zip(get_kernel_size(tensor), get_layer_dims(tensor), strict=True):
                vars.insert(0, StrensorVar(StrensorVarType.KERNEL, size, dim))
            reuse_index = len(vars)
            for var in ssis_dict[tensor].variables:
                if var.effect == LoopEffect.ABSENT:
                    vars.insert(0, StrensorVar(StrensorVarType.ABSENT, var.size, var.dimension))
                elif var.type == IterationVariableType.SPATIAL:
                    vars.insert(0, StrensorVar(StrensorVarType.SPATIAL, var.size, var.dimension))
                elif var.type in (IterationVariableType.TEMPORAL, IterationVariableType.SPATIOTEMPORAL):
                    vars.insert(0, StrensorVar(StrensorVarType.TEMPORAL, var.size, var.dimension))
                if var.reuse == Reuse.REUSE:
                    reuse_index = len(vars)

            return reuse_index, StrensorSpace(tuple(vars))

        # all_ops: list[Operation] = []
        def inputs(node: HasInputs) -> Iterable[SSAValue]:
            for inp in node.inputs:
                for other_node in workload.nodes:
                    if isinstance(other_node, HasOutputs) and inp in other_node.outputs:
                        yield ops[other_node].results[other_node.outputs.index(inp)]

        for node in workload.topological_sort():
            if isinstance(node, InEdge):
                shape = cast(ShapedType, node.output.subview.source.type).get_shape()
                layer_dims = reversed(tuple(get_layer_dims([x for x in workload.successors(node)][0].output)))
                ops[node] = InEdgeOp(
                    node.name,
                    StrensorType(
                        node.output.operand_type,
                        StrensorSpace(
                            tuple(
                                StrensorVar(StrensorVarType.CONSTANT, s, d)
                                for (s, d) in zip(shape, layer_dims, strict=True)
                            )
                        ),
                        [StringAttr("tile_0_0")],
                    ),
                )
            if isinstance(node, OutEdge):
                inps = tuple(inputs(node))
                assert len(inps) == 1
                inp = inps[0]
                ops[node] = OutEdgeOp(node, (inp,))
            if isinstance(node, TransferNode):
                reuse_index, ss = ssis_to_strensorspace(node.outputs[0])
                ops[node] = self.create_transfer_op(
                    node,
                    mapping.get(node),
                    mapping,
                    tuple(inputs(node)),
                    ssis_dict,
                    workload,
                    ss,
                    reuse_index,
                )
            if isinstance(node, ComputationNode):
                reuse_index, ss = ssis_to_strensorspace(node.outputs[0])
                ops[node] = self.create_computation_node_op(
                    node,
                    mapping.get(node),
                    mapping,
                    tuple(inputs(node)),
                    ssis_dict[node],
                    ssis_dict,
                    workload,
                    ss,
                    reuse_index,
                )

        types = []
        remaining_ops = []
        in_edges: list[InEdgeOp] = []
        for op in ops.values():
            if isinstance(op, InEdgeOp):
                types.append(op.output.type)
                in_edges.append(op)
            elif isinstance(op, OutEdgeOp):
                types.append(op.inputs[0].type)
                remaining_ops.append(YieldOp(*op.inputs))
                op.erase()
            else:
                remaining_ops.append(op)

        fusion_group = FusionGroupOp(Region(block := Block(remaining_ops, arg_types=types)))

        for in_edge, block_arg in zip(in_edges, block.args, strict=False):
            in_edge.output.replace_by(block_arg)
            in_edge.erase()

        module = ModuleOp([fusion_group])

        return module

    def codegen_main(self) -> None:
        workload: Workload = self.ctx.get("workload")
        trace_size: int = self.ctx.get("trace_size", 1048576)  # noqa: F841
        assert workload is not None
        mapping = self.ctx.get("scheduler").mapping
        for kernel in (nm.kernel for nm in mapping.values() if nm.kernel is not None):
            self.context.registered_kernels[kernel.unique_name] = kernel

        assert isinstance(mapping, Mapping)

        ssis_dict: dict[Tensor, SteadyStateIterationSpace] = self.ctx.get("scheduler").ssis

        keys = [
            "left_swished",
            "left_swished_1",
            "intermediate_1",
            "output",
            "output_2",
            "output_1",
        ]
        keys = [x for x in ssis_dict.keys() if x.name in keys]
        for key in keys:
            ssis = ssis_dict[key]
            for var in ssis.variables:
                if var.dimension == LayerDim(1) and var.type == IterationVariableType.SPATIAL:
                    var.type = IterationVariableType.TEMPORAL

        module = self.generate_steady_state_workload(workload, mapping, ssis_dict)

        self.module = module

        # with open("test1.mlir", "w") as f:
        #     f.write(str(module))

        output_path = self.ctx.data["output_path"]
        output_path += "/codegen/"
        os.makedirs(output_path, exist_ok=True)

        with open(output_path + "/stream.mlir", "w") as f:
            f.write(str(module))
        module.verify()
        # Lowering Passes:
        SpatialUnrollPass().apply(self.context, module)
        with open(output_path + "/unrolled.mlir", "w") as f:
            f.write(str(module))
        AIEDispatchPass().apply(self.context, module)
        with open(output_path + "/dispatched.mlir", "w") as f:
            f.write(str(module))
        IterationSpaceToFor().apply(self.context, module)
        with open(output_path + "/with_for.mlir", "w") as f:
            f.write(str(module))
        AIEConvertOfs().apply(self.context, module)
        with open(output_path + "/convert_of.mlir", "w") as f:
            f.write(str(module))
        ConvertStreamToAIEPass().apply(self.context, module)
        with open(output_path + "/to_aie.mlir", "w") as f:
            f.write(str(module))
        AIEMoveTileOpsUp().apply(self.context, module)
        ClearMemorySpace().apply(self.context, module)
        with open(output_path + "/final.mlir", "w") as f:
            f.write(str(module))

        # Optionally, Add Tracing Script
        # if False:
        # AIEAddTracingScript(trace_size=trace_size).apply(self.context, module)

        self.module = module

    def is_leaf(self) -> bool:
        return False
