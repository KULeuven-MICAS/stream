from collections.abc import Sequence

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType, StringAttr
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl_aie.dialects.aie import EndOp

from stream.compiler.dialects.stream import ComputationNodeOp, PullOp, PushOp
from stream.workload.steady_state.iteration_space import IterationVariable, IterationVariableType


def iteration_space_to_for(block: Block, rewriter: Rewriter):  # noqa: PLR0912, PLR0915
    ops: Sequence[PushOp | PullOp | ComputationNodeOp] = []
    ssis = None
    for op in block.ops:
        if isinstance(op, ComputationNodeOp):
            ssis = op.ssis.data
        if isinstance(op, PushOp | PullOp | ComputationNodeOp):
            ops.append(op)
        elif isinstance(op, EndOp):
            pass
        else:
            raise RuntimeError("non-steady state op encountered")
    assert ssis is not None

    # some common for loop variables
    lb = ConstantOp.from_int_and_width(0, IndexType())
    step = ConstantOp.from_int_and_width(1, IndexType())
    rewriter.insert_op((lb, step), InsertPoint.at_start(block))

    # create for loop nest:
    for_ops = []
    for_dict: dict[IterationVariable, ForOp] = {}
    vars: list[IterationVariable] = []
    var_selector: list[bool] = []
    for var in ssis.variables:
        if var.type not in (IterationVariableType.TEMPORAL, IterationVariableType.SPATIOTEMPORAL):
            if var.type != IterationVariableType.KERNEL:
                var_selector.append(False)
            continue
        vars.append(var)
        var_selector.append(True)
        # If not relevant, set bound to 1 such that it is canonicalized away
        if var.relevant:
            ub = ConstantOp.from_int_and_width(var.size, IndexType())
        else:
            ub = ConstantOp.from_int_and_width(1, IndexType())
        for_op = ForOp(lb, ub, step, [], Block([*for_ops, YieldOp()], arg_types=[IndexType()]))
        for_op.attributes["layer_dim"] = StringAttr(str(var.dimension))
        for_ops = [ub, for_op]
        for_dict[var] = for_op

    # insert for loop nest
    rewriter.insert_op(for_ops, InsertPoint.after(step))

    for op in ops:
        non_kernel_vars = [var for var in op.ssis.data.variables if var.type != IterationVariableType.KERNEL]
        these_vars = [var for var, selector in zip(non_kernel_vars, var_selector, strict=True) if selector]
        # these_vars = [*op.ssis.data.get_spatio_temporal_variables(), *op.ssis.data.get_temporal_variables()]
        # # get rid of spatio temporal vars that are not present in compute
        # these_vars = these_vars[-len(vars) :]
        # if len(these_vars) != len(vars):
        #     # FIXME: bring back ssis dest for push / pull ops
        #     assert vars[0].type is IterationVariableType.SPATIOTEMPORAL
        #     spatial_var = next(x for x in op.ssis.data.get_spatial_variables() if x.dimension == vars[0].dimension)
        #     these_vars.insert(0, spatial_var)
        # find first relevant one
        first_relevant = next((v for v in these_vars if v.relevant), None)
        if not first_relevant:
            # no relevant vars, put around last for loop
            before = InsertPoint.before(for_dict[vars[-1]])
            after = InsertPoint.after(for_dict[vars[-1]])
        else:
            for_op = for_dict[vars[these_vars.index(first_relevant)]]
            before = InsertPoint.at_start(for_op.body.block)
            assert isinstance(for_op.body.block.last_op, YieldOp)
            after = InsertPoint.before(for_op.body.block.last_op)
        if isinstance(op, PullOp):
            op.detach()
            rewriter.insert_op(op, before)
        else:
            op.detach()
            rewriter.insert_op(op, after)
