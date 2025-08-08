from collections.abc import Sequence

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType, StringAttr
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl_aie.dialects.aie import EndOp

from stream.compiler.dialects.stream import ComputationNodeOp, PullOp, PushOp


def iteration_space_to_for(block: Block, rewriter: Rewriter):  # noqa: PLR0912, PLR0915
    ops: Sequence[PushOp | PullOp | ComputationNodeOp] = []
    ssis = None
    for op in block.ops:
        if isinstance(op, PushOp | PullOp | ComputationNodeOp):
            ssis = op.ssis.data
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
    insertion_point = InsertPoint.after(step)

    # iterate over the steady state ops
    ops_iter = iter(ops)
    op = next(ops_iter, None)
    assert op is not None
    # get reversed list of iteration variables, from outermost to innermost
    ssis = op.ssis.data.get_temporal_variables()[::-1]
    ssis_op = op.ssis.data.get_temporal_variables()[::-1]
    for_op = None
    for i, iter_var in enumerate(ssis):
        # if not any of the following ops are relevant
        while op is not None and not any(iv.relevant for iv in ssis_op[i:]):
            op.detach()
            rewriter.insert_op(op, insertion_point)
            insertion_point = InsertPoint.after(op)
            op = next(ops_iter, None)
            if op is not None:
                ssis_op = op.ssis.data.get_temporal_variables()[::-1]
        # create for loop
        ub = ConstantOp.from_int_and_width(iter_var.size, IndexType())
        for_op = ForOp(lb, ub, step, [], Block([yield_op := YieldOp()], arg_types=[IndexType()]))
        for_op.attributes["layer_dim"] = StringAttr(iter_var.dimension.name)
        rewriter.insert_op((ub, for_op), insertion_point)
        insertion_point = InsertPoint.before(yield_op)
    assert for_op is not None
    # iterate in reverse now:
    # now, a non reversed version of the ssis:
    if op is None:
        return
    ssis = op.ssis.data.get_temporal_variables()
    ssis_op = op.ssis.data.get_temporal_variables()
    for i, _ in enumerate(ssis):
        while op is not None and ssis_op[i].relevant:
            op.detach()
            rewriter.insert_op(op, insertion_point)
            insertion_point = InsertPoint.after(op)
            op = next(ops_iter, None)
            if op is not None:
                ssis_op = op.ssis.data.get_temporal_variables()
                if len(ssis_op) == 0:
                    ssis_op = ssis
        # move up
        for_op = for_op.parent_op()
        if not isinstance(for_op, ForOp):
            assert i == len(ssis) - 1
            break
        assert isinstance(yield_op := for_op.regions[0].block.last_op, YieldOp)
        insertion_point = InsertPoint.before(yield_op)
