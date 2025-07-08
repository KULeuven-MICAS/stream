from typing import Sequence

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType
from xdsl.dialects.scf import ForOp, YieldOp
from xdsl.ir import Block, Operation, Region
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl_aie.dialects.aie import EndOp

from stream.compiler.dialects.stream import ComputationNodeOp, PullOp, PushOp


def iteration_space_to_for(block: Block, rewriter: Rewriter):
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
    for_op = None
    for i, iter_var in enumerate(reversed(ssis.variables)):
        while op is not None and not op.ssis.data.variables[-i - 1].relevant:
            op.detach()
            rewriter.insert_op(op, insertion_point)
            insertion_point = InsertPoint.after(op)
            op = next(ops_iter, None)
        # create for loop
        ub = ConstantOp.from_int_and_width(iter_var.size, IndexType())
        for_op = ForOp(lb, ub, step, [], Block([yield_op := YieldOp()], arg_types=[IndexType()]))
        rewriter.insert_op((ub, for_op), insertion_point)
        insertion_point = InsertPoint.before(yield_op)
    assert for_op is not None
    # iterate in reverse now:
    for i in reversed(range(len(ssis.variables))):
        if op is not None and len(op.ssis.data.variables) == 0:
            pass
        elif op is not None:
            op_ssis = op.ssis.data
        else:
            op_ssis = ssis

        while op is not None and op_ssis.variables[-i - 1].relevant:
            op.detach()
            rewriter.insert_op(op, insertion_point)
            insertion_point = InsertPoint.after(op)
            op = next(ops_iter, None)
            # TODO: fix computation node ssis bug
            if op is not None and len(op.ssis.data.variables) == 0:
                pass
            elif op is not None:
                op_ssis = op.ssis.data
            else:
                op_ssis = ssis
        # move up
        for_op = for_op.parent_op()
        if not isinstance(for_op, ForOp):
            assert i == 0
            break
        assert isinstance(yield_op := for_op.regions[0].block.last_op, YieldOp)
        insertion_point = InsertPoint.before(yield_op)
