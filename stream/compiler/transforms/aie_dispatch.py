from collections import defaultdict

from xdsl.context import Context
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.dialects.csl import RewritePattern
from xdsl.ir import Block, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl_aie.dialects.aie import (
    AIEDeviceEnum,
    CoreOp,
    DeviceOp,
    EndOp,
    RuntimeSequenceOp,
    TileOp,
)

from stream.compiler.dialects.stream import (
    ChannelOp,
    ComputationNodeOp,
    FusionGroupOp,
    PullOp,
    PushOp,
    StrensorType,
    YieldOp,
)


class FusionGroupDispatcher(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, fusion_op: FusionGroupOp, rewriter: PatternRewriter):
        locs: dict[str, list[Operation]] = defaultdict(list)

        # determine new location for each block
        for op in fusion_op.body.block.ops:
            # root ops:
            if isinstance(op, ChannelOp):
                locs["root"].append(op)
                continue
            # otherwise, find core allocation:
            if isinstance(op, PullOp | ComputationNodeOp):
                assert isinstance(strensor := op.output.type, StrensorType)
            elif isinstance(op, PushOp):
                assert isinstance(strensor := op.input.type, StrensorType)
            elif isinstance(op, YieldOp):
                assert len(op.arguments) == 1
                assert isinstance(strensor := op.arguments[0].type, StrensorType)
            else:
                raise NotImplementedError("unknown op found")
            assert len(strensor.core_allocation) == 1
            core = strensor.core_allocation.data[0]
            locs[core.data].append(op)

        root_ops = locs.pop("root")
        for op in root_ops:
            op.detach()

        # Create proper runtime sequence:
        runtime_ops = locs.pop("tile_0_0")
        runtime = RuntimeSequenceOp(Region(Block(arg_types=fusion_op.body.block.arg_types)))
        for op in runtime_ops:
            op.detach()
            rewriter.insert_op(op, InsertPoint.at_end(runtime.body.block))
        for f_arg, r_arg in zip(fusion_op.body.block.args, runtime.body.block.args, strict=False):
            f_arg.replace_by(r_arg)

        core_ops: list[Operation] = []

        # All others go into Core() ops:
        for core, ops in locs.items():
            row = int(core[-1])
            col = int(core[-3])
            tile_op = TileOp(col, row)
            core_op = CoreOp(None, tile_op, Region(Block()))
            for op in ops:
                op.detach()
                rewriter.insert_op(op, InsertPoint.at_end(core_op.region.block))
            rewriter.insert_op(EndOp(), InsertPoint.at_end(core_op.region.block))
            core_ops.extend((tile_op, core_op))

        # All of this gets put in a device op:
        device = AIEDeviceEnum.npu2
        device_op = DeviceOp(
            IntegerAttr.from_int_and_width(device.get_int(), 32),
            Region(Block((*root_ops, runtime, *core_ops, EndOp()))),
        )

        rewriter.replace_matched_op(device_op)


class AIEDispatchPass(ModulePass):
    """
    Performs all dispatches for an aie design.
    """

    name = "aie-dispatch"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(FusionGroupDispatcher()).rewrite_module(op)
