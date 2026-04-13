from collections import defaultdict
from typing import Sequence
from xdsl.context import Context, MLContext
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.dialects.csl import RewritePattern
from xdsl.ir import Block, Operation, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
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


class MoveTileOpsUpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, device_op: DeviceOp, rewriter: PatternRewriter):

        tile_ops = []
        for tile_op in device_op.walk():
            if isinstance(tile_op, TileOp):
                tile_ops.append(tile_op)
                tile_op.detach()

        rewriter.insert_op(tile_ops, InsertPoint.at_start(device_op.region.block))


class AIEMoveTileOpsUp(ModulePass):
    name = "aie-move-tile-ops-up"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(MoveTileOpsUpPattern(), apply_recursively=False).rewrite_module(op)
