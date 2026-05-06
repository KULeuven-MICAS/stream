from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.csl import RewritePattern
from xdsl.ir import OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl_aie.dialects.aie import (
    CoreOp,
    DeviceOp,
    TileOp,
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


class OrderCoreOps(RewritePattern):
    # Complete a bubble-type sorting of core ops for a more deterministic output
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CoreOp, rewriter: PatternRewriter):
        def get_tile_idx(op: CoreOp) -> tuple[int, int]:
            assert isinstance(op.tile, OpResult)
            assert isinstance(op.tile.op, TileOp)
            return (op.tile.op.col.value.data, op.tile.op.row.value.data)

        if not isinstance(next_op := op.next_op, CoreOp):
            return

        if get_tile_idx(op) > get_tile_idx(next_op):
            next_op.detach()
            rewriter.insert_op(next_op, InsertPoint.before(op))


class AIEMoveTileOpsUp(ModulePass):
    name = "aie-move-tile-ops-up"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(MoveTileOpsUpPattern(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(OrderCoreOps()).rewrite_module(op)
