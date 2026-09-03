from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl_aie.dialects.aie import ObjectFifoAcquireOp, ObjectFifoPortEnum

SCALE_PREFIX = "flash_scale"


def _is_consume(op: ObjectFifoAcquireOp) -> bool:
    return int(op.port.value.data) == ObjectFifoPortEnum.Consume.get_int()


@dataclass(frozen=True)
class HoistFlashScaleAcquires(ModulePass):
    """Acquire the running scale before the operands, on the core that consumes it.

    The value core's emitted order was operands first, scale last, while the score core
    holds its probability fifo across the whole step and blocks on a free scale slot.
    That is a circular wait: the probabilities the value core sleeps on cannot be
    released until the scale slot the score core sleeps on frees, and the slot only
    frees when the value core gets far enough to take it. Observed as an unterminated
    LOCK_STALL on both cores whenever the softmax kernel's timing shifts, and as the
    value core's lock waits in steady state. Taking the scale first breaks the loop:
    the ordering between two consumes of the same cadence changes no dependency,
    only which lock the core sleeps on while both are on their way.
    """

    name = "hoist-flash-scale-acquires"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for acquire in [o for o in op.walk() if isinstance(o, ObjectFifoAcquireOp)]:
            if not acquire.objFifo_name.string_value().startswith(SCALE_PREFIX) or not _is_consume(acquire):
                continue
            block = acquire.parent_block()
            if block is None:
                continue
            first_consume = next(
                (
                    o
                    for o in block.ops
                    if isinstance(o, ObjectFifoAcquireOp)
                    and _is_consume(o)
                    and not o.objFifo_name.string_value().startswith(SCALE_PREFIX)
                ),
                None,
            )
            if first_consume is None or not _precedes(first_consume, acquire):
                continue
            acquire.detach()
            Rewriter().insert_op(acquire, InsertPoint.before(first_consume))


def _precedes(first, second) -> bool:
    op = first
    while op is not None:
        op = op.next_op
        if op is second:
            return True
    return False
