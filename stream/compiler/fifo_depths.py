from dataclasses import dataclass, field
from math import prod

from xdsl.ir import SSAValue
from xdsl_aie.dialects.aie import TileOp

MEM_ROW = 1
COMPUTE_ROW = 2
DEFAULT_DEPTH = 2
DEEP_DEPTH = 4
BYTE_MARGIN = 0.5
BD_MARGIN = 0.75
COMPUTE_BYTE_MARGIN = 0.5


@dataclass
class TileBudget:
    bytes_free: float
    bds_free: float


@dataclass
class FifoDepths:
    """Spend the allocator's leftover per-tile capacity on deeper object fifos.

    The MILP costs every fifo at depth 2; whatever memory and buffer descriptors it
    leaves unused on a tile are real slack, and a deeper fifo on a memory tile lets
    the DMA run ahead of the consumer instead of handing over lockstep. Only memory
    tiles are deepened: a compute tile's leftover bytes are not modelled precisely
    enough (stack, kernel-internal buffers) to spend safely.
    """

    budgets: dict[tuple[int, int], TileBudget] = field(default_factory=dict)
    max_depth: int = DEEP_DEPTH

    @staticmethod
    def _coords(tile: SSAValue) -> tuple[int, int] | None:
        owner = tile.owner
        if not isinstance(owner, TileOp):
            return None
        return owner.col.value.data, owner.row.value.data

    def deepen(
        self, depths: tuple[int, ...], tiles: tuple[SSAValue, ...], object_bytes: int, feed: bool = True
    ) -> tuple[int, ...]:
        """Per-endpoint depths, raised where the endpoint's tile has slack for the extra objects.

        Only fifos feeding a compute core deepen (a memory tile handing kernel tiles down
        its column): a deeper feed there lets the memory tile run ahead of the cores'
        consumption jitter. A deeper DRAM prefetch (shim to memory tile) measured 2.5%
        SLOWER on the weight-streaming SwiGLU at seq 2048 -- the shim is already the
        bottleneck and extra runahead only adds burst pressure -- and a deeper drain
        lengthens buffer-descriptor chains into the per-channel limit the per-tile model
        cannot see. Endpoints whose default is not 2 keep it: a 1 was chosen deliberately
        (the allocator costed a single copy) and a larger value already encodes a whole
        turn. A compute tile deepens only on its consuming side; with the stack reserved and buffer rotation charged in the allocator, the margin only covers kernel-internal statics.
        """
        if not feed:
            return depths
        if len(depths) != len(tiles):
            return depths
        out = list(depths)
        for i, tile in enumerate(tiles):
            if out[i] != DEFAULT_DEPTH:
                continue
            coords = self._coords(tile)
            if coords is None:
                continue
            if coords[1] == MEM_ROW:
                margin = BYTE_MARGIN
            elif coords[1] >= COMPUTE_ROW and i > 0:
                margin = COMPUTE_BYTE_MARGIN
            else:
                continue
            budget = self.budgets.get(coords)
            if budget is None:
                continue
            extra = self.max_depth - out[i]
            if extra <= 0:
                continue
            cost = extra * object_bytes
            if budget.bytes_free * margin < cost or budget.bds_free * BD_MARGIN < extra:
                continue
            budget.bytes_free -= cost
            budget.bds_free -= extra
            out[i] = self.max_depth
        return tuple(out)


def object_bytes(elem_bits: int, shape: tuple[int, ...]) -> int:
    return prod(shape) * elem_bits // 8


def elem_bits(element_type) -> int:
    width = getattr(element_type, "bitwidth", None)
    if width is not None:
        return int(width)
    return int(element_type.width.data)
