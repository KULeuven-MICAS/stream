from dataclasses import dataclass, field
from math import prod

from xdsl.ir import SSAValue
from xdsl_aie.dialects.aie import TileOp

MEM_ROW = 1
DEFAULT_DEPTH = 2
DEEP_DEPTH = 4
BYTE_MARGIN = 0.5
BD_MARGIN = 0.75


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

    def deepen(self, depths: tuple[int, ...], tiles: tuple[SSAValue, ...], object_bytes: int) -> tuple[int, ...]:
        """Per-endpoint depths, raised where the endpoint is a memory tile with slack.

        Endpoints whose default is not 2 keep it: a 1 was chosen deliberately (the
        allocator costed a single copy) and a larger value already encodes a whole turn.
        """
        if len(depths) != len(tiles):
            return depths
        out = list(depths)
        for i, tile in enumerate(tiles):
            if out[i] != DEFAULT_DEPTH:
                continue
            coords = self._coords(tile)
            if coords is None or coords[1] != MEM_ROW:
                continue
            budget = self.budgets.get(coords)
            if budget is None:
                continue
            extra = self.max_depth - out[i]
            if extra <= 0:
                continue
            cost = extra * object_bytes
            if budget.bytes_free * BYTE_MARGIN < cost or budget.bds_free * BD_MARGIN < extra:
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
