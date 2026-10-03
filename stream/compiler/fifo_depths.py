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
    """Spend the allocator's leftover per-tile capacity on deeper object fifos."""

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
        """Per-endpoint depths, raised as far as each endpoint's tile has slack for the extra objects. Readers on
        compute tiles deepen together, since a broadcast advances only once every reader has room."""
        if not feed or len(depths) != len(tiles):
            return depths
        out = list(depths)
        readers = []
        for i, tile in enumerate(tiles):
            coords = self._coords(tile)
            if out[i] != DEFAULT_DEPTH or coords is None or coords not in self.budgets:
                continue
            if coords[1] == MEM_ROW:
                out[i] = self._spend(coords, self._affordable(coords, BYTE_MARGIN, object_bytes), object_bytes)
            elif coords[1] >= COMPUTE_ROW and i > 0:
                readers.append((i, coords))
        if readers:
            depth = min(self._affordable(coords, COMPUTE_BYTE_MARGIN, object_bytes) for _, coords in readers)
            for i, coords in readers:
                out[i] = self._spend(coords, depth, object_bytes)
        return tuple(out)

    def _affordable(self, coords: tuple[int, int], margin: float, object_bytes: int) -> int:
        """The deepest depth up to ``max_depth`` whose extra objects fit the tile's slack."""
        budget = self.budgets[coords]
        for depth in range(self.max_depth, DEFAULT_DEPTH, -1):
            extra = depth - DEFAULT_DEPTH
            if budget.bytes_free * margin >= extra * object_bytes and budget.bds_free * BD_MARGIN >= extra:
                return depth
        return DEFAULT_DEPTH

    def _spend(self, coords: tuple[int, int], depth: int, object_bytes: int) -> int:
        budget = self.budgets[coords]
        extra = depth - DEFAULT_DEPTH
        budget.bytes_free -= extra * object_bytes
        budget.bds_free -= extra
        return depth


def object_bytes(elem_bits: int, shape: tuple[int, ...]) -> int:
    return prod(shape) * elem_bits // 8


def elem_bits(element_type) -> int:
    width = getattr(element_type, "bitwidth", None)
    if width is not None:
        return int(width)
    return int(element_type.width.data)
