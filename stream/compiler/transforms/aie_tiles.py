"""Where on the AIE array each placement of the generated code is."""

from dataclasses import dataclass

# The kinds of `Memory.kind` and `Core.kind`
OFFCHIP = "offchip"
MEMORY = "memory"
COMPUTE = "compute"


@dataclass(frozen=True)
class AIETileLocation:
    """Where a placement is: DRAM (``OFFCHIP``), reached from the runtime sequence, or the memory or compute tile at
    ``column``, ``row``."""

    kind: str
    column: int | None
    row: int | None

    @property
    def order(self) -> tuple[int, int]:
        """DRAM first, then the tiles by column and row."""
        return (-1, -1) if self.kind == OFFCHIP else self.tile

    @property
    def tile(self) -> tuple[int, int]:
        """The tile's (column, row); DRAM is on none."""
        if self.column is None or self.row is None:
            raise ValueError("DRAM is on no tile")
        return self.column, self.row
