"""The physical memories of an accelerator, apart from the cores that use them."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core


class Memory:
    """A physical memory, seen by each of its cores as one of that core's top-level memory instances: ``views`` pairs
    each core with the instance's name in it. Every core reads and writes it directly through its own view, with its
    own ports into it; none of them owns it. Data in it moves between them in place, links and buses connect memories,
    and its capacity bounds what all its cores hold together. ``offchip`` for the memory outside the chip, which data
    never passes through on its way between two others."""

    def __init__(self, name: str, views: tuple[tuple[Core, str], ...], offchip: bool = False) -> None:
        if not views:
            raise ValueError(f"memory {name} has no views")
        capacities = {f"{core.id}.{instance}": core.top_memory_capacity(instance) for core, instance in views}
        if len(set(capacities.values())) != 1:
            raise ValueError(
                f"memory {name}: its views declare different capacities {capacities}; they are one memory, so they "
                "must agree"
            )
        self.name = name
        self.views = tuple(sorted(views, key=lambda view: view[0].id))
        self.offchip = offchip
        (self.capacity_bits,) = set(capacities.values())

    @property
    def cores(self) -> tuple[Core, ...]:
        """The cores that see this memory, by id."""
        return tuple(core for core, _ in self.views)

    def instance(self, core: Core) -> str:
        """The name ``core`` gives this memory: its view's top-level instance."""
        return next(instance for viewer, instance in self.views if viewer is core)

    def __lt__(self, other: Memory) -> bool:
        return self.name < other.name

    def __str__(self) -> str:
        return f"Memory({self.name})"

    def __repr__(self) -> str:
        return str(self)
