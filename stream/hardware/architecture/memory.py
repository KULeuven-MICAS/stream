"""The physical memories of an accelerator, apart from the cores that use them."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core
    from stream.hardware.ports import OwnPort, ViewPort


class Memory:
    """A physical memory, which no core owns: ``views`` pairs each core using it with the top-level instance that core
    sees it as, in declaration order, and ``ports`` are its ports by name. Its capacity is its views' or, for a memory
    no core views, ``capacity_bits``. ``offchip`` where it sits off the die; an accelerator has at most one."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        views: tuple[tuple[Core, str], ...],
        ports: dict[str, ViewPort | OwnPort],
        offchip: bool,
        capacity_bits: int | None,
    ) -> None:
        capacities = {core.top_memory_capacity(instance) for core, instance in views}
        if views and capacity_bits is not None:
            raise ValueError(f"memory {name}: its views declare its capacity, so it declares none itself")
        if not views:
            if capacity_bits is None:
                raise ValueError(f"memory {name}: no core views it, so it declares its capacity")
            capacities = {capacity_bits}
        assert len(capacities) == 1, f"memory {name}: its views declare different capacities (the validator checks)"
        if len(banks := {core.memory_banks.get(instance) for core, instance in views} or {None}) != 1:
            raise ValueError(f"memory {name}: its views declare different banks {banks}")
        self.name = name
        self.views = views
        self.ports = ports
        self.offchip = offchip
        (self.capacity_bits,) = capacities
        (self.banks,) = banks

    @property
    def cores(self) -> tuple[Core, ...]:
        return tuple(core for core, _ in self.views)

    @property
    def core(self) -> Core:
        """The one core it is a view of, as an AIE tile's memory is; raises for a memory several cores share."""
        if len(cores := self.cores) != 1:
            raise ValueError(f"memory {self.name} is shared by {len(cores)} cores")
        return cores[0]

    @property
    def kind(self) -> str:
        """``offchip`` for the off-chip memory, ``compute`` for a memory a core computes out of, else its cores' one
        kind (``memory``)."""
        if self.offchip:
            return "offchip"
        kinds = {core.kind for core in self.cores}
        if "compute" in kinds:
            return "compute"
        if len(kinds) != 1:
            raise ValueError(f"memory {self.name} is seen by cores of kinds {sorted(kinds)}")
        (kind,) = kinds
        return kind

    @property
    def namespace(self) -> str | None:
        """Its cores' one namespace, None for a memory no core views."""
        if not self.views:
            return None
        if len(namespaces := {core.namespace for core in self.cores}) != 1:
            raise ValueError(f"memory {self.name} is seen by cores of namespaces {sorted(namespaces)}")
        (namespace,) = namespaces
        return namespace

    @property
    def col_id(self) -> int | None:
        """The column its cores sit in, None where they declare none or no core views it."""
        if not self.views:
            return None
        if len(columns := {core.col_id for core in self.cores}) != 1:
            raise ValueError(f"memory {self.name} spans columns {sorted(columns, key=str)}")
        (column,) = columns
        return column

    def instance(self, core: Core) -> str:
        """The top-level instance ``core`` sees this memory as."""
        for viewer, instance in self.views:
            if viewer is core:
                return instance
        raise ValueError(f"{core} is no view of memory {self.name}")

    def __lt__(self, other: Memory) -> bool:
        return self.name < other.name

    def __str__(self) -> str:
        return f"Memory({self.name})"

    def __repr__(self) -> str:
        return str(self)
