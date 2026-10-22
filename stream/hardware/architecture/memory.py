"""The physical memories of an accelerator, apart from the cores that use them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stream.parser.core_validator import OFF_DIE_KINDS

if TYPE_CHECKING:
    from stream.hardware.architecture.core import Core


class Memory:
    """A physical memory, which no core owns: ``views`` pairs each core using it with the top-level instance that core
    sees it as, in declaration order. ``offchip`` where its cores sit off the die (:data:`OFF_DIE_KINDS`)."""

    def __init__(self, name: str, views: tuple[tuple[Core, str], ...]) -> None:
        if not views:
            raise ValueError(f"memory {name} has no views")
        capacities = {core.top_memory_capacity(instance) for core, instance in views}
        assert len(capacities) == 1, f"memory {name}: its views declare different capacities (the validator checks)"
        if len(banks := {core.banks for core, _ in views}) != 1:
            raise ValueError(f"memory {name}: its views declare different banks {banks}")
        self.name = name
        self.views = views
        (self.capacity_bits,) = capacities
        (self.banks,) = banks
        self.offchip = self.kind in OFF_DIE_KINDS

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
        """``compute`` for a memory a core computes out of, else its cores' one kind (``memory``, ``shim``,
        ``offchip``)."""
        kinds = {core.kind for core in self.cores}
        if "compute" in kinds:
            return "compute"
        if len(kinds) != 1:
            raise ValueError(f"memory {self.name} is seen by cores of kinds {sorted(kinds)}")
        (kind,) = kinds
        return kind

    @property
    def namespace(self) -> str:
        """Its cores' one namespace."""
        if len(namespaces := {core.namespace for core in self.cores}) != 1:
            raise ValueError(f"memory {self.name} is seen by cores of namespaces {sorted(namespaces)}")
        (namespace,) = namespaces
        return namespace

    @property
    def col_id(self) -> int | None:
        """The column its cores sit in, None where they declare none."""
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
