"""The physical memories of an accelerator, apart from the cores that use them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stream.hardware.ports import ANY_OPERAND, OwnPort

if TYPE_CHECKING:
    from collections.abc import Iterable

    from stream.hardware.architecture.core import Core
    from stream.hardware.ports import MemoryDma, ViewPort


class Memory:
    """A physical memory, which no core owns: ``views`` pairs each core using it with the top-level instance that core
    sees it as, in declaration order, and ``ports`` are its ports by name. Its capacity is its views' or, for a memory
    no core views, ``capacity_bits``. ``offchip`` where it sits off the die; an accelerator has at most one. ``dma`` is
    the DMA its own ports are the channels of, where it declares one, as the shim tile's DMA that reaches DRAM."""

    dma: MemoryDma | None = None  # a memory pickled before it had one declares none

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        views: tuple[tuple[Core, str], ...],
        ports: dict[str, ViewPort | OwnPort],
        offchip: bool,
        capacity_bits: int | None,
        dma: MemoryDma | None = None,
    ) -> None:
        capacities = {core.top_memory_capacity(instance) for core, instance in views}
        if views and capacity_bits is not None:
            raise ValueError(f"memory {name}: its views declare its capacity, so it declares none itself")
        if not views:
            if capacity_bits is None:
                raise ValueError(f"memory {name}: no core views it, so it declares its capacity")
            capacities = {capacity_bits}
        if len(capacities) != 1:
            raise ValueError(f"memory {name}: its views declare different capacities {sorted(capacities)}")
        if len(banks := {core.memory_banks.get(instance) for core, instance in views} or {None}) != 1:
            raise ValueError(f"memory {name}: its views declare different banks {banks}")
        self.name = name
        self.views = views
        self.ports = ports
        self.offchip = offchip
        self.dma = dma
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

    def dma_port(self, names: Iterable[str], direction: str, operand: str) -> tuple[str, OwnPort] | None:
        """Of its own ports ``names`` (those a connection passes), the DMA (a port declaring channels) a transfer moving
        ``operand`` passes in ``direction``: the one serving that operand (or every operand) in it, else the one
        serving the direction at all; None where none does. Where several do, the connection leaves which the transfer
        takes undeclared (the parse warned), and the first by name is taken, the same every time. Independent of the
        bandwidth model, as the DMA's channels are."""
        dmas = [(name, port) for name in names if isinstance(port := self.ports.get(name), OwnPort) and port.channels]
        own = [(n, p) for n, p in dmas if (operand, direction) in p.serves or (ANY_OPERAND, direction) in p.serves]
        serving = own or [(n, p) for n, p in dmas if any(d == direction for _, d in p.serves)]
        return min(serving, default=None)

    def dma_channels(self, direction: str) -> int | None:
        """The DMA channels its own ports serving ``direction`` are, together, each carrying one fifo at a time; None
        where none of them is a DMA, so that nothing bounds the fifos it serves at once."""
        counts = [
            port.channels
            for port in self.ports.values()
            if isinstance(port, OwnPort) and port.channels and any(d == direction for _, d in port.serves)
        ]
        return sum(counts) if counts else None

    @property
    def staging(self) -> bool:
        """Whether a memory-kind core views it: a memory tile's, or a scratchpad with a DMA engine of its own, which
        transfers may stage tensors in for other cores to fetch. A staging memory compute cores also compute out of,
        as a TensorCore's VMEM, is of kind ``compute`` all the same: what lands in it is where compute reads it."""
        return any(core.kind == "memory" for core in self.cores)

    @property
    def kind(self) -> str:
        """``offchip`` for the off-chip memory, ``compute`` for a memory a core computes out of, else its cores' one
        kind (``memory``); a memory on chip that no core views is a ``memory`` too, a buffer as a memory tile's.
        Whether transfers may stage tensors in it is :attr:`staging`."""
        if self.offchip:
            return "offchip"
        return "compute" if any(core.kind == "compute" for core in self.cores) else "memory"

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
        """The column its cores sit in; None where they declare none, where they span several columns (a memory
        between columns is in neither) or where no core views it."""
        columns = {core.col_id for core in self.cores}
        if len(columns) != 1:
            return None
        (column,) = columns
        return column

    def instance(self, core: Core) -> str:
        """The top-level instance ``core`` sees this memory as."""
        for viewer, instance in self.views:
            if viewer is core:
                return instance
        raise ValueError(f"{core} is no view of memory {self.name}")

    def instance_of(self, core_id: int) -> str:
        """The top-level instance the core of id ``core_id`` sees this memory as."""
        for viewer, instance in self.views:
            if viewer.id == core_id:
                return instance
        raise ValueError(f"core {core_id} is no view of memory {self.name}")

    def __lt__(self, other: Memory) -> bool:
        return self.name < other.name

    def __str__(self) -> str:
        return f"Memory({self.name})"

    def __repr__(self) -> str:
        return str(self)
