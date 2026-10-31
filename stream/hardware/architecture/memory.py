"""The physical memories of an accelerator."""

from __future__ import annotations

import zlib
from collections.abc import Iterable

from stream.hardware.architecture.core import Core
from stream.hardware.ports import MemoryDma, Port, candidate_ports


class Memory:
    """A physical memory. ``views`` pairs each core that uses it with the name of that core's top-level memory
    instance for it. ``ports`` are its ports by name. The capacity comes from the views, or from ``capacity_bits`` if
    no core views it. ``offchip`` marks the off-chip memory; an accelerator has at most one. ``dma`` is its DMA engine,
    if it has one."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        views: tuple[tuple[Core, str], ...],
        ports: dict[str, Port],
        offchip: bool,
        capacity_bits: int | None,
        dma: MemoryDma | None = None,
    ) -> None:
        if views:
            if capacity_bits is not None:
                raise ValueError(f"memory {name}: its views declare its capacity, so it declares none itself")
            capacities = {core.top_memory_capacity(instance) for core, instance in views}
            if len(capacities) != 1:
                raise ValueError(f"memory {name}: its views declare different capacities {sorted(capacities)}")
            (capacity_bits,) = capacities
        elif capacity_bits is None:
            raise ValueError(f"memory {name}: no core views it, so it declares its capacity")
        banks = {core.memory_banks.get(instance) for core, instance in views}
        if len(banks) > 1:
            raise ValueError(f"memory {name}: its views declare different banks {banks}")
        self.name = name
        self.views = views
        self.ports = ports
        self.offchip = offchip
        self.dma = dma
        self.capacity_bits = capacity_bits
        self.banks = next(iter(banks), None)

    @property
    def cores(self) -> tuple[Core, ...]:
        return tuple(core for core, _ in self.views)

    @property
    def core(self) -> Core:
        """The one core that views it; raises if it has several."""
        cores = self.cores
        if len(cores) != 1:
            raise ValueError(f"memory {self.name} is shared by {len(cores)} cores")
        return cores[0]

    def dma_port(self, names: Iterable[str], direction: str, operand: str) -> Port | None:
        """Of the ports ``names``, the DMA port (one with channels) that a transfer of ``operand`` in ``direction``
        uses, chosen as in :func:`~stream.hardware.ports.candidate_ports`. None if there is none; an error if several
        qualify."""
        dmas = tuple(self.ports[name] for name in names if name in self.ports and self.ports[name].channels)
        serving = candidate_ports(dmas, direction, operand)
        if len(serving) > 1:
            names = sorted(p.key.port for p in serving)
            raise ValueError(
                f"memory {self.name}: DMA ports {names} all serve {operand} {direction}; name the one a transfer passes"
            )
        return serving[0] if serving else None

    @property
    def staging(self) -> bool:
        """Whether a memory-kind core views it, so transfers may stage tensors in it for other cores to fetch. This is
        independent of :attr:`kind`: a staging memory that compute cores also use is of kind ``compute``."""
        return any(core.kind == "memory" for core in self.cores)

    @property
    def kind(self) -> str:
        """``offchip`` for the off-chip memory, ``compute`` if a compute core uses it, else ``memory`` (also for an
        on-chip memory no core views)."""
        if self.offchip:
            return "offchip"
        return "compute" if any(core.kind == "compute" for core in self.cores) else "memory"

    @property
    def namespace(self) -> str | None:
        """The namespace of its cores; None if no core views it."""
        namespaces = {core.namespace for core in self.cores}
        if len(namespaces) > 1:
            raise ValueError(f"memory {self.name} is seen by cores of namespaces {sorted(namespaces)}")
        return next(iter(namespaces), None)

    @property
    def col_id(self) -> int | None:
        """The column of its cores; None if they declare none or no core views it."""
        columns = {core.col_id for core in self.cores}
        if len(columns) > 1:
            raise ValueError(f"memory {self.name} spans columns {sorted(columns, key=str)}")
        return next(iter(columns), None)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Memory) and self.name == other.name

    def __hash__(self) -> int:
        """By a digest of its name that every run shares, so sets of memories iterate alike and build the same model."""
        return zlib.crc32(self.name.encode())

    def __lt__(self, other: Memory) -> bool:
        return self.name < other.name

    def __str__(self) -> str:
        return f"Memory({self.name})"

    def __repr__(self) -> str:
        return str(self)
