from functools import cached_property
from typing import Any

import networkx as nx

from stream.compiler.kernels.library import KernelLibrary
from stream.cost_model.bandwidth import BandwidthModel
from stream.cost_model.communication_manager import CommunicationManager
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.ports import PortKey, PortRegistry


class MemoryGraph(nx.MultiDiGraph):
    """The memories of an accelerator, with directed edges for the links and buses that move data between them; two
    memories may be joined by several, each its own resource."""


class Accelerator:
    """A set of cores and the memories they compute out of. Each core reads and writes one memory directly, which
    several cores may share; the memories are a graph whose edges are the links and buses that move data between
    them, one of them off-chip."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        cores: list[Core],
        memories: list[Memory],
        network: MemoryGraph,
        offchip_core_id: int | None = None,
        kernel_library: KernelLibrary | None = None,
        bandwidth: dict[Memory, BandwidthModel] | None = None,
        *,
        reconfiguration: dict | None = None,
        port_bandwidth: dict[PortKey, BandwidthModel] | None = None,
    ):
        self.name = name
        self.kernel_library = kernel_library
        self._cores = {core.id: core for core in sorted(cores, key=lambda core: core.id)}
        self.memories = {memory.name: memory for memory in sorted(memories)}
        self._memory_of = {core.id: memory for memory in memories for core in memory.cores}
        if missing := [core_id for core_id in self._cores if core_id not in self._memory_of]:
            raise ValueError(f"cores {missing} are in no memory")
        self.network = network
        self.offchip_core_id = offchip_core_id
        # memory -> the measured bandwidth every transfer into or out of it shares
        self.bandwidth = bandwidth or {}
        # (memory, port name) -> measured model replacing the port's declared width
        self.port_bandwidth = port_bandwidth or {}
        self.reconfiguration = reconfiguration or {}
        self.communication_manager = CommunicationManager(self)

    def get_core(self, core_id: int) -> Core:
        """
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        if core_id not in self._cores:
            raise ValueError(f"no core with id {core_id} in {self}")
        return self._cores[core_id]

    def memory_of(self, core: Core) -> Memory:
        """The memory ``core`` reads and writes directly, its top-level memory."""
        return self._memory_of[core.id]

    @property
    def offchip_memory(self) -> Memory | None:
        return None if self.offchip_core_id is None else self.memory_of(self.get_core(self.offchip_core_id))

    @cached_property
    def ports(self) -> PortRegistry:
        """The top-level memory ports of every core."""
        return PortRegistry.from_accelerator(self)

    @property
    def core_list(self) -> list[Core]:
        return list(self._cores.values())

    def get_ir(self) -> dict:
        """Return a dictionary representation of the accelerator for serialization.

        Captures:
        - Top-level accelerator metadata (name, offchip core, its memories and the cores of each).
        - Per-core IR produced by :meth:`~stream.hardware.architecture.core.Core.get_ir`,
          which includes type-specific fields keyed by ``core_type`` namespace.
        - The connections between memories as a list of ``bus`` or directed ``link`` entries,
          mirroring the structure of the hardware YAML input.
        """
        cores_ir: list[dict] = [core.get_ir() for core in self.core_list]

        # Buses: every edge of one CommunicationLink object (bidirectional) is one bus entry listing its memories.
        # Point-to-point links are emitted as directed pairs.
        links_ir: list[dict] = []
        seen_buses: set[int] = set()
        for src, dst, data in self.network.edges(data=True):
            cl = data["cl"]
            if cl.bidirectional:
                if id(cl) in seen_buses:
                    continue
                seen_buses.add(id(cl))
                links_ir.append(
                    {
                        "type": "bus",
                        "memories": sorted(memory.name for memory in cl.members),
                        "bandwidth": cl.bandwidth,
                        "unit_energy_cost": cl.unit_energy_cost,
                    }
                )
            else:
                links_ir.append(
                    {
                        "type": "link",
                        "from_memory": src.name,
                        "to_memory": dst.name,
                        "bandwidth": cl.bandwidth,
                        "unit_energy_cost": cl.unit_energy_cost,
                    }
                )

        return {
            "name": self.name,
            "num_cores": len(self.core_list),
            "offchip_core_id": self.offchip_core_id,
            "memories": {
                memory.name: [f"{core.id}.{instance}" for core, instance in memory.views]
                for memory in self.memories.values()
            },
            "cores": cores_ir,
            "connections": links_ir,
        }

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self) -> dict[str, Any]:
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name, "cores": self.core_list}
