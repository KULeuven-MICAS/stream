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
    """The memories of an accelerator, with directed edges for the links and buses between them. Two memories may be
    joined by several edges, each a separate resource."""


class Accelerator:
    """The cores, the memories they use and the network between those memories. Each top-level memory instance of a
    core is a view of one memory, and several cores may view the same memory. Every on-chip memory has a route from
    one of its off-chip memories."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        cores: list[Core],
        memories: list[Memory],
        network: MemoryGraph,
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
        # (core id, top-level instance) -> the memory that instance is a view of
        self._viewed: dict[tuple[int, str], Memory] = {}
        for memory in memories:
            for core, instance in memory.views:
                other = self._viewed.setdefault((core.id, instance), memory)
                if other is not memory:
                    raise ValueError(f"{core.id}.{instance} is a view of both {other.name} and {memory.name}")
        for core in self._cores.values():
            bound = {instance for core_id, instance in self._viewed if core_id == core.id}
            unbound = sorted(core.top_instances() - bound)
            if unbound:
                raise ValueError(f"{core}'s top-level memories {unbound} are views of no memory")
        self.offchip_memories = tuple(memory for memory in memories if memory.offchip)
        self.network = network
        if self.offchip_memories:
            reached = set(self.offchip_memories).union(
                *(nx.descendants(network, memory) for memory in self.offchip_memories if memory in network)
            )
            unreached = sorted(memory.name for memory in memories if memory not in reached)
            if unreached:
                names = " or ".join(memory.name for memory in self.offchip_memories)
                raise ValueError(f"memories {unreached} have no route to off-chip memory {names}")
        # memory -> the measured bandwidth every transfer into or out of it shares
        self.bandwidth = bandwidth or {}
        # (memory, port name) -> measured model replacing the port's declared width
        self.port_bandwidth = port_bandwidth or {}
        self.reconfiguration = reconfiguration or {}
        self.communication_manager = CommunicationManager(self)

    @property
    def offchip_memory(self) -> Memory | None:
        """The off-chip memory, None if there is none; raises if there are several."""
        if len(self.offchip_memories) > 1:
            raise ValueError("several off-chip memories; use offchip_memories")
        return next(iter(self.offchip_memories), None)

    def get_core(self, core_id: int) -> Core:
        """
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        if core_id not in self._cores:
            raise ValueError(f"no core with id {core_id} in {self}")
        return self._cores[core_id]

    def memory_of(self, core: Core, role: str) -> Memory:
        """The memory ``core`` reads or writes the operand of ``role`` ('input<k>' or 'output') in: the one its
        top-level instance holding that operand is a view of."""
        return self._viewed[(core.id, core.top_instance(role))]

    def memory_holding(self, core: Core, role: str) -> Memory | None:
        """:meth:`memory_of`, or None if ``core`` holds no operand of ``role``."""
        instance = core.top_instance_holding(role)
        return None if instance is None else self._viewed[(core.id, instance)]

    def memory_viewed(self, core: Core, instance: str) -> Memory:
        """The memory ``core``'s top-level instance ``instance`` is a view of."""
        return self._viewed[(core.id, instance)]

    def memories_of(self, core: Core) -> tuple[Memory, ...]:
        """Every memory ``core`` sees, one per top-level instance."""
        return tuple(dict.fromkeys(memory for (core_id, _), memory in self._viewed.items() if core_id == core.id))

    @cached_property
    def ports(self) -> PortRegistry:
        """The ports every memory declares."""
        return PortRegistry.from_accelerator(self)

    @property
    def namespace(self) -> str:
        """The namespace of its cores, which must all share one."""
        namespaces = {core.namespace for core in self._cores.values()}
        if len(namespaces) != 1:
            raise ValueError(f"{self} has cores of namespaces {sorted(namespaces)}")
        (namespace,) = namespaces
        return namespace

    @property
    def core_list(self) -> list[Core]:
        return list(self._cores.values())

    def get_ir(self) -> dict:
        """The accelerator as plain data: its name, cores, memories and connections (one entry per declared link or
        bus)."""
        cores_ir: list[dict] = [core.get_ir() for core in self.core_list]
        connections: dict[tuple, dict] = {}
        for _, _, data in self.network.edges(data=True):
            cl = data["cl"]
            ends = {memory.name: list(ports) for memory, ports in cl.ports.items()}
            # a bus is one object for all its edges; a link is two, one per direction
            key = id(cl) if cl.members else (tuple(sorted(ends)), cl.parallel)
            connections.setdefault(
                key,
                {
                    "type": "bus" if cl.members else "link",
                    "ends": ends,
                    "bandwidth": cl.bandwidth,
                    "unit_energy_cost": cl.unit_energy_cost,
                },
            )

        return {
            "name": self.name,
            "num_cores": len(self.core_list),
            "memories": {
                memory.name: {
                    "offchip": memory.offchip,
                    "views": [f"{core.id}.{instance}" for core, instance in memory.views],
                }
                for memory in self.memories.values()
            },
            "cores": cores_ir,
            "connections": list(connections.values()),
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
