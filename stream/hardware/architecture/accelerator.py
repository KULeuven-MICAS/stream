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
    """A set of cores and the memories they compute out of: each top-level memory instance of a core is a view of
    one memory, which several cores may share. The memories are a graph whose edges are the links and buses that move
    data between them; the one an off-die core views is off-chip."""

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
        # (core id, top-level instance) -> the memory that view is of
        self._viewed = {(core.id, instance): memory for memory in memories for core, instance in memory.views}
        for core in self._cores.values():
            bound = {instance for core_id, instance in self._viewed if core_id == core.id}
            if unbound := sorted(core.top_instances() - bound):
                raise ValueError(f"{core}'s top-level memories {unbound} are views of no memory")
        if len(offchip := [memory for memory in memories if memory.offchip]) > 1:
            raise ValueError(f"several off-chip memories: {offchip}")
        self.offchip_memory = offchip[0] if offchip else None
        if offchip_core_id is not None and (viewed := self.memories_of(self.get_core(offchip_core_id))) != tuple(
            offchip
        ):
            raise ValueError(
                f"offchip_core_id {offchip_core_id} views {viewed}, not the off-chip memory {offchip}, the one the "
                f"off-die cores view"
            )
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

    def memory_of(self, core: Core, role: str) -> Memory:
        """The memory ``core`` reads or writes the operand of ``role`` ('input<k>' or 'output') in: the one its
        top-level instance holding that operand is a view of."""
        return self._viewed[(core.id, core.top_instance(role))]

    def memories_of(self, core: Core) -> tuple[Memory, ...]:
        """Every memory ``core`` sees, one per top-level instance."""
        return tuple(dict.fromkeys(memory for (core_id, _), memory in self._viewed.items() if core_id == core.id))

    @cached_property
    def ports(self) -> PortRegistry:
        """The top-level memory ports of every core."""
        return PortRegistry.from_accelerator(self)

    @property
    def core_list(self) -> list[Core]:
        return list(self._cores.values())

    def get_ir(self) -> dict:
        """The accelerator as plain data: its name, off-chip core, cores (:meth:`Core.get_ir`), memories with their
        views, and connections, one entry per link or bus as the hardware description declares them."""
        cores_ir: list[dict] = [core.get_ir() for core in self.core_list]
        connections: dict[tuple, dict] = {}
        for src, dst, data in self.network.edges(data=True):
            cl = data["cl"]
            ends = [memory.name for memory in cl.members] if cl.members else [src.name, dst.name]
            # a bus is one object for all its edges; a link is two, one per direction
            key = id(cl) if cl.members else (tuple(sorted(ends)), cl.parallel)
            connections.setdefault(
                key,
                {
                    "type": "bus" if cl.members else "link",
                    "memories": ends,
                    "bandwidth": cl.bandwidth,
                    "unit_energy_cost": cl.unit_energy_cost,
                },
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
