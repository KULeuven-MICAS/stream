from typing import Any

from zigzag.mapping.spatial_mapping import SpatialMapping
from zigzag.utils import DiGraphWrapper

from stream.cost_model.communication_manager import CommunicationManager
from stream.hardware.architecture.core import Core


class CoreGraph(DiGraphWrapper[Core]):
    """Represents the core structure of an accelerator"""


class Accelerator:
    """
    The Accelerator class houses a set of Cores with an additional Global Buffer.
    This Global Buffer sits above the cores, and can optionally be disabled.
    In this Stream version, the cores are actually a graph with directed edges representing communication links.
    """

    def __init__(
        self,
        name: str,
        cores: CoreGraph,
        nb_shared_mem_groups: int,
        offchip_core_id: int | None = None,
    ):
        """ """
        self.name = name
        self.cores = cores
        self.offchip_core_id = offchip_core_id
        self.nb_shared_mem_groups = nb_shared_mem_groups
        self.communication_manager = CommunicationManager(self)

    def get_core(self, core_id: int) -> Core:
        """
        Return the core with id 'core_id'.
        Raises ValueError() when a core_id is not found in the available cores.
        """
        return self.cores.get_node_with_id(core_id)

    def get_offchip_core(self) -> Core:
        """Return the offchip core."""
        assert self.offchip_core_id, "This accelerator has no offchip core id."
        return self.get_core(self.offchip_core_id)

    def get_spatial_mapping_from_core(self, core_allocation: list[int]) -> SpatialMapping:
        """Iff the dataflows of all given cores is the same, return that dataflow. Otherwise, throw an error"""
        all_dataflows = [self.get_core(core_id).dataflows for core_id in core_allocation]
        some_dataflow = all_dataflows.pop()

        # All cores have same dataflow
        if some_dataflow is not None and all(some_dataflow == dataflow for dataflow in all_dataflows):
            return some_dataflow

        raise ValueError("Unclear which dataflow to return or no valid dataflow found.")

    @property
    def core_list(self) -> list[Core]:
        return list(self.cores.node_list)

    def get_ir(self) -> dict:
        """Return a dictionary representation of the accelerator for serialization.

        Captures:
        - Top-level accelerator metadata (name, offchip core, shared memory groups).
        - Per-core IR produced by :meth:`~stream.hardware.architecture.core.Core.get_ir`,
          which includes type-specific fields keyed by ``core_type`` namespace.
        - Core connectivity expressed as a list of ``bus`` or directed ``link`` entries,
          mirroring the structure of the hardware YAML input.
        """
        # --- cores ---
        cores_ir: list[dict] = [core.get_ir() for core in self.cores.node_list]

        # --- connectivity ---
        # Buses: all edges that share the same CommunicationLink object (bidirectional=True)
        # are collected into a single bus entry listing every participating core.
        # Point-to-point links (bidirectional=False) are emitted as directed pairs.
        links_ir: list[dict] = []
        seen_bus_ids: set[int] = set()
        for src, dst, data in self.cores.edges(data=True):
            cl = data.get("cl")
            if cl is None:
                continue
            if cl.bidirectional:
                bus_id = id(cl)
                if bus_id in seen_bus_ids:
                    continue
                seen_bus_ids.add(bus_id)
                # Collect every core that participates in this shared bus instance
                connected_ids: set[int] = set()
                for u, v, edge_data in self.cores.edges(data=True):
                    if edge_data.get("cl") is cl:
                        connected_ids.add(u.id)
                        connected_ids.add(v.id)
                links_ir.append(
                    {
                        "type": "bus",
                        "cores": sorted(connected_ids),
                        "bandwidth": cl.bandwidth,
                        "unit_energy_cost": cl.unit_energy_cost,
                    }
                )
            else:
                links_ir.append(
                    {
                        "type": "link",
                        "from_core": src.id,
                        "to_core": dst.id,
                        "bandwidth": cl.bandwidth,
                        "unit_energy_cost": cl.unit_energy_cost,
                    }
                )

        return {
            "name": self.name,
            "num_cores": len(list(self.cores.nodes)),
            "offchip_core_id": self.offchip_core_id,
            "nb_shared_mem_groups": self.nb_shared_mem_groups,
            "cores": cores_ir,
            "core_connectivity": links_ir,
        }

    def __str__(self) -> str:
        return f"Accelerator({self.name})"

    def __repr__(self) -> str:
        return str(self)

    def __jsonrepr__(self) -> dict[str, Any]:
        """
        JSON representation used for saving this object to a json file.
        """
        return {"name": self.name, "cores": self.cores}
