from typing import Any

from zigzag.hardware.architecture.memory_level import MemoryLevel
from zigzag.parser.accelerator_factory import AcceleratorFactory as ZigZagCoreFactory

from stream.hardware.architecture.accelerator import Accelerator, CoreGraph
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink, get_bidirectional_edges


class AcceleratorFactory:
    """! Converts valid user-provided accelerator data into an `Accelerator` instance"""

    def __init__(self, data: dict[str, Any]):
        """! Generate an `Accelerator` instance from the validated user-provided data."""
        self.data = data

    def create(self) -> Accelerator:
        """! Create an Accelerator instance from the user-provided data."""
        cores: list[Core] = []
        unique_shared_mem_group_ids: set[int] = set()

        for core_id, core_data in self.data["cores"].items():
            shared_mem_group_id = self.get_shared_mem_group_id(core_id)
            core = self.create_core(core_data, core_id, shared_mem_group_id)
            cores.append(core)
            unique_shared_mem_group_ids.add(shared_mem_group_id)

        # Extra check on shared memory
        if self.have_non_identical_shared_memory(cores):
            raise ValueError(
                "Some cores that were specified as having shared memory have non-identical top level "
                "memories. Make sure that all properties of the shared memory instances are the same in "
                "both yaml files, including the memory instance name."
            )

        # Save offchip core id if it exists
        offchip_core_id = self.data.get("offchip_core_id", None)

        cores_graph = self.create_core_graph(cores)
        nb_shared_mem_groups = len(unique_shared_mem_group_ids)

        # Take next available core id
        return Accelerator(
            name=self.data["name"],
            cores=cores_graph,
            offchip_core_id=offchip_core_id,
            nb_shared_mem_groups=nb_shared_mem_groups,
        )

    def create_core(self, core_data: dict[str, Any], core_id: int, shared_mem_group_id: int | None = None):
        core_factory = ZigZagCoreFactory(core_data)
        core = core_factory.create(core_id, shared_mem_group_id=shared_mem_group_id)
        # Typecast
        core = Core.from_zigzag_core(core)
        core.type = core_data.get("type", "compute")  # Default type is 'compute'
        return core

    def get_shared_mem_group_id(self, core_id: int):
        """Calculate the memory group id for the given core. If the core shares the top level memory with other cores,
        the mem group id of all cores that share this memory will be the same. By default, the mem group id is equal
        to the core id"""
        shared_mem_groups: list[tuple[int]] = self.data["core_memory_sharing"]

        # Core does not share memory with other cores
        if not any(core_id in group for group in shared_mem_groups):
            return core_id
        pair_this_core = next(group for group in shared_mem_groups if core_id in group)
        # Mem sharing group id is the first core id of the sharing list
        return pair_this_core[0]

    def have_non_identical_shared_memory(self, cores: list[Core]):
        """Given the list of cores (where the index of the list equals the core id) and the user-specified shared
        memory connections, check wether all cores that are supposed to share memory have identical top level memories
        """
        shared_mem_groups: list[tuple[int, ...]] = self.data["core_memory_sharing"]
        for shared_mem_group in shared_mem_groups:
            core_a = cores[shared_mem_group[0]]
            if any(self.have_non_identical_top_memory(core_a, cores[id_b]) for id_b in shared_mem_group[1:]):
                return True
        return False

    def have_non_identical_top_memory(self, core_a: Core, core_b: Core):
        """Check wether the top level memories of two cores is exactly the same. This should be the case when the user
        has specified the cores share memory"""

        top_levels_a: list[MemoryLevel] = list(
            (level for level, out_degree in core_a.memory_hierarchy.out_degree() if out_degree == 0)
        )
        top_levels_b: list[MemoryLevel] = list(
            (level for level, out_degree in core_b.memory_hierarchy.out_degree() if out_degree == 0)
        )
        top_instances_a = [level.memory_instance for level in top_levels_a]
        top_instances_b = [level.memory_instance for level in top_levels_b]
        if len(top_instances_a) != len(top_instances_b):
            return True
        for instance_a, instance_b in zip(top_instances_a, top_instances_b, strict=False):
            if frozenset(instance_a.__dict__.values()) != frozenset(instance_b.__dict__.values()):
                return True
        return False

    def create_core_graph(self, cores: list[Core]):
        assert all(core.id == i for i, core in enumerate(cores))
        edges: list[tuple[Core, Core, dict[str, CommunicationLink]]] = []
        current_bus_id = 0
        core_connectivity: list[dict[str, Any]] = self.data["core_connectivity"]
        default_unit_energy_cost = self.data.get("unit_energy_cost", 0)

        # All links between cores
        for connection in core_connectivity:
            connection_type = connection.get("type", "link")
            bw = connection["bandwidth"]
            uec = connection.get("unit_energy_cost", default_unit_energy_cost)
            core_objs = [cores[cid] for cid in connection["cores"]]
            CONNECTION_LENGTH_FOR_ONE_TO_ONE = 2
            if connection_type == "link":
                if len(core_objs) != CONNECTION_LENGTH_FOR_ONE_TO_ONE:
                    raise ValueError(
                        f"Invalid connection type '{connection_type}' for connection {core_objs}. "
                        f"Expected exactly {CONNECTION_LENGTH_FOR_ONE_TO_ONE} cores for a link."
                    )
                core_a, core_b = core_objs
                edges += get_bidirectional_edges(
                    core_a,
                    core_b,
                    bandwidth=bw,
                    unit_energy_cost=uec,
                    link_type=connection_type,
                )
            elif connection_type == "bus":
                # Connect cores to bus, edge by edge
                # Make sure all links refer to the same `CommunicationLink` instance
                bus_instance = CommunicationLink("Any", "Any", bw, uec, bus_id=current_bus_id)
                current_bus_id += 1
                pairs_this_connection = [(a, b) for idx, a in enumerate(core_objs) for b in core_objs[idx + 1 :]]
                for core_a, core_b in pairs_this_connection:
                    edges += get_bidirectional_edges(
                        core_a,
                        core_b,
                        bandwidth=bw,
                        unit_energy_cost=uec,
                        link_type="bus",
                        bus_instance=bus_instance,
                    )
            else:
                raise ValueError(
                    f"Invalid connection type '{connection_type}' for connection {core_objs}. Expected 'link' or 'bus'."
                )
        return CoreGraph(edges)
