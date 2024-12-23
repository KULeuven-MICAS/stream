from typing import Any

from zigzag.datatypes import Constants
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

        # Create offchip core
        if "offchip_core" not in self.data:
            offchip_core_id = None
            offchip_core = None
        else:
            offchip_core_id = max(self.data["cores"]) + 1
            offchip_core = self.create_core(self.data["offchip_core"], offchip_core_id)

        cores_graph = self.create_core_graph(cores, offchip_core)
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
        for instance_a, instance_b in zip(top_instances_a, top_instances_b):
            if frozenset(instance_a.__dict__.values()) != frozenset(instance_b.__dict__.values()):
                return True
        return False

    def create_core_graph(self, cores: list[Core], offchip_core: Core | None):
        assert all(core.id == i for i, core in enumerate(cores))
        bandwidth = self.data["bandwidth"]
        unit_energy_cost = self.data["unit_energy_cost"]
        connections: list[tuple[int, ...]] = self.data["core_connectivity"]
        edges: list[tuple[Core, Core, dict[str, CommunicationLink]]] = []
        current_bus_id = 0

        # All links between cores
        for connection in connections:
            connected_cores = [cores[core_id] for core_id in connection]

            if len(connection) == 2:
                core_a, core_b = connected_cores
                edges += get_bidirectional_edges(
                    core_a,
                    core_b,
                    bandwidth=bandwidth,
                    unit_energy_cost=unit_energy_cost,
                    link_type="link",
                )
            else:
                # Connect cores to bus, edge by edge
                # Make sure all links refer to the same `CommunicationLink` instance
                bus_instance = CommunicationLink("Any", "Any", bandwidth, unit_energy_cost, bus_id=current_bus_id)
                current_bus_id += 1
                pairs_this_connection = [
                    (a, b) for idx, a in enumerate(connected_cores) for b in connected_cores[idx + 1 :]
                ]
                for core_a, core_b in pairs_this_connection:
                    edges += get_bidirectional_edges(
                        core_a,
                        core_b,
                        bandwidth=bandwidth,
                        unit_energy_cost=unit_energy_cost,
                        link_type="bus",
                        bus_instance=bus_instance,
                    )

        # All links between cores and offchip core
        if offchip_core is not None:
            edges += self.get_edges_to_offchip_core(cores, offchip_core)
        return CoreGraph(edges)

    def get_edges_to_offchip_core(self, cores: list[Core], offchip_core: Core):
        edges: list[tuple[Core, Core, dict[str, CommunicationLink]]] = []

        unit_energy_cost = self.data["unit_energy_cost"]

        offchip_read_bandwidth = offchip_core.mem_r_bw_dict[Constants.OUTPUT_MEM_OP][0]
        offchip_write_bandwidth = offchip_core.mem_w_bw_dict[Constants.OUTPUT_MEM_OP][0]

        # if the offchip core has only one port
        if len(offchip_core.mem_hierarchy_dict[Constants.OUTPUT_MEM_OP][0].port_list) == 1:
            to_offchip_link = CommunicationLink(
                offchip_core,
                "Any",
                offchip_write_bandwidth,
                unit_energy_cost,
                bidirectional=True,
            )
            from_offchip_link = to_offchip_link

        # if the offchip core has more than one port
        else:
            to_offchip_link = CommunicationLink("Any", offchip_core, offchip_write_bandwidth, unit_energy_cost)
            from_offchip_link = CommunicationLink(offchip_core, "Any", offchip_read_bandwidth, unit_energy_cost)

        # Create edge for each core
        for core in cores:
            edges.append((core, offchip_core, {"cl": to_offchip_link}))
            edges.append((offchip_core, core, {"cl": from_offchip_link}))

        return edges
