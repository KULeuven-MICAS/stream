from typing import Any

from zigzag.parser.accelerator_factory import AcceleratorFactory as ZigZagCoreFactory

from stream.hardware.architecture.accelerator import Accelerator, CoreGraph
from stream.hardware.architecture.backends import AIE2CoreBackend, ZigZagCoreBackend
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink, get_bidirectional_edges
from stream.parser.core_validator import ALLOWED_KINDS, ALLOWED_NAMESPACES, CoreValidatorRegistry


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
            coordinates = self.data.get("core_coordinates", {}).get(core_id)
            core = self.create_core(core_data, core_id, shared_mem_group_id, coordinates)
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

    def create_core(
        self,
        core_data: dict[str, Any],
        core_id: int,
        shared_mem_group_id: int | None = None,
        coordinates: list[int] | None = None,
    ) -> Core:
        # Resolve the fully-qualified core type (e.g. "aie2.compute")
        raw_type = core_data.get("type")
        default_kind = raw_type if raw_type in ALLOWED_KINDS else "compute"
        core_type = CoreValidatorRegistry.normalize_core_type(
            raw_type,
            default_namespace=CoreValidatorRegistry.default_namespace,
            default_kind=default_kind,
        )
        namespace = core_type.split(".")[0] if "." in core_type else ""

        col_id = coordinates[0] if coordinates else None
        row_id = coordinates[1] if coordinates else None

        # Read operator_types from raw core_data before any validation strips unknown fields
        operator_types = core_data.get("operator_types", None)

        if namespace == "aie2":
            # ---- AIE2 native path: lightweight backend ----
            mem = core_data["memory"]
            backend = AIE2CoreBackend(
                memory_capacity_bits=mem["capacity"],
                bandwidth_min=mem.get("bandwidth_min", 0),
                bandwidth_max=mem.get("bandwidth_max", 0),
            )
            core = Core(
                backend=backend,
                core_id=core_id,
                name=core_data.get("name", f"core_{core_id}"),
                core_type=core_type,
                utilization=core_data.get("utilization", 100),
                max_object_fifo_depth=core_data.get("max_object_fifo_depth", 0),
                col_id=col_id,
                row_id=row_id,
            )
            core.operator_types = operator_types
            return core

        if namespace == "zigzag":
            # ---- ZigZag path: full hierarchy via ZigZagCoreFactory ----
            zigzag_core = ZigZagCoreFactory(core_data).create(core_id, shared_mem_group_id=shared_mem_group_id)
            # ZigZagCoreFactory returns a raw zigzag Accelerator — upgrade to
            # our ZigZagCoreBackend subclass so the backend protocol methods
            # (get_memory_capacity, get_max_memory_bandwidth, get_ir) are available.
            zigzag_core.__class__ = ZigZagCoreBackend

            core = Core(
                backend=zigzag_core,
                core_id=zigzag_core.id,
                name=zigzag_core.name,
                core_type=core_type,
                utilization=core_data.get("utilization", 100),
                max_object_fifo_depth=core_data.get("max_object_fifo_depth", 0),
                col_id=col_id,
                row_id=row_id,
            )
            core.operator_types = operator_types
            return core

        raise ValueError(
            f"Unknown core namespace '{namespace}' in core type '{core_type}'. "
            f"Supported namespaces: {', '.join(sorted(ALLOWED_NAMESPACES))}"
        )

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

    def have_non_identical_shared_memory(self, cores: dict[int, Core]):
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
        return core_a.get_memory_capacity() != core_b.get_memory_capacity()

    def create_core_graph(self, cores: list[Core]):
        assert all(core.id == i for i, core in enumerate(cores))
        edges: list[tuple[Core, Core, dict[str, CommunicationLink]]] = []
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
                bus_instance = CommunicationLink("Any", "Any", bw, uec, bidirectional=True)
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
