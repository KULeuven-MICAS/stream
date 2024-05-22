from typing import Any
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.hardware.architecture.noc.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.noc.bus import get_bus
from zigzag.hardware.architecture.Core import Core


from zigzag.parser.accelerator_factory import CoreFactory


class AcceleratorFactory:
    """! Converts valid user-provided accelerator data into an `Accelerator` instance"""

    def __init__(self, data: dict[str, Any]):
        """! Generate an `Accelerator` instance from the validated user-provided data."""
        self.data = data

    def create(self) -> Accelerator:
        """! Create an Accelerator instance from the user-provided data."""
        cores: list[Core] = []
        for core_id, core_data in enumerate(self.data["cores"]):
            core_factory = CoreFactory(core_data)
            core = core_factory.create(core_id)
            cores.append(core)

        if self.data["graph"]["type"] == "2d_mesh":
            cores_graph = self.create_2d_mesh(cores)
        elif self.data["graph"]["type"] == "bus":
            cores_graph = self.create_bus(cores)
        else:
            raise ValueError(f"Invalid graph type {self.data['graph']['type']}.")

        offchip_core_id: int | None = self.data["graph"]["offchip_core_id"]
        return Accelerator(name=self.data["name"], cores=cores_graph, offchip_core_id=offchip_core_id)

    def create_2d_mesh(self, cores: list[Core]):
        pooling_core_id: int | None = self.data["graph"]["pooling_core_id"]
        simd_core_id: int | None = self.data["graph"]["simd_core_id"]
        offchip_core_id: int | None = self.data["graph"]["offchip_core_id"]

        # Grab special cores at given indices
        pooling_core = cores[pooling_core_id] if pooling_core_id is not None else None
        simd_core = cores[simd_core_id] if simd_core_id is not None else None
        offchip_core = cores[offchip_core_id] if offchip_core_id is not None else None

        # Remove special corse from 'regular' core list
        if pooling_core is not None:
            cores.remove(pooling_core)
        if simd_core is not None:
            cores.remove(simd_core)
        if offchip_core is not None:
            cores.remove(offchip_core)

        cores_graph = get_2d_mesh(
            cores=cores,
            nb_rows=self.data["graph"]["nb_rows"],
            nb_cols=self.data["graph"]["nb_cols"],
            bandwidth=self.data["graph"]["bandwidth"],
            unit_energy_cost=self.data["graph"]["unit_energy_cost"],
            pooling_core=pooling_core,
            simd_core=simd_core,
            offchip_core=offchip_core,
        )

        return cores_graph

    def create_bus(self, cores: list[Core]):
        pooling_core_id: int | None = self.data["graph"]["pooling_core_id"]
        simd_core_id: int | None = self.data["graph"]["simd_core_id"]
        offchip_core_id: int | None = self.data["graph"]["offchip_core_id"]

        # Grab special cores at given indices
        pooling_core = cores[pooling_core_id] if pooling_core_id is not None else None
        simd_core = cores[simd_core_id] if simd_core_id is not None else None
        offchip_core = cores[offchip_core_id] if offchip_core_id is not None else None

        # Remove special corse from 'regular' core list
        if pooling_core is not None:
            cores.remove(pooling_core)
        if simd_core is not None:
            cores.remove(simd_core)
        if offchip_core is not None:
            cores.remove(offchip_core)

        cores_graph = get_bus(
            cores=cores,
            bandwidth=self.data["graph"]["bandwidth"],
            unit_energy_cost=self.data["graph"]["unit_energy_cost"],
            pooling_core=pooling_core,
            simd_core=simd_core,
            offchip_core=offchip_core,
        )

        return cores_graph
