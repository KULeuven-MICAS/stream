from typing import Any
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.classes.hardware.architecture.noc.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.noc.bus import get_bus


from zigzag.hardware.architecture.Core import Core
#from stream.classes.hardware.architecture.stream_core import Core

from zigzag.parser.accelerator_factory import CoreFactory


class AcceleratorFactory:
    """! Converts valid user-provided accelerator data into an `Accelerator` instance"""

    def __init__(self, data: dict[str, Any]):
        """! Generate an `Accelerator` instance from the validated user-provided data."""
        self.data = data

    def create(self) -> Accelerator:
        """! Create an Accelerator instance from the user-provided data."""
        cores: list[Core] = []
        cores_type: list[int] = []
        if "cores_type" in self.data:
            for core_id, core_type in enumerate(self.data["cores_type"]):
                cores_type.append(core_type)

        i = 0
        for core_id, core_data in enumerate(self.data["cores"]):
            core_factory = CoreFactory(core_data)
            core = core_factory.create(core_id)
            if len(cores_type) > 0:
                core.core_type = cores_type[i]
            else:  # force all cores to be compute
                core.core_type = 0
            i += 1
            cores.append(core)
        
        if self.data["graph"]["type"] == "2d_mesh":
            cores_graph = self.create_2d_mesh(cores)
        elif self.data["graph"]["type"] == "bus":
            cores_graph = self.create_bus(cores)
        else:
            raise ValueError(f"Invalid graph type {self.data['graph']['type']}.")

        offchip_core_id: int | None = self.data["graph"]["offchip_core_id"]

        if "parallel_links_flag" in self.data["graph"]:
            parallel_links_flag=self.data["graph"]["parallel_links_flag"]
        else:
            parallel_links_flag=True # default is true
        
        return Accelerator(name=self.data["name"], cores=cores_graph, nb_rows=self.data["graph"]["nb_rows"], nb_cols=self.data["graph"]["nb_cols"], parallel_links_flag=parallel_links_flag, offchip_core_id=offchip_core_id)

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

        if "parallel_links_flag" in self.data["graph"]:
            parallel_links_flag=self.data["graph"]["parallel_links_flag"]
        else:
            parallel_links_flag=None

        if "use_shared_mem_flag" in self.data["graph"]:
            use_shared_mem_flag=self.data["graph"]["use_shared_mem_flag"]
        else:
            use_shared_mem_flag=None

        if "offchip_read_channels_num" in self.data["graph"]:
            offchip_read_channels_num=self.data["graph"]["offchip_read_channels_num"]
        else:
            offchip_read_channels_num=1

        if "offchip_write_channels_num" in self.data["graph"]:
            offchip_write_channels_num=self.data["graph"]["offchip_write_channels_num"]
        else:
            offchip_write_channels_num=1

        if "memTile_read_channels_num" in self.data["graph"]:
            memTile_read_channels_num=self.data["graph"]["memTile_read_channels_num"]
        else:
            memTile_read_channels_num=0

        if "memTile_write_channels_num" in self.data["graph"]:
            memTile_write_channels_num=self.data["graph"]["memTile_write_channels_num"]
        else:
            memTile_write_channels_num=0

        cores_graph = get_2d_mesh(
            cores=cores,
            nb_rows=self.data["graph"]["nb_rows"],
            nb_cols=self.data["graph"]["nb_cols"],
            bandwidth=self.data["graph"]["bandwidth"],
            unit_energy_cost=self.data["graph"]["unit_energy_cost"],
            pooling_core=pooling_core,
            simd_core=simd_core,
            offchip_core=offchip_core,
            parallel_links_flag=parallel_links_flag,
            use_shared_mem_flag=use_shared_mem_flag,
            offchip_read_channels_num = offchip_read_channels_num,
            offchip_write_channels_num = offchip_write_channels_num,
            memTile_read_channels_num = memTile_read_channels_num,
            memTile_write_channels_num = memTile_write_channels_num,
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
