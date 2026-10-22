import re
from typing import Any

from zigzag.parser.accelerator_factory import AcceleratorFactory as ZigZagCoreFactory

from stream.compiler.kernels.library import KernelLibrary
from stream.cost_model.bandwidth import BandwidthModel
from stream.hardware.architecture.accelerator import Accelerator, MemoryGraph
from stream.hardware.architecture.backends import AIE2CoreBackend, ZigZagCoreBackend
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.hardware.ports import MemoryBanks, PortKey
from stream.parser.accelerator_validator import PORT_REF_REGEX, parse_port_ref
from stream.parser.core_validator import ALLOWED_KINDS, ALLOWED_NAMESPACES, CoreValidatorRegistry


class AcceleratorFactory:
    """! Converts valid user-provided accelerator data into an `Accelerator` instance"""

    def __init__(self, data: dict[str, Any]):
        """! Generate an `Accelerator` instance from the validated user-provided data."""
        self.data = data

    def create(self) -> Accelerator:
        """! Create an Accelerator instance from the user-provided data."""
        coordinates = self.data["core_coordinates"]
        cores = {
            core_id: self.create_core(core_data, core_id, coordinates.get(core_id))
            for core_id, core_data in self.data["cores"].items()
        }
        memories = {
            name: Memory(name, tuple((cores[core_id], instance) for core_id, instance in views))
            for name, views in self.data["memories"].items()
        }
        measured: dict[Memory, BandwidthModel] = {}
        port_models: dict[PortKey, BandwidthModel] = {}
        for key, description in self.data["bandwidth"].items():
            if re.match(PORT_REF_REGEX, key):
                port_models[parse_port_ref(key)] = BandwidthModel.from_description(description)
            else:
                measured[memories[key]] = BandwidthModel.from_description(description)

        accelerator = Accelerator(
            name=self.data["name"],
            cores=list(cores.values()),
            memories=list(memories.values()),
            network=self.create_network(memories),
            offchip_core_id=self.data["offchip_core_id"],
            kernel_library=KernelLibrary.load(self.data.get("kernel_library")),
            bandwidth=measured,
            reconfiguration=self.data.get("reconfiguration"),
            port_bandwidth=port_models,
        )
        if accelerator.port_bandwidth:
            # Building the registry rejects a port key that names no port a core's backend models.
            _ = accelerator.ports
        return accelerator

    def create_core(
        self,
        core_data: dict[str, Any],
        core_id: int,
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
            dma = core_data.get("dma", {})
            backend = AIE2CoreBackend(
                memory_capacity_bits=mem["capacity"],
                bandwidth_min=mem.get("bandwidth_min", 0),
                bandwidth_max=mem.get("bandwidth_max", 0),
                dma_mm2s=dma.get("mm2s", 0),
                dma_s2mm=dma.get("s2mm", 0),
                dma_channel_bits=dma.get("channel_bits", 0),
                dma_buffer_descriptors=dma.get("buffer_descriptors", 0),
                dma_iterations=dma.get("iterations", 0),
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
            return _with_stream_fields(core, core_data, operator_types)

        if namespace == "zigzag":
            # ---- ZigZag path: full hierarchy via ZigZagCoreFactory ----
            # The memories are the accelerator's (`memories`): ZigZag only tells its instances apart by this id.
            zigzag_core = ZigZagCoreFactory(core_data).create(core_id, shared_mem_group_id=core_id)
            # ZigZagCoreFactory returns a raw zigzag Accelerator — upgrade to
            # our ZigZagCoreBackend subclass so the backend protocol methods
            # (top_memory_capacity, get_max_memory_bandwidth, get_ir) are available.
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
            return _with_stream_fields(core, core_data, operator_types)

        raise ValueError(
            f"Unknown core namespace '{namespace}' in core type '{core_type}'. "
            f"Supported namespaces: {', '.join(sorted(ALLOWED_NAMESPACES))}"
        )

    def create_network(self, memories: dict[str, Memory]) -> MemoryGraph:
        """The memories, with an edge each way for every link and between every two members of a bus, all of a bus's
        edges one shared `CommunicationLink`."""
        edges: list[tuple[Memory, Memory, dict[str, CommunicationLink]]] = []
        seen: dict[tuple[str, tuple[str, ...]], int] = {}
        for connection in self.data["connections"]:
            kind, bandwidth, energy = connection["type"], connection["bandwidth"], connection["unit_energy_cost"]
            ends = tuple(memories[name] for name in connection["memories"])
            key = (kind, tuple(sorted(memory.name for memory in ends)))
            parallel = seen[key] = seen.get(key, -1) + 1
            if kind == "link":
                a, b = ends
                edges += [
                    (a, b, {"cl": CommunicationLink.link(a, b, bandwidth, energy, parallel)}),
                    (b, a, {"cl": CommunicationLink.link(b, a, bandwidth, energy, parallel)}),
                ]
            else:
                bus = CommunicationLink.bus(ends, bandwidth, energy, parallel)
                edges += [(a, b, {"cl": bus}) for a in ends for b in ends if a is not b]
        return MemoryGraph(edges)


def _with_stream_fields(core: Core, core_data: dict[str, Any], operator_types: list[str] | None) -> Core:
    """``core`` with the fields Stream reads beside any backend's: the operators it runs, its operand precision and
    the banks of its memory."""
    core.operator_types = operator_types
    core.operand_precision = core_data.get("operand_precision") or {}
    core.banks = MemoryBanks(**core_data["banks"]) if core_data.get("banks") else None
    return core
