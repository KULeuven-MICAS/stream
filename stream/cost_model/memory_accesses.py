from collections import defaultdict

from stream.hardware.architecture.core import Core
from stream.hardware.architecture.memory import Memory
from stream.workload.tensor import Tensor


class AccessCount:
    """Class to track read and write counts."""

    def __init__(self):
        self.read = 0
        self.write = 0


class CoreMemoryAccesses:
    """
    Tracks read/write access counts per Core and Tensor, and per off-chip memory and Tensor.
    Uses nested defaultdict for automatic handling of missing keys.
    """

    def __init__(self):
        # Structure: {core: {tensor: AccessCount}}
        self.accesses: dict[Core, dict[Tensor, AccessCount]] = defaultdict(lambda: defaultdict(AccessCount))
        self.offchip: dict[Memory, dict[Tensor, AccessCount]] = defaultdict(lambda: defaultdict(AccessCount))

    def add_read(self, core: Core, tensor: Tensor, count: int = 1) -> None:
        """Add read accesses for a core-tensor pair."""
        self.accesses[core][tensor].read += count

    def add_write(self, core: Core, tensor: Tensor, count: int = 1) -> None:
        """Add write accesses for a core-tensor pair."""
        self.accesses[core][tensor].write += count

    def add_offchip_read(self, memory: Memory, tensor: Tensor, count: int) -> None:
        """Add read accesses of an off-chip memory for a tensor."""
        self.offchip[memory][tensor].read += count

    def add_offchip_write(self, memory: Memory, tensor: Tensor, count: int) -> None:
        """Add write accesses of an off-chip memory for a tensor."""
        self.offchip[memory][tensor].write += count

    def to_ir(self) -> dict:
        """Serialise the accesses for inspection: per core, per tensor, read/write counts.

        Counts are in *bandwidth-wide transfers* (a tensor tile's bits divided by the port bandwidth,
        times how many times it fires) -- the unit the cost model charges. Each tensor's ``size_bits``
        is carried alongside so a consumer can recover the bytes moved. ``offchip`` holds each off-chip
        memory's counts per tensor.
        """
        cores: list[dict] = []
        for core, tensors in self.accesses.items():
            try:
                read_bw = int(core.get_max_memory_bandwidth(type="read"))
                write_bw = int(core.get_max_memory_bandwidth(type="write"))
            except Exception:  # noqa: BLE001 -- a bandwidth-less core still reports its access counts
                read_bw = write_bw = 0
            tensor_rows: list[dict] = []
            core_read = core_write = 0
            for tensor, counts in tensors.items():
                core_read += counts.read
                core_write += counts.write
                tensor_rows.append(_tensor_row(tensor, counts))
            tensor_rows.sort(key=lambda t: -(t["read"] + t["write"]))
            cores.append(
                {
                    "core_id": core.id,
                    "core_name": core.name,
                    "kind": core.kind,
                    "read_bandwidth": read_bw,
                    "write_bandwidth": write_bw,
                    "read": core_read,
                    "write": core_write,
                    "total": core_read + core_write,
                    "tensors": tensor_rows,
                }
            )
        cores.sort(key=lambda c: c["core_id"])
        offchip = [
            {"memory": memory.name, **_tensor_row(tensor, counts)}
            for memory, tensors in self.offchip.items()
            for tensor, counts in tensors.items()
        ]
        return {"cores": cores, "offchip": offchip}

    def __repr__(self) -> str:
        """String representation of all accesses."""
        result = "CoreMemoryAccesses:\n"
        for core, tensors in self.accesses.items():
            for tensor, counts in tensors.items():
                result += f"  {core} - {tensor}: read={counts.read}, write={counts.write}\n"
        return result


def _tensor_row(tensor: Tensor, counts: AccessCount) -> dict:
    return {"tensor": tensor.name, "read": counts.read, "write": counts.write, "size_bits": tensor.size_bits()}
