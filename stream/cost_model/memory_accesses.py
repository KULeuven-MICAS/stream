from collections import defaultdict

from stream.hardware.architecture.core import Core
from stream.workload.tensor import Tensor


class AccessCount:
    """Class to track read and write counts."""

    def __init__(self):
        self.read = 0
        self.write = 0


class CoreMemoryAccesses:
    """
    Tracks read/write access counts for different Core and Tensor combinations.
    Uses nested defaultdict for automatic handling of missing keys.
    """

    def __init__(self):
        # Structure: {core: {tensor: AccessCount}}
        self.accesses: dict[Core, dict[Tensor, AccessCount]] = defaultdict(lambda: defaultdict(AccessCount))

    def add_read(self, core: Core, tensor: Tensor, count: int = 1) -> None:
        """Add read accesses for a core-tensor pair."""
        self.accesses[core][tensor].read += count

    def add_write(self, core: Core, tensor: Tensor, count: int = 1) -> None:
        """Add write accesses for a core-tensor pair."""
        self.accesses[core][tensor].write += count

    def get_reads(self, core: Core, tensor: Tensor) -> int:
        """Get read count for a core-tensor pair."""
        return self.accesses[core][tensor].read

    def get_writes(self, core: Core, tensor: Tensor) -> int:
        """Get write count for a core-tensor pair."""
        return self.accesses[core][tensor].write

    def get_total(self, core: Core, tensor: Tensor) -> int:
        """Get total (read + write) accesses for a core-tensor pair."""
        reads = self.accesses[core][tensor].read
        writes = self.accesses[core][tensor].write
        return reads + writes

    def to_ir(self, offchip_core_id: int | None = None) -> dict:
        """Serialise the accesses for inspection: per core, per tensor, read/write counts.

        Counts are in *bandwidth-wide transfers* (a tensor tile's bits divided by the port bandwidth,
        times how many times it fires) -- the unit the cost model charges. Each tensor's ``size_bits``
        is carried alongside so a consumer can recover the bytes moved, and the port bandwidths and an
        ``is_offchip`` flag let it separate off-chip (DRAM/HBM) traffic from on-chip.
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
                size_bits = 0
                try:
                    size_bits = int(tensor.size_bits())
                except Exception:  # noqa: BLE001
                    pass
                tensor_rows.append(
                    {
                        "tensor": getattr(tensor, "name", str(tensor)),
                        "read": counts.read,
                        "write": counts.write,
                        "size_bits": size_bits,
                    }
                )
            tensor_rows.sort(key=lambda t: -(t["read"] + t["write"]))
            kind = getattr(core, "type", None) or getattr(core, "kind", None)
            is_offchip = (offchip_core_id is not None and core.id == offchip_core_id) or kind in ("offchip", "shim")
            cores.append(
                {
                    "core_id": core.id,
                    "core_name": getattr(core, "name", f"core_{core.id}"),
                    "kind": kind,
                    "is_offchip": bool(is_offchip),
                    "read_bandwidth": read_bw,
                    "write_bandwidth": write_bw,
                    "read": core_read,
                    "write": core_write,
                    "total": core_read + core_write,
                    "tensors": tensor_rows,
                }
            )
        cores.sort(key=lambda c: c["core_id"])
        return {"cores": cores}

    def __repr__(self) -> str:
        """String representation of all accesses."""
        result = "CoreMemoryAccesses:\n"
        for core, tensors in self.accesses.items():
            for tensor, counts in tensors.items():
                result += f"  {core} - {tensor}: read={counts.read}, write={counts.write}\n"
        return result
