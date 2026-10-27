from collections import defaultdict

from stream.hardware.architecture.core import Core
from stream.workload.tensor import Tensor


def _size_bits(tensor: Tensor) -> int:
    """The tensor's bits, 0 where it cannot say."""
    try:
        return int(tensor.size_bits())
    except Exception:  # noqa: BLE001 -- a stub tensor in a report still gets a row
        return 0


class AccessCount:
    """Class to track read and write counts."""

    def __init__(self):
        self.read = 0
        self.write = 0


class CoreMemoryAccesses:
    """
    Tracks read/write access counts for different Core and Tensor combinations, and those of the off-chip memory per
    Tensor. Uses nested defaultdict for automatic handling of missing keys.
    """

    def __init__(self):
        # Structure: {core: {tensor: AccessCount}}
        self.accesses: dict[Core, dict[Tensor, AccessCount]] = defaultdict(lambda: defaultdict(AccessCount))
        self.offchip: dict[Tensor, AccessCount] = defaultdict(AccessCount)
        # Bits one off-chip access moves, per direction ("read", "write"), as the counts were taken
        self.offchip_width: dict[str, float] = {}

    def add_read(self, core: Core, tensor: Tensor, count: int = 1) -> None:
        """Add read accesses for a core-tensor pair."""
        self.accesses[core][tensor].read += count

    def add_write(self, core: Core, tensor: Tensor, count: int = 1) -> None:
        """Add write accesses for a core-tensor pair."""
        self.accesses[core][tensor].write += count

    def add_offchip_read(self, tensor: Tensor, count: int) -> None:
        """Add read accesses of the off-chip memory for a tensor, each as wide as :attr:`offchip_width` says."""
        self.offchip[tensor].read += count

    def add_offchip_write(self, tensor: Tensor, count: int) -> None:
        """Add write accesses of the off-chip memory for a tensor, each as wide as :attr:`offchip_width` says."""
        self.offchip[tensor].write += count

    def to_ir(self) -> dict:
        """Serialise the accesses for inspection: per core, per tensor, read/write counts.

        Counts are in *bandwidth-wide transfers* (a tensor tile's bits divided by the port bandwidth,
        times how many times it fires) -- the unit the cost model charges. Each tensor's ``size_bits``
        is carried alongside so a consumer can recover the bytes moved; ``offchip`` holds the off-chip
        memory's per tensor, with the bits one access moves each way (``read_bandwidth``, ``write_bandwidth``),
        as the allocation models the memory: its measured rate, else its widest port serving the direction.
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
                tensor_rows.append(
                    {
                        "tensor": getattr(tensor, "name", str(tensor)),
                        "read": counts.read,
                        "write": counts.write,
                        "size_bits": _size_bits(tensor),
                    }
                )
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
            {
                "tensor": getattr(tensor, "name", str(tensor)),
                "read": counts.read,
                "write": counts.write,
                "size_bits": _size_bits(tensor),
                "read_bandwidth": self.offchip_width.get("read", 0),
                "write_bandwidth": self.offchip_width.get("write", 0),
            }
            for tensor, counts in self.offchip.items()
        ]
        return {"cores": cores, "offchip": offchip}

    def __repr__(self) -> str:
        """String representation of all accesses."""
        result = "CoreMemoryAccesses:\n"
        for core, tensors in self.accesses.items():
            for tensor, counts in tensors.items():
                result += f"  {core} - {tensor}: read={counts.read}, write={counts.write}\n"
        return result
