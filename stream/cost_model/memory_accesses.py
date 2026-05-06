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

    def __repr__(self) -> str:
        """String representation of all accesses."""
        result = "CoreMemoryAccesses:\n"
        for core, tensors in self.accesses.items():
            for tensor, counts in tensors.items():
                result += f"  {core} - {tensor}: read={counts.read}, write={counts.write}\n"
        return result
