"""Structural analysis of the workload graph: WL colouring and repeated-block detection."""

from stream.workload.structure.block_detect import BlockClass, find_repeated_blocks
from stream.workload.structure.wl import refine_colours

__all__ = ["BlockClass", "find_repeated_blocks", "refine_colours"]
