"""How the AIE code generator names the memory a transfer lands in."""

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")

from stream.stages.codegen.aie_code_generation import RUNTIME_TILE, _memory_tile  # noqa: E402
from stream.stages.parsing.accelerator_parser import parse_accelerator  # noqa: E402


def test_dram_is_named_by_the_runtime_tile_and_a_tile_memory_by_its_tile():
    """The lowering reaches DRAM from the runtime sequence, through the shim of the other end's column."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    assert _memory_tile(accelerator.offchip_memory, "aie2").data == RUNTIME_TILE
    assert _memory_tile(accelerator.memories["tile_2_3"], "aie2").data == "tile_2_3"
