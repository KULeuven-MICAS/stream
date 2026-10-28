"""How the AIE code generator names the memory a transfer lands in."""

import pytest

pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")

from stream.stages.codegen.aie_code_generation import RUNTIME_TILE, AIECodeGenerationStage, _memory_tile  # noqa: E402
from stream.stages.parsing.accelerator_parser import parse_accelerator  # noqa: E402


def _stage(accelerator, trace_size: int) -> AIECodeGenerationStage:
    stage = AIECodeGenerationStage.__new__(AIECodeGenerationStage)
    stage.ctx = {"accelerator": accelerator}
    stage.trace_size = trace_size
    return stage


def test_the_shim_descriptors_and_iterations_are_the_off_chip_memorys_dma():
    """The shim limits come from DRAM's DMA, one descriptor fewer with a trace, and are zero without a DMA."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    assert _stage(accelerator, trace_size=0)._shim_limits() == (16, 64)
    assert _stage(accelerator, trace_size=8192)._shim_limits() == (15, 64)
    accelerator.offchip_memory.dma = None
    assert _stage(accelerator, trace_size=0)._shim_limits() == (0, 0)


def test_dram_is_named_by_the_runtime_tile_and_a_tile_memory_by_its_tile():
    """The lowering reaches DRAM from the runtime sequence, through the shim of the other end's column."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array.yaml")
    assert _memory_tile(accelerator.offchip_memory).data == RUNTIME_TILE
    assert _memory_tile(accelerator.memories["tile_2_3"]).data == "tile_2_3"
