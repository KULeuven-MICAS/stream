"""The picture of an accelerator names every core, memory, view, port and connection Stream reads from its
description."""

import re
from pathlib import Path

import pytest
import yaml

from stream.stages.parsing.accelerator_parser import parse_accelerator
from stream.visualization.hardware_graph import draw_hardware

SHIPPED = sorted(
    str(path)
    for path in Path("stream/inputs").glob("**/hardware/*.yaml")
    if "cores" in (yaml.safe_load(path.read_text()) or {})
)


def _text(path: Path) -> str:
    return " ".join(re.findall(r">([^<]*)</text>", path.read_text()))


@pytest.mark.parametrize("hardware", SHIPPED)
def test_every_core_memory_view_port_and_connection_is_drawn(tmp_path, hardware):
    accelerator = parse_accelerator(hardware)
    draw_hardware(accelerator, str(tmp_path / "hardware.svg"))
    text = _text(tmp_path / "hardware.svg")
    for core in accelerator.core_list:
        assert f"Core {core.id} · {core.name}" in text
    for memory in accelerator.memories.values():
        assert f"Memory {memory.name}" in text
        assert ", ".join(f"{core.id}.{instance}" for core, instance in memory.views) in text
    for port in accelerator.ports:
        assert f"port {port.key.port}:" in text
    declared = yaml.safe_load(Path(hardware).read_text())["connections"]
    assert text.count("Link") + text.count("Bus") == len(declared)


def test_a_core_shows_which_memory_holds_which_operand(tmp_path):
    """meta_prototype keeps its activations and outputs in one SRAM and its weights in another."""
    accelerator = parse_accelerator("stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml")
    draw_hardware(accelerator, str(tmp_path / "hardware.svg"))
    text = _text(tmp_path / "hardware.svg")
    assert "sram_1MB_A → meta_prototype_0: holds I1, O" in text
    assert "sram_1MB_W → meta_prototype_0_weights: holds I2" in text


def test_aie_shows_dma_channels_shared_neighbour_memories_and_two_way_links(tmp_path):
    """Every compute tile of a column of four reaches its one or two neighbours' memories without a DMA, and every
    link carries data both ways."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/single_col.yaml")
    draw_hardware(accelerator, str(tmp_path / "hardware.svg"))
    svg = (tmp_path / "hardware.svg").read_text()
    text = _text(tmp_path / "hardware.svg")
    assert "DMA 6 mm2s + 6 s2mm channels × 64 b" in text and "DMA 2 mm2s + 2 s2mm channels × 64 b" in text
    assert "port mm2s: 128 b/cycle, read any; its own" in text
    assert svg.count('stroke-dasharray="5 3"') == 2 * 3
    assert svg.count("marker-start") == 2 * len(
        yaml.safe_load(Path("stream/inputs/aie/hardware/single_col.yaml").read_text())["connections"]
    )
