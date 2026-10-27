"""The picture of an accelerator names every core, memory, view, port and connection Stream reads from its
description."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from stream.parser.accelerator_validator import resolve_core_path
from stream.stages.parsing.accelerator_parser import parse_accelerator
from stream.visualization.hardware_graph import _core_lines, draw_hardware

SHIPPED = sorted(
    str(path)
    for path in Path("stream/inputs").glob("**/hardware/*.yaml")
    if "cores" in (yaml.safe_load(path.read_text()) or {})
)


def _text(path: Path) -> str:
    return " ".join(re.findall(r">([^<]*)</text>", path.read_text()))


def test_a_measured_memory_shows_its_model_in_place_of_its_ports(tmp_path):
    """Strix measures DRAM, so the model has no ports on it: the picture says so rather than listing declared widths
    the solve never uses."""
    accelerator = parse_accelerator("stream/inputs/aie/hardware/whole_array_strix.yaml")
    draw_hardware(accelerator, str(tmp_path / "hardware.svg"))
    text = _text(tmp_path / "hardware.svg")
    assert "measured 296.6 b/cycle, 148.3 one way; its ports not modelled" in text
    assert "port mm2s_c0" not in text and "port mm2s:" in text  # the tiles' DMAs are modelled, DRAM's shims are not


def test_a_memory_no_core_views_is_drawn_as_a_memory(tmp_path):
    """A buffer between the cores' SRAMs and DRAM, which no core computes out of, is a memory of kind ``memory``."""
    path = Path("stream/inputs/examples/hardware/eyeriss_like_single_core.yaml")
    data = yaml.safe_load(path.read_text())
    data["cores"] = {
        i: yaml.safe_load(Path(resolve_core_path(f, str(path.parent))).read_text()) for i, f in data["cores"].items()
    }
    data["memories"]["scratch"] = {
        "offchip": False,
        "views": [],
        "capacity_bits": 8 * 1024 * 1024,
        "ports": {"rw": {"bandwidth": 64, "serves": {"any": ["read", "write"]}, "read_energy": 0, "write_energy": 0}},
    }
    (offchip,) = [name for name, memory in data["memories"].items() if memory["offchip"]]
    data["connections"].append({"type": "link", "ends": {"scratch": ["rw"], offchip: []}, "bandwidth": 64})
    (tmp_path / "hw.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    accelerator = parse_accelerator(str(tmp_path / "hw.yaml"))
    assert accelerator.memories["scratch"].kind == "memory"
    draw_hardware(accelerator, str(tmp_path / "hardware.svg"))
    assert "Memory scratch kind memory, 1 MiB views none" in _text(tmp_path / "hardware.svg")


def test_a_core_holding_no_second_input_is_drawn_without_one():
    """A one-input core, its activations and outputs topping out in different instances, holds no I2 anywhere, and
    its backend says so; the picture lists what it holds rather than failing on an operand it has not got."""

    core = SimpleNamespace(
        id=0,
        name="eltwise",
        core_type="zigzag.compute",
        namespace="zigzag",
        col_id=None,
        top_instance_holding=lambda role: None if role == "input2" else "act",
        top_instances=lambda: frozenset({"act", "state"}),
        memory_banks={},
        max_object_fifo_depth=0,
    )
    memories = {"act": SimpleNamespace(name="l1"), "state": SimpleNamespace(name="l1_state")}
    accelerator = SimpleNamespace(memory_viewed=lambda core, instance: memories[instance])
    assert _core_lines(accelerator, core)[2:] == [
        "act → l1: holds I1, O",
        "state → l1_state: holds nothing it computes",
    ]


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
    assert "port mm2s: 128 b/cycle as 2 channels, read any; its own" in text
    assert "DMA 16 BDs × 64 steps" in text
    assert svg.count('stroke-dasharray="5 3"') == 2 * 3
    assert svg.count("marker-start") == 2 * len(
        yaml.safe_load(Path("stream/inputs/aie/hardware/single_col.yaml").read_text())["connections"]
    )
