"""The conversion checker compares what two hardware files describe, whatever their memories and ports are called."""

import importlib.util
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FUSEMAX = ROOT / "stream/inputs/examples/hardware/fusemax.yaml"


def _checker():
    spec = importlib.util.spec_from_file_location(
        "check_hardware_conversion", ROOT / "scripts/check_hardware_conversion.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant(tmp_path: Path, change) -> str:
    """fusemax.yaml with its SRAM renamed and ``change`` applied, beside a copy of its core files."""
    shutil.copytree(FUSEMAX.parent / "cores", tmp_path / "cores")
    data = yaml.safe_load(FUSEMAX.read_text())
    data["memories"]["shared"] = data["memories"].pop("sram")
    for connection in data["connections"]:
        connection["ends"]["shared"] = connection["ends"].pop("sram")
    change(data)
    path = tmp_path / "fusemax.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_renamed_memories_describe_the_same_accelerator(tmp_path):
    checker = _checker()
    old = checker.summarize(str(FUSEMAX), None, None, None)
    new = checker.summarize(_variant(tmp_path, lambda data: None), None, None, None)
    assert checker.compare(old, new, 0.01) == []


def test_a_changed_port_and_connection_are_reported(tmp_path):
    def change(data):
        data["memories"]["dram"]["ports"]["rw_port_1"]["bandwidth"] = 1600
        data["connections"][0]["bandwidth"] = 256

    checker = _checker()
    old = checker.summarize(str(FUSEMAX), None, None, None)
    differences = checker.compare(old, checker.summarize(_variant(tmp_path, change), None, None, None), 0.01)
    assert sorted(line.split(":")[0] for line in differences) == [
        "connection added",
        "connection lost",
        "memory offchip",
        "memory offchip",
    ]
    assert any(line.startswith("memory offchip: port added: ") and "ceiling=1600" in line for line in differences)


def test_occupancy_may_swap_between_equal_memories():
    checker = _checker()
    solve = {"latency": 10, "energy_pj": 5.0}
    old = {"cores": {}, "memories": {}, "connections": [], "solve": {**solve, "occupancy": {"a": [3, 8], "b": [5, 8]}}}
    swapped = {**old, "solve": {**solve, "occupancy": {"a": [5, 8], "b": [3, 8]}}}
    fuller = {**old, "solve": {**solve, "occupancy": {"a": [6, 8], "b": [3, 8]}}}
    assert checker.compare(old, swapped, 0.01) == []
    assert checker.compare(old, fuller, 0.01) == [
        "solve: a holds 3 of 8 bits -> 6 of 8",
        "solve: b holds 5 of 8 bits -> 3 of 8",
    ]
