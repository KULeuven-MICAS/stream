"""A systolic array drains once per run: its tiles follow each other through the array, so a tile keeps its core busy
for its steady-state cycles only, and the results of the run's last tiles leaving the array add to its fill."""

import shutil

import yaml

from stream.api import SolveOptions, evaluate_mapping

HARDWARE = "stream/inputs/examples/hardware"
SWIGLU = "stream/inputs/aie/workload/swiglu_256_512_2048.onnx"


def _broadcast_copy(tmp_path):
    """TPU7x with its MXUs broadcasting their operands instead of moving them through the array."""
    shutil.copytree(HARDWARE, tmp_path / "hardware")
    mxu = tmp_path / "hardware" / "cores" / "tpu_v7_mxu.yaml"
    with open(mxu, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["operational_array"].pop("systolic_dimensions") == ["D1", "D2"]
    mxu.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(tmp_path / "hardware" / "tpu_v7_ironwood.yaml")


def _solve(hardware, tmp_path):
    estimate = evaluate_mapping(hardware, SWIGLU, str(tmp_path), options=SolveOptions(artifacts=False))
    entries = {
        (node.name, core.id): entry
        for node, costs in estimate.context.get("cost_lut").lut.items()
        for core, entry in costs.items()
    }
    fills = [allocation["latency"]["fill"] for allocation in estimate.context.get("group_allocations").values()]
    return estimate, entries, fills


def test_a_systolic_array_drains_once_per_run(tmp_path):
    systolic, systolic_entries, systolic_fills = _solve(f"{HARDWARE}/tpu_v7_ironwood.yaml", tmp_path / "systolic")
    broadcast, broadcast_entries, broadcast_fills = _solve(_broadcast_copy(tmp_path), tmp_path / "broadcast")

    gemms = [key for key in systolic_entries if key[0].startswith("Gemm")]
    assert gemms
    for key in gemms:
        entry = systolic_entries[key]
        drain = entry.metadata["drain_cycles"]
        assert drain > 0
        assert entry.latency_total == entry.cme.latency_total2 - drain
        assert entry.latency_total == broadcast_entries[key].latency_total
        assert broadcast_entries[key].metadata["drain_cycles"] == 0

    added = [s - b for s, b in zip(systolic_fills, broadcast_fills, strict=True)]
    assert all(a > 0 for a in added)
    assert systolic.cycles - broadcast.cycles == sum(added)
