"""Every example hardware runs the generic pipeline on a small conv pair and on a layer-fused SwiGLU, and every solve
must describe a sound dataflow. The hardware is discovered from the examples directory, so a new architecture joins the
matrix by adding its yaml; a new invariant joins by adding a line to ``_assert_sound_dataflow``."""

import math
import tempfile
import tomllib
from pathlib import Path

import pytest
from zigzag.utils import open_yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.workload.make_2_conv import make_2_conv_workload
from stream.inputs.testing.workload.make_attention_head import make_multi_head_attention
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.node import ComputationNode

HARDWARE = [pytest.param(str(p), id=p.stem) for p in sorted(Path("stream/inputs/examples/hardware").glob("*.yaml"))]

# The `just swiglu` FFN block, layer-fused: the solver costs one seq x hidden x embedding tile, then the iterations.
SWIGLU = {"seq_len": 256, "embedding_dim": 2048, "hidden_dim": 8192}
SWIGLU_TILING = [
    {"dim": "Gemm_Left.D1", "tile": 128},  # embedding: K of the branch gemms
    {"dim": "Gemm_Down.D2", "tile": 128},  # embedding: N of the down projection
    {"dim": "Gemm_Left.D2", "tile": 32},  # hidden
    {"dim": "Gemm_Left.D0", "tile": 16},  # seq, shared by the whole group
]


# Total and per-iteration cycles and energy in pJ of each example with one off-chip memory and no placements, under the
# default backend: a change to the cost model that moves any of them must update them here.
REFERENCE = {
    ("eyeriss_like_dual_core", "two_conv"): (80115, 80115, 73007120.0),
    ("eyeriss_like_dual_core", "swiglu"): (201851076, 581, 360554019771.73),
    ("eyeriss_like_quad_core", "two_conv"): (45693, 45693, 69452666.67),
    ("eyeriss_like_quad_core", "swiglu"): (201588932, 965, 360564068625.07),
    ("eyeriss_like_single_core", "two_conv"): (114999, 114999, 81211750.67),
    ("eyeriss_like_single_core", "swiglu"): (305135749, 424, 360548995345.07),
    ("fusemax", "two_conv"): (3168, 2148, 39247872.0),
    ("fusemax", "swiglu"): (26215424, 29, 231339773132.8),
    ("meta_prototype_dual_core_simd_offchip", "two_conv"): (16081, 16081, 13667377.16),
    ("meta_prototype_dual_core_simd_offchip", "swiglu"): (201850911, 416, 211031409295.36),
    ("simba", "two_conv"): (12613, 1849, 4367679.24),
    ("simba", "swiglu"): (201588922, 955, 29766476103.68),
    ("simba_small", "two_conv"): (13617, 2360, 4039968.36),
    ("simba_small", "swiglu"): (201588838, 871, 29763363143.68),
    ("tpu_like_quad_core", "two_conv"): (15316, 15208, 18617649.32),
    ("tpu_like_quad_core", "swiglu"): (201588878, 787, 212543061098.5),
    ("tpu_v7_ironwood", "two_conv"): (788, 64, 4927635.57),
    ("tpu_v7_ironwood", "swiglu"): (393891, 10, 63630107721.73),
}

# Multi-head attention over 8192 positions on TPU7x: total cycles, energy in pJ and the fused groups the generic
# pipeline makes, its attention fitting whole so that it is not split along its sequence.
ATTENTION_REFERENCE = (
    1838250,
    519800302403.58,
    [["proj_K", "proj_Q", "proj_V"], ["scores", "softmax_max", "softmax_exp", "softmax_sum", "softmax_div", "context"]],
)


def _assert_sound_dataflow(ctx, node_count: int) -> None:
    """Every node runs on a compute core, and each group's allocation is consistent: utilizations are fractions,
    no memory holds more than it has, energy is spent, and each transfer's layout orders all dims of a tile that fits
    its tensor, read and written in runs no longer than the tile by loops that visit the whole tile."""
    accelerator, mapping, workload = ctx.get("accelerator"), ctx.get("mapping"), ctx.get("workload")
    nodes = [n for n in workload.nodes if isinstance(n, ComputationNode)]
    assert len(nodes) == node_count
    for node in nodes:
        cores = {core.id for group in mapping.get(node).resource_allocation for core in group}
        assert cores and all(accelerator.get_core(core).kind == "compute" for core in cores), node.name

    assert ctx.get("total_latency") > 0
    for group, allocation in ctx.get("group_allocations").items():
        perf = allocation["performance"]
        aggregate = perf["aggregate"]
        assert not aggregate["degenerate"], group
        assert 0 < aggregate["latency_weighted_mac_spatial_utilization"] <= 1, group
        assert 0 < aggregate["end_to_end_mac_utilization"] <= 1, group
        assert perf["energy"]["total_pj"] > 0, group
        for memory in perf["memory_occupancy"]:
            assert memory["resident_bits"] <= memory["capacity_bits"], memory
        for layout in perf["layouts"]:
            dims = list(range(len(layout["full_shape"])))
            assert sorted(layout["source_order"]) == sorted(layout["target_order"]) == dims, layout
            assert all(t <= f for t, f in zip(layout["tile_shape"], layout["full_shape"], strict=True)), layout
            tile_bytes = math.prod(layout["tile_shape"]) * layout["element_bits"] / 8
            assert 0 < min(layout["read_run_bytes"], layout["write_run_bytes"]), layout
            assert max(layout["read_run_bytes"], layout["write_run_bytes"]) <= tile_bytes, layout
            for side in (layout["read"], layout["write"]):
                assert math.prod(size for size, _ in side["access"]) == math.prod(layout["tile_shape"]), layout


def _caption(values: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _record_metrics(record_metric, ctx, hparams: str) -> None:
    """The advisory metrics the CI comment compares against ``tests/golden_metrics.json``."""
    solution = ctx.get("allocation").solution
    aggregate = solution.performance["aggregate"]
    group_latencies = ctx.get("group_latencies")
    record_metric("total_latency", ctx.get("total_latency"))
    record_metric("group_latencies_max", max(group_latencies.values()))
    record_metric("objective", solution.solve_stats.objective)
    record_metric("mip_gap", solution.solve_stats.mip_gap)
    record_metric("solve_time_s", solution.solve_stats.solve_time_s)
    record_metric("mac_spatial_utilization", aggregate["latency_weighted_mac_spatial_utilization"])
    record_metric("end_to_end_mac_utilization", aggregate["end_to_end_mac_utilization"])
    record_metric("degenerate", aggregate["degenerate"])
    record_metric("workload_hparams", hparams)


def _assert_reference(ctx, hardware: str, workload: str) -> None:
    (group,) = ctx.get("group_allocations").values()
    found = (ctx.get("total_latency"), group["latency"]["per_iteration"], group["performance"]["energy"]["total_pj"])
    total, per_iteration, energy = REFERENCE[(Path(hardware).stem, workload)]
    assert found == (total, per_iteration, pytest.approx(energy, abs=0.01))


def _run(hardware: str, workload: str, options: SolveOptions | None = None):
    with tempfile.TemporaryDirectory() as tmpdir:
        return evaluate_mapping(hardware, workload, tmpdir, options=options).context


@pytest.mark.parametrize("hardware", HARDWARE)
def test_hardware_two_conv(hardware: str, two_conv, record_metric) -> None:
    """Two 3x3 convs, auto-tiled by the generic mapper."""
    ctx = _run(hardware, make_2_conv_workload(two_conv))
    _assert_sound_dataflow(ctx, node_count=2)
    _assert_reference(ctx, hardware, "two_conv")
    _record_metrics(record_metric, ctx, f"{_caption(vars(two_conv))}; generic auto-tiling")


@pytest.mark.parametrize("hardware", HARDWARE)
def test_hardware_swiglu(hardware: str, record_metric) -> None:
    """The five-node SwiGLU block, layer-fused under ``SWIGLU_TILING``; Silu and Mul land on a simd core where there
    is one."""
    options = SolveOptions(stage_options={"intra_core_tiling": SWIGLU_TILING})
    ctx = _run(hardware, make_small_swiglu_workload(**SWIGLU), options)
    _assert_sound_dataflow(ctx, node_count=5)
    _assert_reference(ctx, hardware, "swiglu")
    tiles = _caption({t["dim"]: t["tile"] for t in SWIGLU_TILING})
    _record_metrics(record_metric, ctx, f"{_caption(SWIGLU)}, bf16; fused tiles {tiles}")


def test_attention_on_one_off_chip_memory_keeps_its_reference_groups(tmp_path) -> None:
    ctx = _run(
        "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml", make_multi_head_attention(tmp_path / "mha.onnx", 8192)
    )
    groups = [[node.name for node in workload.get_computation_nodes()] for workload in ctx.get("sub_workloads")]
    energy = sum(group["performance"]["energy"]["total_pj"] for group in ctx.get("group_allocations").values())
    total, reference_energy, reference_groups = ATTENTION_REFERENCE
    assert (ctx.get("total_latency"), groups) == (total, reference_groups)
    assert energy == pytest.approx(reference_energy, abs=0.01)


@pytest.mark.parametrize("hardware", HARDWARE)
def test_hardware_parses(hardware: str) -> None:
    """Each hardware definition validates and builds an accelerator."""
    validator = AcceleratorValidator(open_yaml(hardware), hardware)
    assert validator.validate()
    assert AcceleratorFactory(validator.normalized_data).create() is not None


def test_every_hardware_file_ships_with_the_package():
    """A pip-installed package carries each hardware description, the testing ones too."""
    patterns = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["stream"]
    shipped = {path for pattern in patterns for path in Path("stream").glob(pattern)}
    assert set(Path("stream/inputs").glob("**/hardware/**/*.yaml")) <= shipped
