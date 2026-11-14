"""An output buffer (a memory level below a core's top level that holds only outputs) keeps a running sum across the
fused loops summing it where every running sum on the core fits one instance; the top memory then holds only the
finished sum."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.architecture.backends.zigzag import OutputBufferUse
from stream.hardware.cost import RunningSumCall, kept_in_output_buffer
from stream.mapping.capacity_tiler import CapacityTiler
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage

from .test_capacity_tiler import _gemm

HARDWARE = "stream/inputs/examples/hardware"
_NODE = SimpleNamespace(type="Gemm", outputs=[SimpleNamespace(operand_type=SimpleNamespace(bitwidth=16))])


@dataclass(frozen=True)
class _Core:
    """A core accumulating in fp32 whose output levels below its top take ``shares`` of every call."""

    name: str
    shares: tuple[OutputBufferUse, ...]
    operand_precision: dict[str, str] = field(default_factory=lambda: {"accumulator": "fp32"}, hash=False)

    def output_buffers(self, elements: int, cme: object = None) -> tuple[OutputBufferUse, ...]:
        return self.shares


def _level(bits: int, outputs: int | None, resident: int = 0) -> OutputBufferUse:
    return OutputBufferUse(bits, outputs, resident, 0.0)


def _call(*cores: _Core, tiles: int = 1) -> RunningSumCall:
    return RunningSumCall(_NODE, cores, 0, tiles)  # type: ignore[arg-type]


def _parse(hardware: str, workload: str | None = None):
    stages = [AcceleratorParserStage, *([ONNXModelParserStage] if workload else []), LeafStage]
    ctx = StageContext.from_kwargs(accelerator=hardware, workload_path=workload, output_path=tempfile.mkdtemp())
    out = MainStage(stages, ctx).run()[0]
    return out.get("accelerator"), out.get("workload") if workload else None


@pytest.mark.parametrize(
    ("hardware", "core", "levels"),
    [
        ("fusemax.yaml", 0, [(32, None)]),
        ("eyeriss_like_quad_core.yaml", 0, [(65536, 8192), (128, None)]),
        ("tpu_like_quad_core.yaml", 4, []),
    ],
    ids=["fusemax-rf", "eyeriss-psum-buffer-and-rf", "pooling"],
)
def test_without_a_mapping_only_a_level_of_one_instance_takes_every_output(hardware, core, levels):
    """The output buffers, highest first, for a call of 8192 outputs whose mapping is not known:
    fusemax's 32-bit rf_O is in each of 65536 PEs, so it is not known to take outputs of its own; eyeriss's 65536-bit
    psum buffer is one instance taking all 8192 outputs, above an rf_16B in each PE; the pooling core has none."""
    accelerator, _ = _parse(f"{HARDWARE}/{hardware}")
    shares = accelerator.get_core(core).output_buffers(8192)
    assert [(s.bits, s.outputs) for s in shares] == levels
    assert all(s.resident == 8192 and s.offload == 0 for s in shares)


def test_a_core_holds_its_running_sums_only_where_all_fit_one_instance():
    """At fp32, core a's 1000-bit instance takes 20 outputs of each of two sums, 1280 bits, over its 1000, so it holds
    neither; b's takes one sum's 31, 992 bits, which it holds; c has no level holding outputs only."""
    a, b, c = _Core("a", (_level(1000, 20),)), _Core("b", (_level(1000, 31),)), _Core("c", ())
    held = kept_in_output_buffer({"x": _call(a), "y": _call(a, tiles=1), "z": _call(b), "w": _call(c)})
    assert held == {"z": {b: _level(1000, 31)}}


def test_a_core_leaves_room_for_what_another_node_s_call_holds_at_once():
    """Core a's 1000-bit instance fits a sum of 20 fp32 outputs, 640 bits, alone, but not next to the 12 another node's
    call on a holds there at once, 384 bits."""
    alone, shared = _Core("a", (_level(1000, 20, resident=12),)), _Core("a", (_level(1000, 20, resident=12),))
    assert kept_in_output_buffer({"x": _call(alone)}) == {"x": {alone: _level(1000, 20, resident=12)}}
    assert kept_in_output_buffer({"x": _call(shared), "y": _call(shared, tiles=0)}) == {}


def test_a_core_holds_its_sums_in_the_highest_level_where_each_instance_takes_outputs_of_its_own():
    """Core a's psum buffer (level 1) takes 64 fp32 outputs of the sum in its 1000 bits, too many; its register files
    (level 0) take 8, 256 bits of their 1000. Core b's psum buffer takes 16, 512 bits; it holds the sum there."""
    a = _Core("a", (_level(1000, 64), _level(1000, 8)))
    b = _Core("b", (_level(1000, 16), _level(1000, None)))
    assert kept_in_output_buffer({"x": _call(a), "y": _call(b)}) == {
        "x": {a: _level(1000, 8)},
        "y": {b: _level(1000, 16)},
    }


def test_a_sum_split_over_cores_is_held_only_where_every_core_holds_it():
    """A running sum on cores a and b, 25 fp32 outputs an instance on each: a's 1000-bit instance fits them, b's 500-bit
    one does not, so neither holds it, as every core of a node holds its tile alike."""
    a, b = _Core("a", (_level(1000, 25),)), _Core("b", (_level(500, 25),))
    assert kept_in_output_buffer({"x": _call(a, b)}) == {}


def _packed(tmp_path, m: int, l1_kib: int, served: str, result_buffer: int, link: int = 256) -> tuple[str, str]:
    """An int8 ``m`` x 512 by 512 x 64 Gemm on the packed core accumulating in int32, its result buffer of
    ``result_buffer`` bits serving ``served`` of its 16 x 16 x 4 array, its L1 reaching the off-chip memory at ``link``
    bits a cycle."""
    chip, gemm = _gemm(tmp_path, m, 512, 64, l1_kib=l1_kib)
    core = tmp_path / "packed.yaml"
    text = core.read_text().replace("{input: int8}", "{input: int8, accumulator: int32}")
    text = text.replace("served_dimensions: [D2, D3]", f"served_dimensions: [{served}]")
    core.write_text(text.replace("{size: 4096,", f"{{size: {result_buffer},"))
    path = tmp_path / "chip.yaml"
    path.write_text(path.read_text().replace("bandwidth: 256", f"bandwidth: {link}"))
    return chip, gemm


@pytest.mark.parametrize(
    ("served", "rows", "out"),
    [("D1, D2, D3", 32, 32 * 64 * 8), ("D1, D2, D3", 64, 64 * 64 * 32), ("D2, D3", 32, 32 * 64 * 32)],
    ids=["one-buffer-fits", "one-buffer-too-small", "sixteen-lanes"],
)
def test_the_tiler_holds_a_sum_below_the_top_memory_only_where_every_mapping_would(tmp_path, served, rows, out):
    """An int8 Gemm, 512-long contraction cut to 64, 64 output columns, on the packed core accumulating in int32 with
    65536 bits of result buffer. One buffer serving the whole array takes every output of the call whatever the
    mapping: a 32-row tile's 2048 int32 sums, 65536 bits, fit, so the L1 holds only the finished int8 tile, 32 x 64 x 8
    bits, next to a's and w's 64-long tiles, double-buffered; a 64-row tile's 4096 do not, so the L1 holds them at
    int32, 64 x 64 x 32 bits. Split in 16 lanes of 4096 bits, a mapping may spread the contraction over the lanes, so
    the tiler holds the 32-row tile's sums in the L1 at int32, 32 x 64 x 32 bits."""
    chip, gemm = _packed(tmp_path, rows, 1056, served, 65536 if served.startswith("D1") else 4096)
    accelerator, w = _parse(chip, gemm)
    (cn,) = w.get_computation_nodes()
    m, k, n = (w.leading_dim(d)[0] for d in w.get_dims(cn))
    resident = {m: rows, k: 64, n: 64}
    bits = CapacityTiler(w, accelerator).memory_bits((cn,), {cn: [accelerator.get_core(0)]}, {}, resident)["l1"]
    assert bits == out + 2 * (rows * 64 * 8 + 64 * 64 * 8)


_ROWS_OUTER = """
layers:
- {name: Gemm, core_allocation: [[0]]}
fused_groups:
- {name: G, layers: [Gemm], intra_core_tiling: [{dim: Gemm.D1, tile: 64}, {dim: Gemm.D0, tile: 32}]}
"""


@pytest.mark.parametrize(
    ("served", "result_buffer", "held", "out", "w", "reuse"),
    [
        ("D2, D3", 4096, True, 2 * 32 * 64 * 8, 512 * 64 * 8, 2),
        ("D2, D3", 2048, False, 2 * 32 * 64 * 32, 2 * 64 * 64 * 8, 1),
        ("", 4096, False, 2 * 32 * 64 * 32, 2 * 64 * 64 * 8, 1),
    ],
    ids=["lanes-hold-the-sum", "lanes-too-small", "reduction-spread-over-pes"],
)
def test_a_sum_the_lanes_hold_keeps_the_weights_reused_across_row_tiles(
    tmp_path, served, result_buffer, held, out, w, reuse
):
    """A 64 x 512 by 512 x 64 int8 Gemm on the packed core accumulating in int32, rows tiled to 32 outside a contraction
    tiled to 64, in a 44 KiB L1 (360448 bits) reaching the off-chip memory at 64 bits a cycle. Its cost backend sums
    the contraction within each of 16 lanes, their result buffer serving D2 and D3, and spreads 16 rows over the
    lanes: each takes 32 x 64 / 16 = 128 sums of the tile. At 4096 bits a lane, 128 int32 sums fit, so the L1 holds
    the finished int8 tile, double-buffered as the rows step it, 2 x 32 x 64 x 8 bits, and the whole of w,
    512 x 64 x 8 = 262144 bits, read once for both row tiles: with a's 32768 bits, 327680 in all. At 2048 bits a lane
    they do not fit; with the result buffer in each PE, the cost backend spreads the contraction over the PEs, which
    then hold partial sums of the same outputs. Either way the L1 holds the sum at int32, 2 x 32 x 64 x 32 = 131072
    bits, which leaves no room for the whole of w: it holds w's double-buffered 64 x 64 tiles, read again for each
    row tile."""
    chip, gemm = _packed(tmp_path, 64, 44, served, result_buffer, link=64)
    (tmp_path / "mapping.yaml").write_text(_ROWS_OUTER)
    estimate = evaluate_mapping(chip, gemm, str(tmp_path / "out"), str(tmp_path / "mapping.yaml"), SolveOptions())
    (ir,) = estimate.context.data["group_allocations"].values()
    space = estimate.context.get("allocation").space
    (l1,) = (m for m in ir["performance"]["memory_occupancy"] if m["memory"] == "l1")
    reused = {t["tensor"]: t["reuse_factor"] for t in ir["performance"]["tensor_reuse"]}
    assert {t.name for t in space.sums_in_output_buffer} == ({"out_1"} if held else set())
    assert {t["tensor"]: t["bits"] for t in l1["tensors"]} == {"a_1": 32768, "w_1": w, "out_1": out}
    assert reused["w_1"] == reuse


_ROWS_INNER = """
layers:
- {name: Gemm, core_allocation: [[0]]}
fused_groups:
- {name: G, layers: [Gemm], intra_core_tiling: [{dim: Gemm.D0, tile: 32}, {dim: Gemm.D1, tile: 64}]}
"""


def test_the_tiler_estimates_a_sum_the_output_buffer_holds_as_written_off_chip_once(tmp_path):
    """A 128 x 512 by 512 x 64 int8 Gemm on the packed core, 32-row tiles inside a contraction tiled to 64. The result
    buffer holds the 4 row tiles' sums across the 8 contraction tiles, so the tiler estimates, as the allocation
    moves, a, w and the output off chip once, (128 x 512 + 512 x 64 + 128 x 64) x 8 = 851968 bits, not the output
    2 x 8 - 1 = 15 times."""
    chip, gemm = _packed(tmp_path, 128, 17, "D1, D2, D3", 262144)
    accelerator, w = _parse(chip, gemm)
    (cn,) = w.get_computation_nodes()
    tiling = [{"dim": "Gemm.D0", "tile": 32}, {"dim": "Gemm.D1", "tile": 64}]
    estimate = CapacityTiler(w, accelerator).estimate((cn,), {cn: [accelerator.get_core(0)]}, {}, set(), tiling)
    assert estimate.offchip_bits == (128 * 512 + 512 * 64 + 128 * 64) * 8
    (tmp_path / "mapping.yaml").write_text(_ROWS_INNER)
    solved = evaluate_mapping(chip, gemm, str(tmp_path / "out"), str(tmp_path / "mapping.yaml"), SolveOptions())
    (accesses,) = solved.context.data["group_memory_accesses"].values()
    assert {t.name for t in solved.context.get("allocation").space.sums_in_output_buffer} == {"out_1"}
    assert {
        (r["tensor"], r["read"] * r["read_bandwidth"], r["write"] * r["write_bandwidth"]) for r in accesses["offchip"]
    } == {
        ("a", 128 * 512 * 8, 0),
        ("w", 512 * 64 * 8, 0),
        ("out", 0, 128 * 64 * 8),
    }
