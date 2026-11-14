"""The capacity-aware intra-core tiler streams a resident weight on overflow, leaves fitting groups alone."""

import math
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import onnx
import pytest
import yaml
from conv_windows import conv, conv_chain, onnx_graph
from onnx import TensorProto, helper
from tpu_quad import tpu_quad

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.cost import running_sum_bits
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.inputs.testing.workload.make_attention_head import make_multi_head_attention
from stream.inputs.testing.workload.make_decode_attention import make_decode_attention
from stream.mapping.capacity_tiler import CapacityTiler, TilingEstimate, _divisors_desc
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.mapping.mapping import whole_bits
from stream.opt.allocation.constraint_optimization.hardware import build_hardware_facts
from stream.parser.mapping_validator import MappingValidator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.normalization import expand_normalizations, source_operator

_TPU_QUAD = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
# A single large Gemm (K=8192) whose weight slice overflows the 2 MB matmul core when kept resident.
_GEMM = "stream/inputs/aie/workload/gemm_256_8192_2048.onnx"
# A SwiGLU small enough to fit the same cores without any streaming.
_SWIGLU_FITS = "stream/inputs/aie/workload/swiglu_256_512_2048.onnx"
_TWO_CONV = "stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx"
_TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"


def _parse(hardware: str, workload: str):
    ctx = StageContext.from_kwargs(accelerator=hardware, workload_path=workload, output_path=tempfile.mkdtemp())
    ctxs = MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()
    return ctxs[0].get("accelerator"), ctxs[0].get("workload")


def _worst_core_ratio(gen: GenericMappingGenerator, sub, cns, tiling) -> float:
    """Worst memory footprint / room the tiler measures for ``tiling``."""
    tiler = CapacityTiler(sub, gen.accelerator)
    cores = {cn: gen._select_cores_for_node(cn) for cn in cns}
    unroll = gen._inter_core_unrolling(sub, cns)
    dims = {sub.leading_dim(d)[0] for cn in cns for d in sub.get_dims(cn)}
    per_core = {d: sub.get_dimension_size(d) // unroll.get(d, 1) for d in dims}
    resident = per_core | tiler._seed_resident(cns, tiling, per_core)
    held = tiler.memory_bits(cns, cores, unroll, resident)
    capacities = tiler._capacities(cns, cores)
    return max(bits / capacities[m] for m, bits in held.items())


def test_a_running_sum_takes_its_accumulator_precision(tmp_path):
    """A 64 x 256 by 256 x 64 int8 Gemm on a core with an int32 accumulator holds its 64 x 64 output at 32 bits with
    the contraction tiled to 64, next to the double-buffered 64 x 64 int8 tiles of a and w, 4 x 32768 bits; and at int8
    with the contraction whole, next to a and w whole, 2 x 131072 bits."""
    chip, gemm = _gemm(tmp_path, 64, 256, 64, l1_kib=1056)
    core = tmp_path / "packed.yaml"
    core.write_text(core.read_text().replace("{input: int8}", "{input: int8, accumulator: int32}"))
    acc, w = _parse(chip, gemm)
    (cn,) = w.get_computation_nodes()
    m, k, _ = (w.leading_dim(d)[0] for d in w.get_dims(cn))
    whole = {w.leading_dim(d)[0]: w.get_dimension_size(d) for d in w.get_dims(cn)}
    tiler = CapacityTiler(w, acc)
    cores = {cn: [acc.get_core(0)]}
    outputs = [
        tiler.memory_bits((cn,), cores, {}, resident, None, (cn,), frozenset({m}))["l1"]
        for resident in (whole | {k: 64}, whole)
    ]
    assert outputs == [64 * 64 * 32 + 4 * 64 * 64 * 8, 64 * 64 * 8 + 2 * 64 * 256 * 8]


def _mlp(tmp_path) -> str:
    """X[64,128] @ W1[128,256] -> H1[64,256] @ W2[256,128] -> Y[64,128], all bf16."""
    shapes = {"X": [64, 128], "W1": [128, 256], "W2": [256, 128], "Y": [64, 128]}
    info = {name: helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape) for name, shape in shapes.items()}
    nodes = [
        helper.make_node("Gemm", ["X", "W1"], ["H1"], name="Gemm1"),
        helper.make_node("Gemm", ["H1", "W2"], ["Y"], name="Gemm2"),
    ]
    graph = helper.make_graph(nodes, "mlp", [info["X"], info["W1"], info["W2"]], [info["Y"]])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "mlp.onnx")
    return str(tmp_path / "mlp.onnx")


@pytest.mark.parametrize(("inner", "bits", "fits"), [("channels", 35328, False), ("columns", 33024, True)])
def test_a_halo_kept_across_an_inner_loop_decides_whether_the_tiles_fit(inner, bits, fits):
    """Two bf16 3x3 convolutions (4 -> 8 -> 8 channels on 12 x 16) on one eyeriss core, in tiles of 2 channels by 4
    columns, hold 2064 elements, 33024 bits. Looped channels inside columns, the column loop also keeps the 2-column
    halo of the 3 other channel slices, 3 x 2 x 12 x 2 = 144 elements: 35328 bits, over a budget of 34176."""
    shapes = {"input": [1, 4, 12, 16], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}
    path = onnx_graph([conv("Conv1", "input", "w1", "t1"), conv("Conv2", "t1", "w2", "out")], shapes, ["out"])
    acc, w = _parse("stream/inputs/examples/hardware/eyeriss_like_single_core.yaml", path)
    conv1, conv2 = sorted(w.get_computation_nodes(), key=lambda n: n.name)
    channels, columns = (w.leading_dim(w.get_dims(conv1)[i])[0] for i in (6, 1))
    resident = {d: w.get_dimension_size(d) for d in w.unique_dimensions()[0]} | {channels: 2, columns: 4}
    cores = {conv1: [acc.get_core(0)], conv2: [acc.get_core(0)]}
    order = [channels, columns] if inner == "channels" else [columns, channels]
    tiler = CapacityTiler(w, acc)
    tiler.fill_fraction = 34176 / tiler._capacities((conv1, conv2), cores)["eyeriss_like"]
    held = tiler.memory_bits((conv1, conv2), cores, {}, resident, order=order)
    assert held == {"eyeriss_like": bits}
    assert (bits <= tiler._capacities((conv1, conv2), cores)["eyeriss_like"]) is fits


def test_a_reader_of_a_finished_sum_holds_it_at_its_type(tmp_path):
    """Gemm1 split in two on its contraction and in two on its hidden dim holds a 64 x 128 partial H1 at the int32
    accumulator, 262144 bits, next to X and W1 in bf16, 65536 and 131072 bits. Gemm2, split in four on its outputs,
    reads all 64 x 256 of the finished H1 at bf16, 262144 bits, one buffer with Gemm1's tile, next to W2 and Y, 131072
    and 32768 bits: 622592 bits a core."""
    acc, w = _parse(_TPU_QUAD, _mlp(tmp_path))
    gemm1, gemm2 = sorted(w.get_computation_nodes(), key=lambda n: n.name)
    contraction, hidden, outputs = (
        w.leading_dim(w.get_dims(node)[i])[0] for node, i in ((gemm1, 1), (gemm1, 2), (gemm2, 2))
    )
    full = {d: w.get_dimension_size(d) for d in w.unique_dimensions()[0]}
    unroll = {contraction: 2, hidden: 2, outputs: 4}
    resident = {d: n // unroll.get(d, 1) for d, n in full.items()}
    cores = [acc.get_core(i) for i in range(4)]
    split = {gemm1: {contraction: 2, hidden: 2}, gemm2: {outputs: 4}}
    held = CapacityTiler(w, acc).memory_bits((gemm1, gemm2), dict.fromkeys(split, cores), unroll, resident, split)
    assert held == {f"tpu_like_{i}": 622592 for i in range(4)}


@pytest.mark.parametrize(
    ("conv2_split", "reduction", "bits"),
    [("D6", "int32", None), ("D5", None, None), ("D5", "int32", 32)],
    ids=["output-channels", "input-channels", "input-channels-int32-reduction"],
)
def test_the_tiler_and_the_allocation_agree_on_a_running_sum(tmp_path, conv2_split, reduction, bits):
    """Two bf16 3x3 convolutions over four cores: the second split on its output channels (D6) finishes a quarter on
    each core, no running sum; split on its input channels (D5), each core holds a partial sum at the format the cores
    send it to each other at: the bf16 type, or a declared int32 reduction."""
    hardware = tpu_quad(tmp_path, reduction)
    config = TwoConvWorkloadConfig(1, 8, 8, 8, 16, 16, 3, "bf16", "bf16")
    workload = make_2_conv_workload(config, str(tmp_path))
    acc, w = _parse(hardware, workload)
    conv1, conv2 = sorted(w.get_computation_nodes(), key=lambda n: n.name)
    split = {
        conv1: w.leading_dim(w.get_dims(conv1)[6])[0],
        conv2: w.leading_dim(w.get_dims(conv2)[int(conv2_split[1])])[0],
    }
    cores = [acc.get_core(i) for i in range(4)]
    full = {d: w.get_dimension_size(d) for d in w.unique_dimensions()[0]}
    unroll = dict.fromkeys(split.values(), 4)
    resident = {d: n // unroll.get(d, 1) for d, n in full.items()}
    node_unroll = {cn: {dim: 4} for cn, dim in split.items()}
    tiler = CapacityTiler(w, acc)
    sizes = tiler._node_sizes(conv2, dict.fromkeys(split, cores), unroll, resident, node_unroll)
    assert {d for d, size in sizes.items() if size < full[d]} == {split[conv2]}
    assert running_sum_bits(w, conv2, cores, set(), {split[conv2]}) == bits
    layers = [
        {"name": name, "core_allocation": [[0, 1, 2, 3]], "inter_core_tiling": [[{"dim": dim, "split": 4}]]}
        for name, dim in (("Conv1", "D6"), ("Conv2", conv2_split))
    ]
    (tmp_path / "mapping.yaml").write_text(
        yaml.safe_dump({"layers": layers, "fused_groups": [{"name": "G", "layers": ["Conv1", "Conv2"]}]})
    )
    options = SolveOptions(artifacts=False)
    allocation = evaluate_mapping(hardware, workload, str(tmp_path / "out"), str(tmp_path / "mapping.yaml"), options)
    space = allocation.context.get("allocation").space
    (solved,) = (n for n in space.ssc_nodes if n.name == "Conv2")
    assert space.running_sum_bits(solved.outputs[0]) == bits


@pytest.mark.parametrize(("reduction", "bits"), [(None, 16), ("int32", 32)])
def test_the_tiler_holds_a_split_reduction_s_partial_sum_at_the_partial_sum_format(tmp_path, reduction, bits):
    """Y = X W, X (16,64), W (64,32), bf16, the contraction split over tpu_like_quad_core's cores 0 and 1, nothing
    tiled: each core holds X's 16 x 32 and W's 32 x 32, 8192 and 16384 bits, and its partial Y, 16 x 32 at the format
    the cores send each other it at, the bf16 type (8192 bits) or a declared int32 reduction (16384 bits), not at the
    int32 accumulator it sums in within the firing."""
    workload = onnx_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"], name="Gemm")], {"X": [16, 64], "W": [64, 32]}, ["Y"]
    )
    acc, w = _parse(tpu_quad(tmp_path, reduction), workload)
    (cn,) = w.get_computation_nodes()
    contraction = w.leading_dim(w.get_dims(cn)[2])[0]
    full = {d: w.get_dimension_size(d) for d in w.unique_dimensions()[0]}
    resident = full | {contraction: 32}
    cores = {cn: [acc.get_core(0), acc.get_core(1)]}
    held = CapacityTiler(w, acc).memory_bits((cn,), cores, {contraction: 2}, resident)
    assert held == dict.fromkeys(("tpu_like_0", "tpu_like_1"), 8192 + 16384 + 16 * 32 * bits)


def _conv(c: int, k: int, h: int, w: int) -> tuple[str, str]:
    """A bf16 3 x 3 convolution, ``c`` -> ``k`` channels on ``h`` x ``w``, on fusemax."""
    path = onnx_graph([conv("Conv", "x", "w", "y")], {"x": [1, c, h, w], "w": [k, c, 3, 3]}, ["y"])
    return "stream/inputs/examples/hardware/fusemax.yaml", path


def test_a_convolution_keeps_its_channels_whole_where_rows_and_columns_save_more():
    """A bf16 3 x 3 convolution, 56 -> 16 channels on 540 x 960, on fusemax's 256 x 256 array and its 2 MiB SRAM:
    cutting rows and columns saves the input and the output alike, more than cutting either channel dim, so the tiles
    cut only rows and columns and fit the SRAM's 16777216 bits."""
    acc, w = _parse(*_conv(56, 16, 540, 960))
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = gen.groups()
    cns = tuple(sub.get_computation_nodes())
    tiling = gen._build_intra_core_tiling(sub, cns)
    assert {entry["dim"] for entry in tiling} == {"Conv.D1", "Conv.D2"}
    assert _worst_core_ratio(gen, sub, cns, tiling) <= 1.0


def test_the_tiler_has_the_room_the_allocation_gives_tensors():
    """An AIE2 compute tile's 64 KiB data memory, less the 1 KiB stack its toolchain reserves: 516096 bits, for the
    tiler as for the allocation."""
    acc, w = _parse("stream/inputs/aie/hardware/single_core.yaml", _GEMM)
    memory = next(m for core in acc.core_list if core.type == "compute" for m in acc.memories_of(core))
    assert CapacityTiler(w, acc)._room(memory) == build_hardware_facts(acc, 0).usable_bits(memory) == 516096


def test_the_tiler_fills_the_share_of_the_room_it_is_given():
    """By default the tiles may fill all of fusemax's 16777216-bit SRAM; given half, 8388608 bits."""
    acc, w = _parse(*_conv(64, 64, 56, 56))
    (cn,) = w.get_computation_nodes()
    cores = {cn: [acc.get_core(0)]}
    assert CapacityTiler(w, acc)._capacities((cn,), cores) == {"sram": 16777216}
    assert CapacityTiler(w, acc, fill_fraction=0.5)._capacities((cn,), cores) == {"sram": 8388608}


def test_a_group_that_fits_whole_is_cut_where_its_tiles_overlap_compute_and_transfers():
    """A bf16 3 x 3 convolution, 64 -> 64 channels on 56 x 56, fits fusemax's SRAM whole: C = 28224 compute cycles,
    the weights' 1152 and the input's 6272 over the 512-bit link, C + X = 35648. Cut into 8 tiles of 7 rows, each tile
    reads the weights again under its compute: (28224 + 6272 + 8 x 1152) / 8 + 7 / 8 x 28224 = 30160. Held, the
    weights would load first and still take their share of each tile: 30304. Read again, they cross the off-chip port
    8 times next to the input and the output once: 8 x 589824 + 2 x 3211264."""
    acc, w = _parse(*_conv(64, 64, 56, 56))
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = gen.groups()
    cns = tuple(sub.get_computation_nodes())
    assert gen._build_intra_core_tiling(sub, cns) == [{"dim": "Conv.D1", "tile": 7}]
    tiler, cores = CapacityTiler(sub, acc), {cns[0]: [acc.get_core(0)]}
    whole = tiler._slices({})
    rows = sub.leading_dim(sub.get_dims(cns[0])[1])[0]
    cut = {d: n for d, n in whole.items() if n > 1}
    budgets = tiler._capacities(cns, cores)
    assert tiler.compute_cycles(cns, cores, {}, whole) == 28224
    assert tiler._pipelined_estimate(cns, cores, {}, whole, None, budgets, cut, [])[:3] == (35648, 28224, 7424)
    estimate = tiler._pipelined_estimate(cns, cores, {}, whole | {rows: 7}, None, budgets, cut, [rows])
    assert (estimate.cycles, estimate.offchip_bits) == (30160, 8 * 589824 + 2 * 3211264)


def test_a_group_that_fits_whole_stays_whole_where_its_tiles_read_the_weights_again():
    """A bf16 3 x 3 convolution, 256 -> 256 channels on 14 x 14, fits fusemax's SRAM whole: C = 1764 cycles, X =
    20000, 21764 whole. In 14 tiles of 2 rows by 7 columns each tile reads the weights and its input window over the
    SRAM's read port, 16380 cycles; the weights, held, come before the first tile and take their share of each:
    18432 + (16380 + 20000) / 14 + 13 / 14 x 20000 = 39602, so the convolution stays whole."""
    acc, w = _parse(*_conv(256, 256, 14, 14))
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = gen.groups()
    cns = tuple(sub.get_computation_nodes())
    assert gen._build_intra_core_tiling(sub, cns) == [{"dim": "Conv.D0", "tile": 1}]
    tiler, cores = CapacityTiler(sub, acc), {cns[0]: [acc.get_core(0)]}
    whole = tiler._slices({})
    rows, columns = (sub.leading_dim(d)[0] for d in sub.get_dims(cns[0])[1:3])
    cut = {d: n for d, n in whole.items() if n > 1}
    budgets = tiler._capacities(cns, cores)
    tiles = whole | {rows: 2, columns: 7}
    assert tiler._input_port_cycles(cns, cores, {}, tiles, None, 14) == 16380
    assert tiler._pipelined_estimate(cns, cores, {}, whole, None, budgets, cut, [])[:3] == (21764, 1764, 20000)
    assert tiler._pipelined_estimate(cns, cores, {}, tiles, None, budgets, cut, [rows, columns])[0] == 39602


def test_a_group_that_fits_whole_is_cut_for_overlap_only_along_dims_every_node_steps_and_no_core_splits():
    """Two 3 x 3 convolutions, 8 -> 16 -> 32 channels on 32 x 32, their rows split over tpu_like_quad_core's four cores:
    both have the batch, the columns, the rows and the 16 channels between them, and the rows are split over the
    cores. So only the batch, the columns and the 16 channels may be cut, and the group stays whole."""
    acc, w = _parse(_TPU_QUAD, _TWO_CONV)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = gen.groups()
    cns = tuple(sub.get_computation_nodes())
    batch, columns, rows, *_, channels = (sub.leading_dim(d)[0] for d in sub.get_dims(cns[0]))
    unroll = gen._inter_core_unrolling(sub, cns)
    assert {sub.leading_dim(d)[0]: f for d, f in unroll.items()} == {rows: 4}
    assert CapacityTiler(sub, acc)._overlap_dims(cns, unroll, None) == {batch, columns, channels}
    assert gen._build_intra_core_tiling(sub, cns) == [{"dim": "Conv1.D0", "tile": 1}]


def test_a_group_that_fits_whole_is_not_cut_for_overlap_where_its_tiles_fold_worse_onto_the_array():
    """Two 3 x 3 convolutions, 8 -> 16 -> 32 channels on 32 x 32, on eyeriss_like_quad_core's 14 x 12 arrays. Conv1's
    32 columns and 16 output channels take 3 passes of 14 and 2 of 12, filling 32/42 x 16/24 of the array; cut to 16
    columns, 2 passes of 14, 16/28 x 16/24: less, so the columns are not cut and the group stays whole."""
    acc, w = _parse("stream/inputs/examples/hardware/eyeriss_like_quad_core.yaml", _TWO_CONV)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = gen.groups()
    cns = tuple(sub.get_computation_nodes())
    cores = {cn: gen._select_cores_for_node(cn) for cn in cns}
    unroll = {sub.leading_dim(d)[0]: f for d, f in gen._inter_core_unrolling(sub, cns).items()}
    tiler = CapacityTiler(sub, acc)
    whole = tiler._slices(unroll)
    columns = sub.leading_dim(sub.get_dims(cns[0])[1])[0]
    fills = [
        [tiler._node_fill(cns[0], cores, unroll, tiles, None, folded=folded)[0] for folded in (False, True)]
        for tiles in (whole, whole | {columns: 16})
    ]
    assert fills == [
        [0.0, pytest.approx(math.log(32 / 42 * 16 / 24))],
        [0.0, pytest.approx(math.log(16 / 28 * 16 / 24))],
    ]
    assert gen._build_intra_core_tiling(sub, cns) == [{"dim": "Conv1.D0", "tile": 1}]


def _whole_group(hardware: str, workload: str):
    """The generator's tiler for the workload fused in one group, with its nodes, cores, split, whole per-core slices
    and budgets."""
    acc, w = _parse(hardware, workload)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = w.split_fusion_groups(cut_points=[])
    cns = tuple(sub.get_computation_nodes())
    tiler, cores, unroll, _, node_unroll = gen._tiler(sub, cns)
    unroll = {sub.leading_dim(d)[0]: f for d, f in unroll.items()}
    return tiler, cns, cores, unroll, node_unroll, tiler._slices(unroll), tiler._capacities(cns, cores)


def test_a_tensor_two_nodes_read_from_one_memory_moves_into_it_once():
    """Two bf16 3 x 3 convolutions read one 16 x 32 x 32 input, 262144 bits, from fusemax's one SRAM, and are added:
    the input moves into the SRAM once for both, as it leaves off chip once."""
    shapes = {"x": [1, 16, 32, 32], "w1": [16, 16, 3, 3], "w2": [16, 16, 3, 3]}
    nodes = [conv("Conv1", "x", "w1", "a"), conv("Conv2", "x", "w2", "b"), helper.make_node("Add", ["a", "b"], ["y"])]
    tiler, cns, cores, unroll, node_unroll, whole, budgets = _whole_group(
        "stream/inputs/examples/hardware/fusemax.yaml", onnx_graph(nodes, shapes, ["y"])
    )
    (x,) = (t for t in cns[0].inputs if t.name == "x")
    moved = tiler._moved(cns, cores, unroll, whole, node_unroll, [], budgets, spills=True, tensors={x})
    assert moved == {"sram": 262144, None: 262144}


def test_a_tensor_read_from_a_staging_memory_its_cores_compute_from_moves_only_their_slices():
    """TPU7x's VMEMs stage tensors for other TensorCores and hold their own cores' operands. A ReLU over a bf16
    1024 x 4096 input, its rows split over the four vector units, moves into each VMEM the quarter its unit reads,
    67108864 / 4 bits, and no whole copy besides."""
    x = helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [1024, 4096])
    y = helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [1024, 4096])
    path = f"{tempfile.mkdtemp()}/relu.onnx"
    onnx.save(helper.make_model(helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "relu", [x], [y])), path)
    tiler, cns, cores, unroll, node_unroll, whole, budgets = _whole_group(_TPU_V7, path)
    (x,) = cns[0].inputs
    moved = tiler._moved(cns, cores, unroll, whole, node_unroll, [], budgets, spills=True, tensors={x})
    assert moved == dict.fromkeys(["vmem_0", "vmem_1", "vmem_2", "vmem_3"], 67108864 / 4) | {None: 67108864}


def test_a_convolution_fused_with_a_strided_pool_fills_the_array_along_its_own_rows():
    """A 3 x 3 convolution, 8 -> 16 channels on 32 x 32, fused with a 3 x 3 max pool of stride 2: the group steps the
    pool's 16 x 16 outputs and their windows, but the convolution's tile spans its own 32 x 32, 2 x 15 + 2 rows and
    columns, laid on fusemax's 256 x 256 array: 32 / 256 x 32 / 256 of it, not the 16 / 256 x 16 / 256 of the pool's."""
    tiler, (conv1, _), cores, unroll, node_unroll, whole, _ = _whole_group(
        "stream/inputs/examples/hardware/fusemax.yaml", conv_chain(stride=2, pool=True)
    )
    log_fill, units = tiler._node_fill(conv1, cores, unroll, whole, node_unroll)
    assert (math.exp(log_fill), units) == (pytest.approx(32 / 256 * 32 / 256), 65536)


def test_a_produced_tensor_takes_one_buffer_the_size_of_its_largest_tile_or_window():
    """Two bf16 3 x 3 convolutions, 16 -> 16 -> 16 channels on 32 x 32, on fusemax's array core, columns cut to 8 and
    double-buffered: the input's 10-column window, 163840 bits, w1 and w2, 36864 each, and y2's 8 columns, 131072. The
    intermediate y1 is held once, at its consumer's 10-column window, 163840 bits, not next to its producer's 8 columns:
    532480 bits."""
    shapes = {"x": [1, 16, 32, 32], "w1": [16, 16, 3, 3], "w2": [16, 16, 3, 3]}
    path = onnx_graph([conv("Conv1", "x", "w1", "y1"), conv("Conv2", "y1", "w2", "y2")], shapes, ["y2"])
    acc, w = _parse("stream/inputs/examples/hardware/fusemax.yaml", path)
    conv1, conv2 = w.get_computation_nodes()
    columns = w.leading_dim(w.get_dims(conv2)[1])[0]
    held = CapacityTiler(w, acc).memory_bits(
        (conv1, conv2), dict.fromkeys((conv1, conv2), [acc.get_core(0)]), {}, {columns: 8}
    )
    assert held == {"sram": 532480}


def test_a_reader_holds_every_window_it_reads_of_a_tile_across_a_loop_its_producer_does_not_step(tmp_path):
    """The bf16 MLP on fusemax's array core, its 64 tokens cut to 16 inside its 128 outputs cut to 64: Gemm1 makes
    each 16 x 256 tile of H1 once, but Gemm2 reads all four token tiles again for its second half of the outputs, so
    it holds them across the outputs loop, 4 x 4096 elements, 262144 bits, the one buffer of H1. Next to it X's 16 x
    128 and Y's 16 x 64, double-buffered, 65536 and 32768 bits, W1 whole, 524288, and W2's 256 x 64 halves,
    double-buffered, 524288: 1409024 bits."""
    acc, w = _parse("stream/inputs/examples/hardware/fusemax.yaml", _mlp(tmp_path))
    gemm1, gemm2 = sorted(w.get_computation_nodes(), key=lambda n: n.name)
    tokens, outputs = (w.leading_dim(w.get_dims(gemm2)[i])[0] for i in (0, 2))
    cores = dict.fromkeys((gemm1, gemm2), [acc.get_core(0)])
    tiler = CapacityTiler(w, acc)
    held = tiler.memory_bits((gemm1, gemm2), cores, {}, {tokens: 16, outputs: 64}, order=[tokens, outputs])
    assert held == {"sram": 1409024}


def test_a_produced_tensor_two_windowed_readers_read_is_held_once_and_each_window_beyond_it(tmp_path):
    """On fusemax's shared SRAM, columns cut to 8 and double-buffered: a 1 x 1 convolution makes y, 8 columns of 16 x
    32, 131072 bits, which two 3 x 3 convolutions each read as a 10-column window, 163840 bits, 32768 beyond y's tile
    apiece, as the allocation charges each copy beyond its source. Next to them x's 8 columns and the two outputs, each
    131072, which the add on the vector core reads as they are, and z, 131072; w0, 4096, and wa and wb, 36864 each:
    798720 bits."""
    nodes = [
        helper.make_node("Conv", ["x", "w0"], ["y"], name="Conv0", kernel_shape=[1, 1]),
        helper.make_node("Conv", ["y", "wa"], ["a"], name="ConvA", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Conv", ["y", "wb"], ["b"], name="ConvB", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["a", "b"], ["z"], name="Add"),
    ]
    shapes = {"x": [1, 16, 32, 32], "w0": [16, 16, 1, 1], "wa": [16, 16, 3, 3], "wb": [16, 16, 3, 3]}
    info = {name: helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape) for name, shape in shapes.items()}
    out = helper.make_tensor_value_info("z", TensorProto.BFLOAT16, [1, 16, 32, 32])
    graph = helper.make_graph(nodes, "g", list(info.values()), [out])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "fan_out.onnx")
    acc, w = _parse("stream/inputs/examples/hardware/fusemax.yaml", str(tmp_path / "fan_out.onnx"))
    cns = {cn.name: cn for cn in w.get_computation_nodes()}
    columns = w.leading_dim(w.get_dims(cns["Conv0"])[1])[0]
    cores = {cn: [acc.get_core(1 if name == "Add" else 0)] for name, cn in cns.items()}
    assert CapacityTiler(w, acc).memory_bits(tuple(cns.values()), cores, {}, {columns: 8}) == {"sram": 798720}


def test_divisors_desc():
    assert _divisors_desc(12) == [12, 6, 4, 3, 2, 1]
    assert _divisors_desc(1) == [1]
    assert _divisors_desc(14336)[0] == 14336  # sqrt enumeration returns the whole dim first


def test_a_gemm_whose_weights_overflow_streams_the_cut_that_reads_them_once():
    """A bf16 256 x 8192 x 2048 Gemm, its 2048 outputs split over tpu_like_quad_core's four cores, overflows a core's
    16777216 bits whole. Streaming its 8192-long contraction in tiles of 512 fills a core exactly: double-buffered
    256 x 512 and 512 x 512 bf16 tiles next to 256 x 512 int32 running sums. It reads every operand once, 33554432 +
    268435456 + 8388608 bits over the 64-bit off-chip port: X = 4849664 next to C = 256 x 8192 x 2048 / (4 x 1024) =
    1048576, (C + X) / 16 + 15 / 16 X = 4915200 in its 16 tiles. Cutting the rows and outputs instead, 16 by 32, reads
    the weights again for every row tile, X = 12713984, so the tiler streams the contraction."""
    acc, w = _parse(_TPU_QUAD, _GEMM)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = gen.groups()
    cns = tuple(sub.get_computation_nodes())
    tiler, cores, unroll, protected, node_unroll = gen._tiler(sub, cns)
    seed = gen._whole_layer_tiling(sub, cns)
    streamed = gen._build_intra_core_tiling(sub, cns)
    assert _worst_core_ratio(gen, sub, cns, seed) > 1.0
    assert streamed == [{"dim": "Gemm.D1", "tile": 512}]
    assert _worst_core_ratio(gen, sub, cns, streamed) == 1.0
    assert tiler.estimate(cns, cores, unroll, protected, streamed, node_unroll)[:3] == (4915200, 1048576, 4849664)
    outputs = [{"dim": "Gemm.D0", "tile": 16}, {"dim": "Gemm.D2", "tile": 32}]
    assert tiler.estimate(cns, cores, unroll, protected, outputs, node_unroll).transfer == 12713984


def test_noop_when_group_fits():
    """A group whose footprint already fits is returned unchanged -- no over-tiling."""
    acc, w = _parse(_TPU_QUAD, _SWIGLU_FITS)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    for sub in w.split_fusion_groups(cut_points=gen._cut_points(None)):
        cns = tuple(sub.get_computation_nodes())
        if not cns:
            continue
        seed = gen._auto_fusion_tiling(sub, cns) or gen._whole_layer_tiling(sub, cns)
        if _worst_core_ratio(gen, sub, cns, seed) <= 1.0:
            assert gen._capacity_refine(sub, cns, seed) == seed


def test_footprint_is_summed_per_physical_core():
    """Nodes on one core sum their tiles, and a tensor both hold with the same tile counts once."""
    acc, w = _parse(_TPU_QUAD, _SWIGLU_FITS)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    sub = next(s for s in w.split_fusion_groups(cut_points=gen._cut_points(None)) if tuple(s.get_computation_nodes()))
    cns = tuple(sub.get_computation_nodes())
    a, b = next((a, b) for a in cns for b in cns if a is not b and set(a.outputs) & set(b.inputs))
    core = gen._select_cores_for_node(a)[0]
    tiler = CapacityTiler(sub, acc)

    def held(*nodes) -> float:
        return sum(tiler.memory_bits(nodes, dict.fromkeys(nodes, [core]), {}, {}).values())

    shared = sum(math.prod(t.shape) * t.operand_type.bitwidth for t in set(a.outputs) & set(b.inputs))
    assert held(a, b) == pytest.approx(held(a) + held(b) - shared)


def test_refined_mapping_validates():
    """The mapping the refined tiling produces still passes MappingValidator."""
    acc, w = _parse(_TPU_QUAD, _GEMM)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    paths, _ = gen.generate_all_groups()
    for path in paths:
        import yaml

        data = yaml.safe_load(open(path))
        assert MappingValidator(data).validate(), MappingValidator(data).errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


_PLAN = """
import sys, tempfile
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage
ctx = StageContext.from_kwargs(accelerator=sys.argv[1], workload_path=sys.argv[2], output_path=tempfile.mkdtemp())
ctx = MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()[0]
gen = GenericMappingGenerator(ctx.get("accelerator"), ctx.get("workload"), tempfile.mkdtemp())
(sub,) = ctx.get("workload").split_fusion_groups(cut_points=gen._cut_points(None))
print(gen._build_intra_core_tiling(sub, tuple(sub.get_computation_nodes())))
"""


def test_a_tiling_over_several_dims_does_not_depend_on_the_hash_seed(tmp_path):
    """The overflowing SwiGLU streams two dims, in the same order whatever order a set iterates in: its hidden dim
    innermost, as the down projection's running sums for all 512 tokens, 512 x 1024 at int32, fill a core alone."""
    workload = make_swiglu_workload(512, 1024, 4096, "bf16", "bf16", output_dir=str(tmp_path))
    plans = {
        subprocess.run(
            [sys.executable, "-c", _PLAN, _TPU_QUAD, workload],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()[-1]
        for seed in range(4)
    }
    assert plans == {str([{"dim": "Gemm_Left.D2", "tile": 32}, {"dim": "Gemm_Left.D0", "tile": 64}])}


def test_a_node_left_on_one_core_holds_its_tensors_whole():
    """Split over four cores, each Gemm core streams its 8192-long contraction 512 at a time next to all 512 of its
    output columns. Left on one core it holds the columns of all four, so the group's per-core tiles shrink to 64 of the
    contraction and 128 columns, 512 there."""
    acc, w = _parse(_TPU_QUAD, _GEMM)
    (cn,) = w.get_computation_nodes()
    unroll = {w.get_dims(cn)[2]: 4}
    cores = [c for c in acc.core_list if c.id in (0, 1, 2, 3)]
    tiler = CapacityTiler(w, acc)
    assert tiler.plan((cn,), {cn: cores}, unroll, set()) == [{"dim": "Gemm.D1", "tile": 512}]
    assert tiler.plan((cn,), {cn: cores[:1]}, unroll, set()) == [
        {"dim": "Gemm.D1", "tile": 64},
        {"dim": "Gemm.D2", "tile": 128},
    ]


def test_a_dim_split_whole_across_cores_counts_one_slice_per_core(tmp_path):
    """Decode over a 32k context, its 8 KV heads split one per MXU: each VMEM holds the K and V of its two MXUs' heads,
    32 of its 64 MiB, so the group fits as it is rather than as if every core held all 8 heads."""
    acc, parsed = _parse(
        "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml", make_decode_attention(tmp_path / "d.onnx", 1, 32768)
    )
    w = expand_normalizations(parsed)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = w.split_fusion_groups(cut_points=gen._cut_points(None))
    cns = tuple(sub.get_computation_nodes())
    seed = gen._auto_fusion_tiling(sub, cns) or gen._whole_layer_tiling(sub, cns)
    assert gen._build_intra_core_tiling(sub, cns) == seed


def test_a_group_the_given_tiling_does_not_name_is_tiled_to_fit():
    """Two bf16 3 x 3 convolutions, 16 channels on 256 x 256, on fusemax, cut after Conv1, a tiling given for Conv1's
    rows only. Conv1's group keeps it. Conv2 whole holds its input and output, 16777216 bits each, and 36864 bits of
    weights: 2.0022 times fusemax's 16777216-bit SRAM. Its group takes the automatic tiling, which fits."""
    shapes = {"input": [1, 16, 256, 256], "w1": [16, 16, 3, 3], "w2": [16, 16, 3, 3]}
    path = onnx_graph([conv("Conv1", "input", "w1", "mid"), conv("Conv2", "mid", "w2", "out")], shapes, ["out"])
    acc, w = _parse("stream/inputs/examples/hardware/fusemax.yaml", path)
    given = [{"dim": "Conv1.D1", "tile": 32}]
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp(), intra_core_tiling=given)
    first, second = gen.groups(["Conv1"])
    assert gen._build_intra_core_tiling(first, tuple(first.get_computation_nodes())) == given
    cns = tuple(second.get_computation_nodes())
    assert _worst_core_ratio(gen, second, cns, gen._whole_layer_tiling(second, cns)) == 33591296 / 16777216
    assert _worst_core_ratio(gen, second, cns, gen._build_intra_core_tiling(second, cns)) <= 1.0


@pytest.mark.slow
def test_an_empty_given_tiling_is_cut_as_without_one(tmp_path):
    """The attention of the test below, which no tiling fits fused: given an empty tiling, no entry names its group,
    and it is cut as without one."""
    acc, parsed = _parse(
        "stream/inputs/examples/hardware/eyeriss_like_quad_core.yaml",
        make_multi_head_attention(tmp_path / "mha.onnx", seq=512),
    )
    gen = GenericMappingGenerator(acc, expand_normalizations(parsed), tempfile.mkdtemp(), intra_core_tiling=[])
    operators = [{"proj_K"}, {"proj_Q"}, {"proj_V"}, {"scores", "softmax", "context"}]
    assert [{source_operator(n) for n in g.get_computation_nodes()} for g in gen.groups()] == operators


def test_a_group_no_tiling_fits_is_cut_where_the_operators_before_stop_fitting(tmp_path):
    """A 4096-wide attention fused on four Eyeriss-like cores cannot hold its projections' weights next to the softmax
    of every head: it is cut after the longest runs that fit, two projections, the third, then the attention with
    all the sub-ops its softmax expands into. The two projections fused take longer, as the tiler estimates them,
    than each apart, so they are cut apart too."""
    acc, parsed = _parse(
        "stream/inputs/examples/hardware/eyeriss_like_quad_core.yaml",
        make_multi_head_attention(tmp_path / "mha.onnx", seq=512),
    )
    w = expand_normalizations(parsed)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (fused,) = w.split_fusion_groups(cut_points=gen._cut_points(None))
    assert gen._unfit_nodes(fused) == ["proj_Q", "proj_V"]
    _, groups = gen.generate_all_groups()
    operators = [{source_operator(n) for n in g.get_computation_nodes()} for g in groups]
    assert operators == [{"proj_K"}, {"proj_Q"}, {"proj_V"}, {"scores", "softmax", "context"}]
    assert all(gen._fits(group) for group in groups)


@pytest.mark.parametrize(
    ("hardware", "workload", "node", "layer_by_layer", "innermost"),
    [
        # The key projection alone: the 8 x 128 x 32 int24 running sums of its two 16-wide head-dim tiles, 96 KiB,
        # stay in a 256 KiB chiplet over its 4096-wide contraction
        ("simba_small", "mha", "proj_K", True, "proj_K.D2"),
        # The fused 64-token SwiGLU's down projection keeps its outputs while the d_ff its weights stream along goes
        # past them
        ("eyeriss_like_quad_core", "swiglu", "Gemm_Down", False, "Gemm_Left.D0"),
        # With 256 tokens, the outputs of every token tile held across the d_ff take 9109504 bits with the group's
        # other tiles, over a core's 8388608, so the d_ff is cut first and loops innermost
        ("eyeriss_like_quad_core", _SWIGLU_FITS, "Gemm_Down", False, "Gemm_Left.D2"),
    ],
)
def test_a_reduction_runs_innermost_only_where_its_outputs_cannot_stay_while_it_accumulates(
    tmp_path, hardware, workload, node, layer_by_layer, innermost
):
    makers = {
        "mha": lambda: make_multi_head_attention(tmp_path / "mha.onnx", seq=512),
        "swiglu": lambda: make_swiglu_workload(64, 512, 2048, "bf16", "bf16", output_dir=str(tmp_path)),
    }
    path = makers[workload]() if workload in makers else workload
    acc, w = _parse(f"stream/inputs/examples/hardware/{hardware}.yaml", path)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    cuts = [n.name for n in w.get_computation_nodes()] if layer_by_layer else gen._cut_points(None)
    group = next(
        g for g in w.split_fusion_groups(cut_points=cuts) if node in {n.name for n in g.get_computation_nodes()}
    )
    assert gen._build_intra_core_tiling(group, tuple(group.get_computation_nodes()))[0]["dim"] == innermost


_PACKED_CORE = """
name: packed
type: zigzag.compute
memories:
  wreg:
    {size: 8, r_cost: 0.1, w_cost: 0.1, area: 0, latency: 1, auto_cost_extraction: False, operands: [I2],
     served_dimensions: [],
     ports: [{name: r_port_1, type: read, bandwidth_min: 8, bandwidth_max: 8, allocation: ["I2, tl"]},
     {name: w_port_1, type: write, bandwidth_min: 8, bandwidth_max: 8, allocation: ["I2, fh"]}]}
  result_buffer:
    {size: 4096, r_cost: 0.1, w_cost: 0.1, area: 0, latency: 1, operands: [O], served_dimensions: [D2, D3],
     ports: [{name: r_port_1, type: read, bandwidth_min: 32, bandwidth_max: 32, allocation: ["O, tl", "O, th"]},
     {name: w_port_1, type: write, bandwidth_min: 32, bandwidth_max: 32, allocation: ["O, fh", "O, fl"]}]}
  l1:
    {size: L1_BITS, r_cost: 10, w_cost: 10, area: 0, latency: 1, operands: [I1, I2, O], served_dimensions: [D1, D2, D3],
     ports: [{name: r_port_1, type: read, bandwidth_min: 64, bandwidth_max: 4096,
              allocation: ["I1, tl", "I2, tl", "O, tl", "O, th"]},
             {name: w_port_1, type: write, bandwidth_min: 64, bandwidth_max: 4096,
              allocation: ["I1, fh", "I2, fh", "O, fh", "O, fl"]}]}
operational_array:
  {unit_energy: 0.02, unit_area: 1, dimensions: [D1, D2, D3], sizes: [16, 16, 4], systolic_dimensions: [D1, D2]}
operand_precision: {input: int8}
operator_types: [MatMul, Gemm]
"""

_PACKED_CHIP = """
name: chip
cores: {0: packed.yaml}
memories:
  l1:
    offchip: false
    views: [0.l1]
    ports: {r_port_1: {from: r_port_1, cores: [0]}, w_port_1: {from: w_port_1, cores: [0]}}
  dram:
    offchip: true
    views: []
    capacity_bits: 10000000000
    ports: {rw_port_1: {bandwidth: 256, serves: {any: [read, write]}, read_energy: 1, write_energy: 1}}
connections: [{type: link, bandwidth: 256, ends: {l1: [r_port_1, w_port_1], dram: [rw_port_1]}}]
"""


_SRAM_CORE = """
name: sram
type: zigzag.memory
memories:
  sram:
    {size: SRAM_BITS, r_cost: 1, w_cost: 1, area: 0, latency: 1, operands: [I1, I2, O], served_dimensions: [D1, D2],
     ports: [{name: r_port_1, type: read, bandwidth_min: 64, bandwidth_max: 512,
              allocation: ["I1, tl", "I2, tl", "O, tl", "O, th"]},
             {name: w_port_1, type: write, bandwidth_min: 64, bandwidth_max: 512,
              allocation: ["I1, fh", "I2, fh", "O, fh", "O, fl"]}]}
operational_array: {unit_energy: 0, unit_area: 0, dimensions: [D1, D2], sizes: [0, 0]}
"""

_STAGED_CHIP = """
name: chip
cores: {0: packed.yaml, 1: sram.yaml}
core_coordinates: {0: [0, 0], 1: [1, 0]}
memories:
  l1:
    offchip: false
    views: [0.l1]
    ports: {r_port_1: {from: r_port_1, cores: [0]}, w_port_1: {from: w_port_1, cores: [0]}}
  sram:
    offchip: false
    views: [1.sram]
    ports: {r_port_1: {from: r_port_1, cores: [1]}, w_port_1: {from: w_port_1, cores: [1]}}
  dram:
    offchip: true
    views: []
    capacity_bits: 10000000000
    ports: {rw_port_1: {bandwidth: 256, serves: {any: [read, write]}, read_energy: 1, write_energy: 1}}
connections:
  - {type: link, bandwidth: 256, ends: {l1: [r_port_1, w_port_1], sram: [r_port_1, w_port_1]}}
  - {type: link, bandwidth: 256, ends: {sram: [r_port_1, w_port_1], dram: [rw_port_1]}}
"""


def _gemm(tmp_path, m: int, k: int, n: int, l1_kib: int, sram_kib: int = 0, link: int = 256) -> tuple[str, str]:
    """The chip and an int8 ``m x k`` by ``k x n`` Gemm on it: one packed core, partial sums in int8 too, with an L1 of
    ``l1_kib``, reaching the off-chip memory through the SRAM of ``sram_kib`` of a memory core, over a link of ``link``
    bits a cycle, or directly where there is none."""
    (tmp_path / "packed.yaml").write_text(_PACKED_CORE.replace("L1_BITS", str(l1_kib * 8192)))
    (tmp_path / "sram.yaml").write_text(_SRAM_CORE.replace("SRAM_BITS", str(sram_kib * 8192)))
    staged = _STAGED_CHIP.replace("bandwidth: 256, ends: {l1", f"bandwidth: {link}, ends: {{l1")
    (tmp_path / "chip.yaml").write_text(staged if sram_kib else _PACKED_CHIP)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["a", "w"], ["out"], name="Gemm")],
        "g",
        [
            helper.make_tensor_value_info("a", TensorProto.INT8, [m, k]),
            helper.make_tensor_value_info("w", TensorProto.INT8, [k, n]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.INT8, [m, n])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "gemm.onnx")
    return str(tmp_path / "chip.yaml"), str(tmp_path / "gemm.onnx")


def _order(tmp_path, m, k, n, l1_kib, sram_kib, nb_cols_to_use=None, link=256) -> list[str]:
    acc, w = _parse(*_gemm(tmp_path, m, k, n, l1_kib=l1_kib, sram_kib=sram_kib, link=link))
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp(), nb_cols_to_use=nb_cols_to_use)
    (sub,) = gen.groups()
    return [entry["dim"] for entry in gen._build_intra_core_tiling(sub, tuple(sub.get_computation_nodes()))]


@pytest.mark.parametrize(
    ("shape", "nb_cols_to_use", "order"),
    [
        # a's 256 KiB exceeds the 96 KiB SRAM, so rows outermost read it once instead of once per column tile.
        ((256, 1024, 4096, 30, 96), None, ["Gemm.D1", "Gemm.D2", "Gemm.D0"]),
        # Rows innermost keep the 128 x 128 partial sums in the L1, so the 2 MiB of weights are read once.
        ((128, 2048, 1024, 56, 0), None, ["Gemm.D0", "Gemm.D1", "Gemm.D2"]),
        # On one column the SRAM's memory core is not the allocation's, so the default order stays.
        ((256, 1024, 4096, 30, 96), 1, ["Gemm.D1", "Gemm.D0", "Gemm.D2"]),
        # The tiles passing through the 28 KiB SRAM fill it, so it holds no block of rows: the default order stays.
        ((128, 512, 8192, 56, 28), None, ["Gemm.D0", "Gemm.D1", "Gemm.D2"]),
        # Held in the SRAM, a would reach the L1 once per column tile over a link no faster than off chip.
        ((64, 512, 1024, 14, 28), 2, ["Gemm.D0", "Gemm.D1", "Gemm.D2"]),
    ],
    ids=["rows-outermost", "rows-innermost", "one-column", "staging-full", "l1-takes-more"],
)
def test_a_tile_loop_order_is_the_one_moving_the_least_data(tmp_path, shape, nb_cols_to_use, order):
    assert _order(tmp_path, *shape, nb_cols_to_use) == order


def test_a_reduction_looped_innermost_is_cut_before_the_tiles_outside_it(tmp_path):
    """a[256,64] @ w1[64,1024] -> mid @ w2[1024,64] -> out[256,64], int8 accumulating in int32, in a 64 KiB L1. The
    first fit, 32 tokens by 128 hidden, cannot hold the running sums of every token tile, 256 x 64 x 32 bits, next to
    them, so the hidden dim loops innermost; refitted with the hidden dim cut first, 64 tokens by 64 hidden fit and
    the weights stream 4 rather than 8 times."""
    chip, _ = _gemm(tmp_path, 256, 64, 64, l1_kib=64)
    core = tmp_path / "packed.yaml"
    core.write_text(core.read_text().replace("{input: int8}", "{input: int8, accumulator: int32}"))
    shapes = {"a": [256, 64], "w1": [64, 1024], "w2": [1024, 64], "out": [256, 64]}
    info = {name: helper.make_tensor_value_info(name, TensorProto.INT8, shape) for name, shape in shapes.items()}
    nodes = [
        helper.make_node("Gemm", ["a", "w1"], ["mid"], name="Gemm1"),
        helper.make_node("Gemm", ["mid", "w2"], ["out"], name="Gemm2"),
    ]
    graph = helper.make_graph(nodes, "mlp", [info["a"], info["w1"], info["w2"]], [info["out"]])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "mlp.onnx")
    acc, w = _parse(chip, str(tmp_path / "mlp.onnx"))
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = gen.groups()
    assert gen._build_intra_core_tiling(sub, tuple(sub.get_computation_nodes())) == [
        {"dim": "Gemm1.D2", "tile": 64},
        {"dim": "Gemm1.D0", "tile": 64},
    ]


def test_a_wide_link_from_the_staging_memory_takes_the_rows_outermost(tmp_path):
    """The last case above with the SRAM's link to the L1 16 times as wide: the L1 takes a again per column tile in
    less time than a is read from off chip again, so the rows go outermost."""
    assert _order(tmp_path, 64, 512, 1024, 14, 28, nb_cols_to_use=2, link=4096) == ["Gemm.D1", "Gemm.D2", "Gemm.D0"]


def test_a_memory_behind_an_on_chip_buffer_takes_no_more_than_before(tmp_path):
    """The last case above with a buffer no core uses between the SRAM and the L1, over a link of 64 bits a cycle: the
    L1 has no link from off chip or a staging memory to time it by, so it may take no more than in the default order,
    which stays."""
    chip, gemm = _gemm(tmp_path, 64, 512, 1024, l1_kib=14, sram_kib=28)
    port = "{bandwidth: 4096, serves: {any: [read, write]}, read_energy: 1, write_energy: 1}"
    buffer = f"  buf:\n    offchip: false\n    views: []\n    capacity_bits: 8\n    ports: {{rw_port_1: {port}}}\n"
    text = (
        open(chip)
        .read()
        .replace("  dram:\n", buffer + "  dram:\n", 1)
        .replace(
            "  - {type: link, bandwidth: 256, ends: {l1: [r_port_1, w_port_1], sram: [r_port_1, w_port_1]}}\n",
            "  - {type: link, bandwidth: 64, ends: {l1: [r_port_1, w_port_1], buf: [rw_port_1]}}\n"
            "  - {type: link, bandwidth: 4096, ends: {buf: [rw_port_1], sram: [r_port_1, w_port_1]}}\n",
        )
    )
    with open(chip, "w") as f:
        f.write(text)
    acc, w = _parse(chip, gemm)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp(), nb_cols_to_use=2)
    (sub,) = gen.groups()
    order = [entry["dim"] for entry in gen._build_intra_core_tiling(sub, tuple(sub.get_computation_nodes()))]
    assert order == ["Gemm.D0", "Gemm.D1", "Gemm.D2"]


def test_more_tiled_dims_than_are_ordered_keep_the_default_order(tmp_path, monkeypatch):
    """The first case above with fewer loops ordered than it tiles: the default order, from the dim indexing the least
    data to the most."""
    monkeypatch.setattr(CapacityTiler, "MAX_ORDERED_LOOPS", 2)
    assert _order(tmp_path, 256, 1024, 4096, 30, 96) == ["Gemm.D1", "Gemm.D0", "Gemm.D2"]


def test_a_reduction_feeding_the_next_node_stays_innermost(tmp_path):
    """Two matmuls fused, the first's 2048-long contraction tiled: the second reads only finished rows of its output,
    so that contraction stays inside every other loop, the second's own contraction too."""
    (tmp_path / "packed.yaml").write_text(_PACKED_CORE.replace("L1_BITS", str(16 * 8192)))
    (tmp_path / "chip.yaml").write_text(_PACKED_CHIP)
    nodes = [
        helper.make_node("Gemm", ["a", "w1"], ["h"], name="G1"),
        helper.make_node("Gemm", ["h", "w2"], ["out"], name="G2"),
    ]
    shapes = {"a": [64, 2048], "w1": [2048, 256], "w2": [256, 512]}
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(name, TensorProto.INT8, shape) for name, shape in shapes.items()],
        [helper.make_tensor_value_info("out", TensorProto.INT8, [64, 512])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), tmp_path / "chain.onnx")
    acc, w = _parse(str(tmp_path / "chip.yaml"), str(tmp_path / "chain.onnx"))
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (sub,) = gen.groups(cut_points=[])
    tiling = gen._build_intra_core_tiling(sub, tuple(sub.get_computation_nodes()))
    assert [entry["dim"] for entry in tiling] == ["G1.D1", "G1.D2", "G2.D2", "G1.D0"]


_GIVEN = [{"dim": "Gemm.D1", "tile": 64}, {"dim": "Gemm.D0", "tile": 32}, {"dim": "Gemm.D2", "tile": 128}]


@pytest.mark.parametrize("given_by", ["stage_option", "mapping"])
def test_a_given_intra_core_tiling_keeps_its_loop_order(tmp_path, given_by):
    """A lock on the order of a given tiling: the rows looped inside the columns, worse than outermost on this chip
    (the first case above), as given, so the steady state walks the contraction's 16 tiles innermost, then the rows'
    8, then the columns' 32."""
    chip, gemm = _gemm(tmp_path, 256, 1024, 4096, l1_kib=32, sram_kib=96)
    mapping = None
    if given_by == "mapping":
        mapping = str(tmp_path / "mapping.yaml")
        group = {"name": "g", "layers": ["Gemm"], "intra_core_tiling": _GIVEN}
        layer = {"name": "Gemm", "core_allocation": [[0]]}
        with open(mapping, "w") as f:
            yaml.safe_dump({"layers": [layer], "fused_groups": [group]}, f)
    stage_options = {"intra_core_tiling": _GIVEN} if given_by == "stage_option" else {}
    options = SolveOptions(nb_cols_to_use=2, artifacts=False, stage_options=stage_options)
    ctx = evaluate_mapping(chip, gemm, str(tmp_path / "out"), mapping, options).context
    (reuse, *_) = ctx.get("allocation").solution.performance["tensor_reuse"]
    temporal = [int(loop.split(",")[1]) for loop in reuse["loop_nest_out_to_in"] if loop.startswith("T(")]
    assert temporal[::-1] == [16, 8, 32]


def test_a_weight_placed_on_chip_takes_its_room_from_the_tiles(tmp_path):
    """A 256 KiB weight placed in the 288 KiB L1 for the whole run leaves 32 KiB: the tiles are cut to fit beside it,
    where sized for the whole L1 they overflowed what is left."""
    chip, gemm = _gemm(tmp_path, 16, 4096, 64, l1_kib=288)
    placement = tmp_path / "placement.yaml"
    placement.write_text(yaml.safe_dump({"runtime_args": {"w": {"placement": {"memories": ["l1"]}}}}))
    options = SolveOptions(nb_cols_to_use=1, artifacts=False)
    estimate = evaluate_mapping(chip, gemm, str(tmp_path / "out"), str(placement), options)
    assert estimate.cycles > 0


def test_a_weight_placed_in_the_l1_is_read_there_in_place(tmp_path):
    """With the 1 MiB weight placed in the 1056 KiB L1 for the whole run, holding it moves nothing: the rows go
    outermost so a is read once, not once per block of columns."""
    acc, w = _parse(*_gemm(tmp_path, 512, 512, 2048, l1_kib=1056))
    runtime_args = {"w": {"placement": {"memories": ["l1"]}}}
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp(), runtime_args=runtime_args, nb_cols_to_use=1)
    (sub,) = gen.groups()
    order = [entry["dim"] for entry in gen._build_intra_core_tiling(sub, tuple(sub.get_computation_nodes()))]
    assert order == ["Gemm.D2", "Gemm.D0"]


def test_a_weight_placed_in_part_in_the_l1_takes_and_moves_the_rest_there(tmp_path):
    """Half of the 64 KiB weight placed in the L1 saves its tile there half of what all of it placed saves, held or
    moved."""
    acc, w = _parse(*_gemm(tmp_path, 64, 256, 256, l1_kib=1056))
    (cn,) = w.get_computation_nodes()
    cores = {cn: [acc.get_core(0)]}
    resident = {w.leading_dim(d)[0]: w.get_dimension_size(d) for d in w.get_dims(cn)}

    def saved(share: float) -> tuple[float, float]:
        tilers = (CapacityTiler(w, acc), CapacityTiler(w, acc, on_chip={"l1": {"w": share * 256 * 256 * 8}}))
        held = [tiler.memory_bits((cn,), cores, {}, resident)["l1"] for tiler in tilers]
        moved = [tiler._moved((cn,), cores, {}, resident, None, [], {"l1": 2**30})["l1"] for tiler in tilers]
        return held[0] - held[1], moved[0] - moved[1]

    assert saved(0.5) == (saved(1)[0] / 2, saved(1)[1] / 2)


def test_each_output_of_a_node_with_several_is_placed_by_the_bridge_carrying_it(tmp_path, monkeypatch):
    """A node writing out and a: each reaches the next group over a bridge named after it, placed apart."""
    acc, w = _parse(*_gemm(tmp_path, 64, 256, 256, l1_kib=1056))
    (cn,) = w.get_computation_nodes()
    a, _, out = cn.tensors
    monkeypatch.setattr(w, "get_computation_nodes", lambda: [SimpleNamespace(name="Gemm", outputs=(out, a))])
    runtime_args = {f"{t.name}_bridge_out": {"placement": {"memories": ["l1"]}} for t in (out, a)}
    gen = GenericMappingGenerator(acc, w, str(tmp_path / "gen"), runtime_args=runtime_args)
    assert gen._on_chip == {"l1": {"out": whole_bits(out), "a": whole_bits(a)}}


def test_a_plan_replaces_another_faster_by_more_than_the_resolution_or_no_slower_moving_less_off_chip():
    """At a resolution of 15% of the slower time: 100 cycles replace 120, 17% faster, whatever they move off chip; 90
    replace 100, 10% faster, only moving fewer bits off chip, as do 100; 110, 9% slower, never replace 100, however few
    bits they move, nor do 120."""

    def estimate(cycles: float, offchip_bits: float) -> TilingEstimate:
        return TilingEstimate(cycles, 0.0, 0.0, offchip_bits)

    assert CapacityTiler.ESTIMATE_TOLERANCE == 0.15
    assert CapacityTiler.preferred(estimate(100, 9), estimate(120, 1))
    assert not CapacityTiler.preferred(estimate(90, 1), estimate(100, 1))
    assert CapacityTiler.preferred(estimate(90, 0), estimate(100, 1))
    assert CapacityTiler.preferred(estimate(100, 0), estimate(100, 1))
    assert not CapacityTiler.preferred(estimate(110, 0), estimate(100, 1))
    assert not CapacityTiler.preferred(estimate(120, 0), estimate(100, 1))


_EYERISS_ONE = "stream/inputs/examples/hardware/eyeriss_like_single_core.yaml"


def test_two_convolutions_on_one_eyeriss_core_stay_fused_where_apart_they_move_more_off_chip_for_little_time():
    """bf16 Conv1 8 -> 16 and Conv2 16 -> 32 channels, 3 x 3 on 32 x 32, on eyeriss_like_single_core: each in a group of
    its own, they are estimated faster in all than fused, but within the resolution, and they move more off chip:
    Conv2 alone reads its weights again for each of its 8 column tiles, and conv1_out leaves the chip and comes back.
    So they stay fused."""
    acc, w = _parse(_EYERISS_ONE, _TWO_CONV)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    (fused,) = w.split_fusion_groups(cut_points=gen._cut_points(None))
    together = gen._estimate(fused)
    apart = [gen._estimate(group) for group in w.split_fusion_groups(cut_points=["Conv1"])]
    apart = TilingEstimate(*map(sum, zip(*apart, strict=True)))
    assert together.cycles * (1 - CapacityTiler.ESTIMATE_TOLERANCE) < apart.cycles < together.cycles
    assert apart.offchip_bits > together.offchip_bits
    assert [[node.name for node in group.get_computation_nodes()] for group in gen.groups()] == [["Conv1", "Conv2"]]


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="the estimate leaves out the cost model's loading and draining per call")
def test_the_strided_pair_is_tiled_no_slower_than_as_one_tile():
    """A bf16 3 x 3 convolution, 4 -> 8 channels on 11 x 15, and a second of stride 2 onto 6 x 8 fit
    eyeriss_like_single_core's buffer as one tile; the generator cuts them into the 8 output columns, estimated
    faster. Solved, the cut should be no slower than the one tile."""
    nodes = [conv("Conv1", "input", "w1", "t1"), conv("Conv2", "t1", "w2", "output", stride=2)]
    workload = onnx_graph(nodes, {"input": [1, 4, 11, 15], "w1": [8, 4, 3, 3], "w2": [8, 8, 3, 3]}, ["output"])

    def solved(tiling: list[dict] | None) -> float:
        options = SolveOptions(artifacts=False, stage_options={"intra_core_tiling": tiling} if tiling else {})
        return evaluate_mapping(_EYERISS_ONE, workload, tempfile.mkdtemp(), options=options).cycles

    assert solved(None) <= solved([{"dim": "Conv1.D0", "tile": 1}])
