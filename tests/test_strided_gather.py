"""A 1x1 stride-2 convolution of a (1, 8, 32, 32) bf16 input holds and moves every other row and column only."""

from fractions import Fraction
from math import prod

import pytest
from conv_windows import HARDWARE, conv, onnx_graph
from onnx import helper

from stream.api import SolveOptions, evaluate_mapping
from stream.cost_model.layout import Layout, shared_run_bytes
from stream.frontends import load_workload
from stream.mapping.mapping import EdgePlacement
from stream.stages.parsing.accelerator_parser import parse_accelerator

SHAPES = {"input": [1, 8, 32, 32], "wd": [16, 8, 1, 1]}
BF16 = 16


def _down(k: int = 1, shape: tuple[int, ...] = (1, 8, 32, 32), **window) -> str:
    """input of ``shape`` -> Down (16, ``k`` x ``k``, unpadded unless ``window`` says) as an ONNX path."""
    node = helper.make_node("Conv", ["input", "wd"], ["d"], name="Down", kernel_shape=[k, k], **window)
    return onnx_graph([node], {"input": list(shape), "wd": [16, shape[1], k, k]}, ["d"])


def _chain() -> str:
    """input (1, 8, 32, 32) -> A (16, 3x3, stride 1, padded) -> a -> B (16, 1x1, stride 2) as an ONNX path."""
    nodes = [conv("A", "input", "w", "a"), conv("B", "a", "wb", "b", k=1, stride=2, pads=0)]
    return onnx_graph(nodes, {"input": [1, 8, 32, 32], "w": [16, 8, 3, 3], "wb": [16, 16, 1, 1]}, ["b"])


def _mapping(layers: dict[str, str], tiled: str) -> str:
    """One fused group of ``layers`` (name: the rest of its YAML entry), ``tiled``'s output columns tiled by 4."""
    entries = "".join(f"- {{name: {name}, {rest}}}\n" for name, rest in layers.items())
    group = f"[{', '.join(layers)}]"
    tiling = f"intra_core_tiling: [{{dim: {tiled}.D1, tile: 4}}]"
    return f"layers:\n{entries}fused_groups:\n- {{name: g, layers: {group}, {tiling}}}\n"


def _solve(path, workload: str, mapping: str, hardware: str = "tpu_like_quad_core"):
    """The allocation and the IR of ``workload`` solved on ``hardware`` under ``mapping``."""
    (path / "mapping.yaml").write_text(mapping)
    estimate = evaluate_mapping(
        HARDWARE.format(hardware), workload, str(path), str(path / "mapping.yaml"), SolveOptions(artifacts=False)
    )
    return estimate.context.get("allocation"), estimate.context.data["group_allocations"][0]


def _transfer(allocation, name: str):
    """The transfer called ``name`` and its chosen route."""
    tr = next(t for t in allocation.problem.workload.get_transfer_nodes() if t.name == name)
    return tr, allocation.mapping.get(tr).resource_allocation[0]


def _layout(ir: dict, name: str) -> dict:
    """The layout row of the transfer called ``name``."""
    return next(r for r in ir["performance"]["layouts"] if r["transfer"] == name)


def _view(tensor) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """The shape, offsets and strides of ``tensor``'s view of its buffer."""
    subview = tensor.subview
    return tuple(tensor.shape), subview.static_offsets.get_values(), subview.static_strides.get_values()


def _columns(workload, tile: int) -> dict:
    """Every unique dim at its extent, Down's output columns at ``tile``."""
    down = workload.get_computation_nodes()[0]
    sizes = {z: workload.get_dimension_size(z) for z in workload.unique_dimensions()[0]}
    sizes[workload.get_dims(down)[1]] = tile
    return sizes


def test_a_strided_reader_holds_only_the_indices_it_reads():
    """Four output columns read input columns 0, 2, 4, 6 of the 16 even rows: a (1, 8, 16, 4) tile, not (1, 8, 31, 7)"""
    workload = load_workload(_down(strides=[2, 2]))
    down, tensor = workload.get_computation_nodes()[0], workload.get_computation_nodes()[0].inputs[0]
    assert workload.index_steps(tensor) == ((1, 0), (1, 0), (2, 0), (2, 0))
    assert workload.get_tensor_shape_with_dimension_sizes(tensor, _columns(workload, 4), down) == (1, 8, 16, 4)


def test_a_strided_and_a_dense_reader_of_one_input_read_every_index():
    """A stride-2 3x3 conv next to the 1x1 one reads ``2*o + f - 1``, every column: the input is held whole."""
    nodes = [
        conv("Conv1", "input", "w1", "a", stride=2),
        conv("Down", "input", "wd", "d", k=1, stride=2, pads=0),
    ]
    workload = load_workload(onnx_graph(nodes, SHAPES | {"w1": [16, 8, 3, 3]}, ["a", "d"]))
    assert workload.index_steps(workload.get_computation_nodes()[0].inputs[0]) == ((1, 0),) * 4


def test_a_tile_is_held_on_the_index_steps_its_own_extents_read():
    """A 3x3 conv of stride 2 and dilation 3 reads ``2*o + 3*f``: every input column over all 13 output columns, but
    only columns 0, 3 and 6 in a tile of one output column."""
    workload = load_workload(_down(3, strides=[2, 2], dilations=[3, 3]))
    assert workload.index_steps(workload.get_computation_nodes()[0].inputs[0]) == ((1, 0),) * 4
    tiled = workload.with_modified_dimension_sizes(_columns(workload, 1))
    assert _view(tiled.get_computation_nodes()[0].inputs[0]) == ((1, 8, 31, 3), (0, 0, 0, 0), (1, 1, 1, 3))


def test_two_orders_share_no_run_along_an_axis_one_side_skips():
    """A (4, 8, 16) block into a copy with the outer two axes swapped shares runs of the last axis, 32 bytes; of one
    element, 2 bytes, where the source reads every other index along it."""
    block, full, outer_swapped = (4, 8, 16), (4, 8, 32), Layout((1, 0, 2))
    extents = (full, block)
    assert shared_run_bytes(block, full, BF16, Layout.row_major(3), outer_swapped, extents) == 32
    steps = ((1, 1, 2), (1, 1, 1))
    assert shared_run_bytes(block, full, BF16, Layout.row_major(3), outer_swapped, extents, steps) == 2


def test_a_walked_split_placement_serves_the_columns_a_strided_read_reads():
    """The input split by columns over two memories holds 8 of the 16 even columns in each. A window of six of them
    stepping four takes 6, 4, 0, 0 from the first and 0, 2, 6, 4 from the second: shares 10/22 and 12/22."""
    workload = load_workload(_down(strides=[2, 2]))
    tiled = workload.with_modified_dimension_sizes(_columns(workload, 4))
    tensor = tiled.get_computation_nodes()[0].inputs[0]
    memories = list(parse_accelerator(HARDWARE.format("simba_small")).memories.values())[:2]
    placement = EdgePlacement(tuple(memories), split=3)
    window = [(0, 0), (0, 7), (0, 15), (0, 5)]
    shares = tiled.placement_shares(placement, tensor, [window])
    assert shares is not None
    assert [Fraction(share).limit_denominator(100) for share in shares.values()] == [Fraction(5, 11), Fraction(6, 11)]


def test_a_strided_copy_moves_only_the_indices_it_reads(tmp_path):
    """Each firing moves 8 channels x 16 rows x 4 columns = 512 elements, read off chip at twice the row and column
    strides, 64 and 2, and written in one run; the transfer's loops span the 31 rows and 7 columns read from."""
    mapping = _mapping({"Down": "core_allocation: [[0]]"}, "Down")
    allocation, ir = _solve(tmp_path, _down(strides=[2, 2]), mapping, "simba_small")
    workload, (tr, route) = allocation.problem.workload, _transfer(allocation, "Transfer(input)")
    assert allocation.space.moved_bits(tr, route) == prod((8, 16, 4)) * BF16
    assert [workload.get_dimension_size(z) for z in workload.get_dims(tr)] == [1, 8, 31, 7]
    row = _layout(ir, "Transfer(input)")
    read = {stride: size for size, stride in row["read"]["access"]}
    assert read[64] == 16 and read[2] == 4
    assert row["read"]["offset"] == 0
    assert row["read_run_bytes"] == 2
    assert row["write"]["access"] == [[512, 1]]


def test_a_strided_gather_is_described_from_the_first_index_it_reads(tmp_path):
    """Padded by 1 over a (1, 8, 30, 30) input the conv reads the odd rows and columns: the DMA starts at row 1,
    column 1, element 30 + 1 = 31, and steps 60 and 2."""
    mapping = _mapping({"Down": "core_allocation: [[0]]"}, "Down")
    _, ir = _solve(tmp_path, _down(shape=(1, 8, 30, 30), strides=[2, 2], pads=[1, 1, 1, 1]), mapping)
    read = _layout(ir, "Transfer(input)")["read"]
    assert read["offset"] == 31
    assert {stride for _, stride in read["access"]} >= {60, 2}


def test_a_strided_side_gives_the_steps_it_takes(tmp_path):
    """The same gather reads rows 1, 3, ..., 29 and columns 1, 3, 5, 7 of the row-major (1, 8, 30, 30) input in steps
    of (1, 1, 2, 2): its last element (7, 14, 3) sits at 31 + 7 x 900 + 14 x 2 x 30 + 3 x 2 = 7177."""
    mapping = _mapping({"Down": "core_allocation: [[0]]"}, "Down")
    _, ir = _solve(tmp_path, _down(shape=(1, 8, 30, 30), strides=[2, 2], pads=[1, 1, 1, 1]), mapping)
    row = _layout(ir, "Transfer(input)")
    read, write = row["read"], row["write"]
    assert row["tile_shape"] == [1, 8, 15, 4]
    assert read["index_steps"] == [1, 1, 2, 2] and write["index_steps"] == [1, 1, 1, 1]
    last = [size - 1 for size in row["tile_shape"]]
    address = read["offset"] + sum(
        i * step * stride for i, step, stride in zip(last, read["index_steps"], read["strides"], strict=True)
    )
    assert address == 7177


def test_a_dilated_strided_window_keeps_its_halo_on_the_index_steps(tmp_path):
    """A 3x3 conv of stride 2, dilation 2 and padding 2, four output columns a step: its window covers 6 even input
    columns and steps 4 of them, so it keeps a halo of 2 and moves 8 x 16 x 4 = 512 elements per firing."""
    mapping = _mapping({"Down": "core_allocation: [[0]]"}, "Down")
    allocation, _ = _solve(tmp_path, _down(3, strides=[2, 2], dilations=[2, 2], pads=[2, 2, 2, 2]), mapping)
    tr, route = _transfer(allocation, "Transfer(input)")
    copy = tr.outputs[0]
    ssis = allocation.space.ssis.get(copy)
    assert list(allocation.problem.workload.sliding_halos(copy, tr, allocation.mapping, ssis).values()) == [(3, 2)]
    assert allocation.space.moved_bits(tr, route) == prod((8, 16, 4)) * BF16


def test_a_core_of_a_split_strided_reader_takes_its_rows_on_the_index_steps(tmp_path):
    """Down split over two cores by output rows: each core reads 8 of the 16 even rows and 4 even columns, a
    (1, 8, 8, 4) view at strides (1, 1, 2, 2), out of the (1, 8, 16, 4) the transfer takes off chip."""
    mapping = _mapping({"Down": "core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D2, split: 2}]]"}, "Down")
    allocation, _ = _solve(tmp_path, _down(strides=[2, 2]), mapping)
    workload, (tr, _) = allocation.problem.workload, _transfer(allocation, "Transfer(input)")
    moved = workload.get_tensor_of_transfer_to_single_core(tr.outputs[0], tr, allocation.mapping, core=1)
    assert _view(moved) == ((1, 8, 8, 4), (0, 0, 0, 0), (1, 1, 2, 2))
    assert _view(workload.get_tensor_single_core(tr.outputs[0], tr, allocation.mapping, 1)) == _view(moved)
    taken = workload.get_tensor_of_transfer_from_single_core(tr.inputs[0], tr, allocation.mapping)
    assert _view(taken) == ((1, 8, 16, 4), (0, 0, 0, 0), (1, 1, 2, 2))


def test_a_split_producer_hands_only_the_indices_its_strided_reader_reads(tmp_path):
    """A split over two cores by output rows computes 16 rows of 8 columns of ``a`` each; B reads the even ones: each
    source core sends (1, 16, 8, 4), a view at strides (1, 1, 2, 2)."""
    split = "inter_core_tiling: [[{dim: D2, split: 2}]]"
    layers = {"A": f"core_allocation: [[0, 1]], {split}", "B": f"core_allocation: [[2, 3]], {split}"}
    allocation, _ = _solve(tmp_path, _chain(), _mapping(layers, "B"))
    tr, _ = _transfer(allocation, "Transfer(a)")
    taken = allocation.problem.workload.get_tensor_of_transfer_from_single_core(tr.inputs[0], tr, allocation.mapping)
    assert _view(taken) == ((1, 16, 8, 4), (0, 0, 0, 0), (1, 1, 2, 2))


def test_a_copy_of_an_intermediate_moves_only_what_its_strided_reader_reads(tmp_path):
    """A computes every index of its (1, 16, 32, 8) tile of ``a``; B on another core reads the even rows and columns,
    so each firing moves (1, 16, 16, 4) = 1024 elements, read in runs of the 16 channels and written in one run."""
    layers = {"A": "core_allocation: [[0]]", "B": "core_allocation: [[1]]"}
    allocation, ir = _solve(tmp_path, _chain(), _mapping(layers, "B"))
    tr, route = _transfer(allocation, "Transfer(a)")
    assert allocation.space.moved_bits(tr, route) == 1024 * BF16
    row = _layout(ir, "Transfer(a)")
    assert row["tile_shape"] == [1, 16, 16, 4]
    assert row["read"]["extent"] == [1, 16, 32, 8] and row["read"]["strides"] == [4096, 1, 128, 16]
    assert row["read"]["access"] == [[16, 256], [4, 32], [16, 1]]
    assert row["write"]["extent"] == [1, 16, 16, 4] and row["write"]["access"] == [[1024, 1]]
    assert (row["read_run_bytes"], row["write_run_bytes"]) == (32, 2048)


def test_each_source_core_hands_the_odd_indices_its_own_tile_holds(tmp_path):
    """A (16, 3x3, padded) on a (1, 8, 30, 30) input is split by columns: core 0 computes columns 0..14 of ``a``, core
    1 columns 15..29. B (1x1, stride 2, padded by 1) reads the odd rows and columns: 7 columns of core 0, 8 of core 1.
    At 2048 bits a read, core 0 reads its 16 x 15 x 7 x 16 = 26880 bits in 14 reads and core 1 its 30720 in 15; the
    transfer moves 16 x 15 x 15 elements, not the whole tensor."""
    nodes = [conv("A", "input", "w", "a"), conv("B", "a", "wb", "b", k=1, stride=2, pads=1)]
    workload = onnx_graph(nodes, {"input": [1, 8, 30, 30], "w": [16, 8, 3, 3], "wb": [16, 16, 1, 1]}, ["b"])
    split = "core_allocation: [[0, 1]], inter_core_tiling: [[{dim: D1, split: 2}]]"
    layers = f"- {{name: A, {split}}}\n- {{name: B, core_allocation: [[2]]}}\n"
    (tmp_path / "mapping.yaml").write_text(f"layers:\n{layers}fused_groups:\n- {{name: g, layers: [A, B]}}\n")
    estimate = evaluate_mapping(
        HARDWARE.format("tpu_like_quad_core"),
        workload,
        str(tmp_path),
        str(tmp_path / "mapping.yaml"),
        SolveOptions(artifacts=False),
    )
    cores = estimate.context.data["group_memory_accesses"][0]["cores"]
    reads = {c["core_id"]: (t["size_bits"], t["read"]) for c in cores for t in c["tensors"] if t["tensor"] == "a"}
    assert reads == {0: (26880, 14), 1: (30720, 15)}
    slots = estimate.context.get("allocation").solution.slot_latency_breakdown["slots"]
    (bits,) = [
        t["tensor_bits"] for slot in slots for t in slot["transfer_contributors"] if t["transfer"] == "Transfer(a)"
    ]
    assert bits == 16 * 15 * 15 * BF16


def test_aie_code_generation_refuses_a_strided_gather():
    """AIE code generation has no lowering for a gather: a node reading every other row and column is refused by
    name."""
    aie_code_generation = pytest.importorskip(
        "stream.stages.codegen.aie_code_generation",
        reason="the AIE dialects are a separate install, via stream-setup-aie",
    )
    with pytest.raises(ValueError, match=r"Down reads \w+ at steps \[1, 1, 2, 2\] .* a strided gather"):
        aie_code_generation.reject_strided_reads(load_workload(_down(strides=[2, 2])))
    aie_code_generation.reject_strided_reads(load_workload(_down(3, strides=[2, 2], pads=[1, 1, 1, 1])))
