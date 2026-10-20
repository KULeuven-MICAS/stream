"""Every tensor moves in one transfer to all its readers, however the readers are grouped, placed or tiled, and in one
more to its group's output edge when a later group reads it."""

from collections import defaultdict

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.workload.blocks import available_blocks, build_block

TPU = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"  # stages through its memory cores
NOT_SOLVABLE = {"flash_attention"}  # multi-output nodes, which the solve does not support yet
SWIGLU = ("Gemm_Left", "Gemm_Right", "Silu", "Elt_Mul", "Gemm_Down")


def _graphs(estimate):
    return [group["steady_state"]["tiled_graph"] for group in estimate.context.get("group_allocations").values()]


def _moves(graph):
    """Per source, the nodes each transfer it feeds ends at, through any staging hop."""
    transfers = {n["name"] for n in graph["nodes"] if n["kind"] == "transfer"}
    targets = defaultdict(list)
    for edge in graph["edges"]:
        targets[edge["source"]].append(edge["target"])

    def ends(node):
        return {end for target in targets[node] for end in (ends(target) if target in transfers else {target})}

    return {src: [ends(t) for t in feeds if t in transfers] for src, feeds in targets.items() if src not in transfers}


def _carrying(graph, tensor):
    return {n["name"] for n in graph["nodes"] if n["kind"] == "transfer" and n["tensor"] == tensor}


def _assert_one_transfer_per_tensor(graph):
    computes = {n["name"] for n in graph["nodes"] if n["kind"] == "compute"}
    for source, moves in _moves(graph).items():
        assert sum(bool(m & computes) for m in moves) <= 1, (source, moves)  # to its readers in the group
        assert sum(not m & computes for m in moves) <= 1, (source, moves)  # to the output edge


@pytest.mark.parametrize("block", [b.key for b in available_blocks() if b.key not in NOT_SOLVABLE])
def test_no_tensor_of_a_block_moves_in_two_transfers(tmp_path, block):
    estimate = evaluate_mapping(TPU, build_block(block), str(tmp_path), options=SolveOptions(artifacts=False))
    for graph in _graphs(estimate):
        _assert_one_transfer_per_tensor(graph)


def _value(name):
    return helper.make_tensor_value_info(name, TensorProto.BFLOAT16, [64, 64])


@pytest.mark.parametrize("hardware", [TPU, TPU_V7], ids=["direct", "staged"])
def test_a_tensor_read_in_its_group_and_after_it_moves_once_to_each(tmp_path, hardware):
    """``h`` feeds ``mm2`` in its group and ``mm3`` after the cut: one transfer to ``mm2``, one to the output edge."""
    layers = (("mm1", "x", "w1", "h"), ("mm2", "h", "w2", "a"), ("mm3", "a", "h", "b"))
    nodes = [helper.make_node("MatMul", [x, w], [y], name=name) for name, x, w, y in layers]
    graph = helper.make_graph(nodes, "g", [_value(t) for t in ("x", "w1", "w2")], [_value("b")])
    path = str(tmp_path / "shared.onnx")
    onnx.save(onnx.shape_inference.infer_shapes(helper.make_model(graph)), path)
    options = SolveOptions(artifacts=False, stage_options={"fusion_cut_points": ["mm2"]})
    first, second = _graphs(evaluate_mapping(hardware, path, str(tmp_path), options=options))

    moves = _moves(first)["mm1"]
    assert len(moves) == 2 and {"mm2"} in moves
    (export,) = (move for move in moves if move != {"mm2"})
    (edge,) = export  # the output edge, reached under the tensor's own name
    assert any(e["target"] == edge and e["source"] in _carrying(first, "h") for e in first["edges"])
    assert any("mm3" in move for move in _moves(second)[edge.replace("_out", "_in")])
    for steady in (first, second):
        _assert_one_transfer_per_tensor(steady)


@pytest.mark.parametrize("split", [False, True], ids=["fused", "split"])
@pytest.mark.parametrize("right_core", [0, 1], ids=["same_core", "two_cores"])
@pytest.mark.parametrize("tiled", [False, True], ids=["untiled", "tiled"])
def test_both_swiglu_projections_read_their_input_from_one_transfer(tmp_path, split, right_core, tiled):
    """Gemm_Left and Gemm_Right read the same input: one transfer feeds both, one copy per core, and a tiling of one
    reaches the other."""
    groups = [SWIGLU[:2], SWIGLU[2:]] if split else [SWIGLU]
    fused = [{"name": f"group_{i}", "layers": list(g)} for i, g in enumerate(groups)]
    if tiled:
        fused[0]["intra_core_tiling"] = [{"dim": "Gemm_Left.D0", "tile": 8}]
    cores = {name: 0 if name == "Gemm_Left" else right_core for name in SWIGLU}
    mapping = tmp_path / "mapping.yaml"
    layers = [{"name": name, "core_allocation": [[core]]} for name, core in cores.items()]
    mapping.write_text(yaml.safe_dump({"layers": layers, "fused_groups": fused, "runtime_args": {}}))
    workload = make_small_swiglu_workload(seq_len=32, embedding_dim=32, hidden_dim=64, output_dir=str(tmp_path))
    estimate = evaluate_mapping(TPU, workload, str(tmp_path), str(mapping), SolveOptions(artifacts=False))

    projections = estimate.context.get("group_allocations")[0]["steady_state"]
    edges = projections["tiled_graph"]["edges"]
    assert {e["target"] for e in edges if e["source"] == "Transfer(input)"} == {"Gemm_Left", "Gemm_Right"}
    (transfer,) = (n for n in projections["tiled_graph"]["nodes"] if n["name"] == "Transfer(input)")
    assert transfer["copies"] == (1 if right_core == 0 else 2)  # one copy per core reading it
    tiles = {op["name"]: op["tensors"][0]["shape"] for op in projections["operators"] if op["name"] in SWIGLU[:2]}
    assert tiles["Gemm_Left"] == tiles["Gemm_Right"] == [8 if tiled else 32, 32]
    for graph in _graphs(estimate):
        _assert_one_transfer_per_tensor(graph)


@pytest.mark.parametrize(("right", "copies"), [([("D0", 2), ("D2", 2)], 1), ([("D2", 2), ("D0", 2)], 2)])
def test_readers_on_the_same_cores_share_a_copy_only_when_each_core_needs_the_same_rows(tmp_path, right, copies):
    """Both projections split the input rows and their outputs over cores 0-3; in opposite orders a core needs other
    rows for each, so each gets its own copy."""
    splits = {"Gemm_Left": [("D0", 2), ("D2", 2)], "Gemm_Right": right}
    layers = [
        {"name": name, "core_allocation": [[0, 1, 2, 3]] if name in splits else [[0]]}
        | ({"inter_core_tiling": [[{"dim": d, "split": f} for d, f in splits[name]]]} if name in splits else {})
        for name in SWIGLU
    ]
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(yaml.safe_dump({"layers": layers, "fused_groups": [{"name": "g", "layers": list(SWIGLU)}]}))
    workload = make_small_swiglu_workload(seq_len=32, embedding_dim=32, hidden_dim=64, output_dir=str(tmp_path))
    estimate = evaluate_mapping(TPU, workload, str(tmp_path), str(mapping), SolveOptions(artifacts=False))
    (transfer,) = (n for n in _graphs(estimate)[0]["nodes"] if n["name"] == "Transfer(input)")
    assert transfer["copies"] == copies
