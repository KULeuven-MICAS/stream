"""Every tensor moves in one transfer to all its readers, however the readers are grouped, placed or tiled."""

from collections import Counter

import pytest
import yaml

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.workload.blocks import available_blocks, build_block

TPU = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
FAILING_ON_MAIN = {"moe", "flash_attention"}  # unrelated to transfers: they fail before lowering finishes
SWIGLU = ("Gemm_Left", "Gemm_Right", "Silu", "Elt_Mul", "Gemm_Down")


def _graphs(estimate):
    return [group["steady_state"]["tiled_graph"] for group in estimate.context.get("group_allocations").values()]


def _transfers_per_source(graph):
    transfers = {n["name"] for n in graph["nodes"] if n["kind"] == "transfer"}
    return Counter(e["source"] for e in graph["edges"] if e["target"] in transfers)


@pytest.mark.parametrize("block", [b.key for b in available_blocks() if b.key not in FAILING_ON_MAIN])
def test_no_tensor_of_a_block_moves_in_two_transfers(tmp_path, block):
    estimate = evaluate_mapping(TPU, build_block(block), str(tmp_path), options=SolveOptions(artifacts=False))
    for graph in _graphs(estimate):
        assert max(_transfers_per_source(graph).values()) == 1


@pytest.mark.parametrize("split", [False, True], ids=["fused", "split"])
@pytest.mark.parametrize("right_core", [0, 1], ids=["same_core", "two_cores"])
@pytest.mark.parametrize("tiled", [False, True], ids=["untiled", "tiled"])
def test_both_swiglu_projections_read_their_input_from_one_transfer(tmp_path, split, right_core, tiled):
    """Gemm_Left and Gemm_Right read the same input: one copy feeds both, and a tiling of one reaches the other."""
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
    tiles = {op["name"]: op["tensors"][0]["shape"] for op in projections["operators"] if op["name"] in SWIGLU[:2]}
    assert tiles["Gemm_Left"] == tiles["Gemm_Right"] == [8 if tiled else 32, 32]
    for graph in _graphs(estimate):
        assert max(_transfers_per_source(graph).values()) == 1
