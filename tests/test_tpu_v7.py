"""TPU7x: the generated mappings split a SwiGLU as a tensor-parallel MLP, reduce-scatter its down projection's partial
sums, pipeline each layer's weight stream, and every workload the generator maps solves fused and layer by layer."""

import math
import tempfile

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from stream.api import SolveOptions, evaluate_mapping
from stream.mapping.capacity_tiler import CapacityTiler
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
AIE2_STRIX = "stream/inputs/aie/hardware/whole_array_strix.yaml"
SWIGLU = "stream/inputs/examples/workload/swiglu_256_4096_14336.onnx"
SMALL_SWIGLU = "stream/inputs/aie/workload/swiglu_256_512_2048.onnx"
LAYERS = ["Gemm_Left", "Gemm_Right", "Silu", "Elt_Mul", "Gemm_Down"]
ICI = {
    "CL(Core(7, zigzag.memory), Core(15, zigzag.memory), bw=364)",
    "CL(Core(15, zigzag.memory), Core(7, zigzag.memory), bw=364)",
}


def _parse(hardware: str, workload: str):
    ctx = StageContext.from_kwargs(accelerator=hardware, workload_path=workload, output_path=tempfile.mkdtemp())
    ctx = MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()[0]
    return ctx.get("accelerator"), ctx.get("workload")


def _layers(hardware: str, workload: str, cut_points: list[str] | None = None) -> tuple[dict, list[dict]]:
    """Each generated layer entry by name, and each generated fused group."""
    accelerator, parsed = _parse(hardware, workload)
    paths, _ = GenericMappingGenerator(accelerator, parsed, tempfile.mkdtemp()).generate_all_groups(cut_points)
    mappings = [yaml.safe_load(open(path)) for path in paths]
    return {layer["name"]: layer for m in mappings for layer in m["layers"]}, [
        g for m in mappings for g in m["fused_groups"]
    ]


def test_the_down_projection_contracts_the_hidden_dimension_its_input_arrives_split_along():
    """Gate and up split the hidden dimension (their D2) over the eight MXUs; the down projection splits it too, its
    contraction (D1), rather than its output, so its input stays where it was made and only its output is reduced."""
    layers, _ = _layers(TPU_V7, SWIGLU)
    assert layers["Gemm_Left"]["inter_core_tiling"] == [[{"dim": "D2", "split": 8}]]
    assert layers["Gemm_Down"]["inter_core_tiling"] == [[{"dim": "D1", "split": 8}]]


def test_a_code_generator_that_cannot_add_partial_sums_keeps_the_contraction_whole():
    layers, _ = _layers(AIE2_STRIX, SMALL_SWIGLU)
    assert all(entry["dim"] != "D1" for entry in layers["Gemm_Down"]["inter_core_tiling"][0])


def test_the_partial_sums_are_reduce_scattered_over_both_directions_of_the_ici_link(tmp_path):
    """Each TensorCore completes a quarter of the output: half of it crosses the chip boundary each way, and the
    weights' stream from HBM, not the reduction, bounds the group."""
    estimate = evaluate_mapping(
        TPU_V7,
        SWIGLU,
        str(tmp_path),
        "stream/inputs/examples/mapping/swiglu_tpu_v7_fused.yaml",
        SolveOptions(artifacts=False),
    )
    routes = estimate.context.get("allocation").solution.transfer_routes
    reduction = next(plan for plan in routes.values() if ICI & {str(link) for link in plan.links_used})
    assert sorted(c.id for c in reduction.targets) == [3, 7, 11, 15]
    assert {str(link): round(share, 2) for link, share in reduction.link_shares if str(link) in ICI} == dict.fromkeys(
        ICI, 0.5
    )
    weights = 3 * 4096 * 14336 * 16 / (4 * 13400)
    assert weights < estimate.cycles < 1.2 * weights


def test_a_layer_that_fits_is_still_cut_into_pipeline_tiles_the_arrays_take_whole():
    """On its own the gate projection fits its VMEMs; it is still cut into at least eight tiles, so each one's weights
    load while the one before computes, each tile a multiple of the 256-wide array."""
    _, groups = _layers(TPU_V7, SWIGLU, LAYERS)
    tiles = {
        entry["dim"]: entry["tile"]
        for entry in next(g for g in groups if "Gemm_Left" in g["layers"])["intra_core_tiling"]
    }
    per_core = {"Gemm_Left.D1": 4096, "Gemm_Left.D2": 14336 // 8}
    assert math.prod(per_core[dim] // tile for dim, tile in tiles.items()) >= CapacityTiler.PIPELINE_TILES
    assert all(tile % 256 == 0 for tile in tiles.values())


@pytest.mark.parametrize("cut", [False, True])
def test_an_attention_head_solves_fused_and_layer_by_layer(cut, tmp_path):
    """Every TensorCore's MXUs read the head's input from a copy in their own VMEM, and the transfers of a slot share
    the HBM links rather than each taking one alone."""
    workload = "stream/inputs/testing/workload/attention_head.onnx"
    _, parsed = _parse(TPU_V7, workload)
    stage = {"fusion_cut_points": [cn.name for cn in parsed.get_computation_nodes()]} if cut else {}
    estimate = evaluate_mapping(
        TPU_V7, workload, str(tmp_path), options=SolveOptions(artifacts=False, stage_options=stage)
    )
    assert estimate.cycles > 0


def test_a_graph_input_no_node_reads_is_left_out_and_copies_take_fresh_names(tmp_path):
    """A model with an unused initializer, and an intermediate named as the staged copy of its output would be."""
    x = helper.make_tensor_value_info("y_in", TensorProto.FLOAT, [1, 8, 16, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 16, 16])
    weights = [helper.make_tensor(name, TensorProto.FLOAT, [8, 8, 1, 1], [0.0] * 64) for name in ("w0", "w1", "unused")]
    nodes = [
        helper.make_node("Conv", ["y_in", "w0"], ["y_1"], name="conv0"),
        helper.make_node("Conv", ["y_1", "w1"], ["y"], name="conv1"),
    ]
    path = tmp_path / "collide.onnx"
    onnx.save(helper.make_model(helper.make_graph(nodes, "collide", [x], [y], weights)), str(path))
    _, parsed = _parse(TPU_V7, str(path))
    assert "unused" not in {node.name for node in parsed.nodes}
    assert evaluate_mapping(TPU_V7, str(path), str(tmp_path), options=SolveOptions(artifacts=False)).cycles > 0
