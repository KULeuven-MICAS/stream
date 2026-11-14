"""TPU7x: the generated mappings split a SwiGLU as a tensor-parallel MLP, reduce-scatter its down projection's partial
sums, pipeline each layer's weight stream, and every workload the generator maps solves fused and layer by layer."""

import math
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from stream.api import SolveOptions, evaluate_mapping
from stream.hardware.ports import READ, WRITE
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.inputs.testing.workload.make_attention_head import make_multi_head_attention
from stream.inputs.testing.workload.make_resnet_subgraph import (
    ResNetPattern,
    ResNetSubgraphConfig,
    make_resnet_subgraph,
)
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload
from stream.mapping.capacity_tiler import CapacityTiler
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.opt.allocation.constraint_optimization.families.offchip import duplicate_offchip_bits
from stream.opt.allocation.constraint_optimization.families.traffic import _sides
from stream.opt.allocation.constraint_optimization.utils import active_fraction
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.steady_state.iteration_space import LoopEffect

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
AIE2_STRIX = "stream/inputs/aie/hardware/whole_array_strix.yaml"
SWIGLU = "stream/inputs/examples/workload/swiglu_256_4096_14336.onnx"
SMALL_SWIGLU = "stream/inputs/aie/workload/swiglu_256_512_2048.onnx"
LAYERS = ["Gemm_Left", "Gemm_Right", "Silu", "Elt_Mul", "Gemm_Down"]
ICI = {
    "CL(vmem_1, vmem_3, bw=364)",
    "CL(vmem_3, vmem_1, bw=364)",
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


def _long_down_projection(tmp_path):
    """The generator, and the down projection's group, of a 2048-token bf16 SwiGLU, 4096 -> 14336 -> 4096, each layer
    a group of its own."""
    make_swiglu_workload(2048, 4096, 14336, "bf16", "bf16", output_dir=str(tmp_path))
    accelerator, parsed = _parse(TPU_V7, str(tmp_path / "swiglu_2048_4096_14336.onnx"))
    generator = GenericMappingGenerator(accelerator, parsed, tempfile.mkdtemp())
    return generator, generator.groups(LAYERS)[-1]


DOWN_SUMS = 2048 * 4096 * 16 / 728
"""Cycles the long down projection's partial sums take between the chips: each of the four VMEMs holds sums of all
2048 x 4096 outputs at their bf16, and one 364-bit ICI link each way joins the chips."""
DOWN_COMPUTE = 2048 * 14336 * 4096 / (8 * 131072)
"""Cycles its eight MXUs of 131072 MACs take, 114688."""


def test_a_down_projection_whose_sums_cross_the_chips_is_cut_so_each_tile_s_sums_leave_once_computed(tmp_path):
    """The long down projection splits its hidden dim over the eight MXUs; its sums crossing the chips bound it. Cut
    into 4 x 2 tiles of tokens and outputs, the sums of each tile leave as soon as it is computed, the link busy all
    but the first tile's compute: R + C / 8 (no tile reads the weights twice). Cut into 4 output tiles of 7 hidden
    tiles each, a tile's sums leave only after its seventh: R + C / 28 + R / 4 - R / 28."""
    generator, down = _long_down_projection(tmp_path)
    cns = tuple(down.get_computation_nodes())
    tiler, cores, unroll, protected, node_unroll = generator._tiler(down, cns)
    tiling = generator._build_intra_core_tiling(down, cns)
    assert tiling == [{"dim": "Gemm_Down.D0", "tile": 512}, {"dim": "Gemm_Down.D2", "tile": 2048}]
    estimate = tiler.estimate(cns, cores, unroll, protected, tiling, node_unroll)
    times = (estimate.cycles, estimate.compute, estimate.transfer)
    assert times == pytest.approx((DOWN_SUMS + DOWN_COMPUTE / 8, DOWN_COMPUTE, DOWN_SUMS))
    hidden = [{"dim": "Gemm_Down.D1", "tile": 256}, {"dim": "Gemm_Down.D2", "tile": 1024}]
    assert tiler.estimate(cns, cores, unroll, protected, hidden, node_unroll).cycles == pytest.approx(
        DOWN_SUMS + DOWN_COMPUTE / 28 + DOWN_SUMS / 4 - DOWN_SUMS / 28
    )


@pytest.mark.slow
def test_the_down_projection_cut_the_tiler_picks_solves_faster_than_256_token_tiles(tmp_path):
    """Solved, the down projection the generator tiles takes less than 256-token tiles, which read its weights once
    per tile, and than the hidden cut whose sums wait for every hidden tile."""
    generator, down = _long_down_projection(tmp_path)
    workload = str(tmp_path / "swiglu_2048_4096_14336.onnx")

    def solved(tiling: list[dict] | None) -> float:
        stage = {"fusion_cut_points": LAYERS} | ({"intra_core_tiling": tiling} if tiling else {})
        options = SolveOptions(artifacts=False, stage_options=stage)
        return evaluate_mapping(TPU_V7, workload, tempfile.mkdtemp(), options=options).group_cycles[-1]

    picked = solved(None)
    assert picked < solved([{"dim": "Gemm_Down.D0", "tile": 256}])
    assert picked < solved([{"dim": "Gemm_Down.D1", "tile": 256}, {"dim": "Gemm_Down.D2", "tile": 1024}])


def test_a_code_generator_that_cannot_add_partial_sums_keeps_the_contraction_whole():
    layers, _ = _layers(AIE2_STRIX, SMALL_SWIGLU)
    assert all(entry["dim"] != "D1" for entry in layers["Gemm_Down"]["inter_core_tiling"][0])


def test_the_partial_sums_are_reduce_scattered_over_both_directions_of_the_ici_link(tmp_path):
    """Each TensorCore completes a quarter of the output: half of it crosses the chip boundary each way as partial
    sums at bf16, the output's type, as the MXUs declare no partial-sum format, once after the last of the 14
    iterations: ceil(256 x 4096 x 16 / 2 / 364) = 23046 cycles; each quarter is then written to HBM in bf16,
    ceil(256 x 4096 x 16 / (4 x 13400)) = 314 cycles. Both fire once in the run: they are its epilogue. The weights'
    stream from HBM bounds the iterations. The solve proves its latency optimal rather than stopping at the time
    limit."""
    estimate = evaluate_mapping(
        TPU_V7,
        SWIGLU,
        str(tmp_path),
        "stream/inputs/examples/mapping/swiglu_tpu_v7_fused.yaml",
        SolveOptions(artifacts=False),
    )
    solution = estimate.context.get("allocation").solution
    assert solution.solve_stats.mip_gap == 0
    tr, reduction = next(
        (tr, plan) for tr, plan in solution.transfer_routes.items() if ICI & {str(link) for link in plan.links_used}
    )
    assert sorted(m.name for m in reduction.targets) == ["vmem_0", "vmem_1", "vmem_2", "vmem_3"]
    assert {str(link): round(share, 2) for link, share in reduction.link_shares if str(link) in ICI} == dict.fromkeys(
        ICI, 0.5
    )
    (group,) = estimate.context.data["group_allocations"].values()
    detail = next(n for n in group["steady_state"]["tiled_graph"]["nodes"] if n["name"] == tr.name)
    firing = math.ceil(256 * 4096 * 16 / 2 / 364)
    write = math.ceil(256 * 4096 * 16 / (4 * 13400))
    assert (detail["copy_moved_bits"], detail["reuse_factor"], detail["route_cycles"]) == (
        [256 * 4096 * 16],
        14,
        firing,
    )
    for slot in solution.slot_latency_breakdown["slots"]:
        for transfer in slot["transfer_contributors"]:
            if transfer["transfer"] in (tr.name, "Transfer(output_2)"):
                assert slot["slot_latency_cycles"] == transfer["latency_contribution_cycles"] == 0
    latency = group["latency"]
    steady = 14 * latency["per_iteration"] - 13 * latency["overlap_between_iterations"]
    weights = 3 * 4096 * 14336 * 16 / (4 * 13400)
    assert weights < steady < 1.2 * weights
    assert estimate.cycles == steady + latency["fill"] >= steady + firing + write


_SOLVE = """
import hashlib, json, os, sys, tempfile
from stream.api import SolveOptions, evaluate_mapping
from stream.ir import AllocationIR
from stream.opt.allocation.constraint_optimization.allocation_model import AllocationModel
models = tempfile.mkdtemp()
solve = AllocationModel.solve
def write_then_solve(self, *args, **kwargs):
    self.model.write(os.path.join(models, f"{len(os.listdir(models))}.mps"))
    return solve(self, *args, **kwargs)
AllocationModel.solve = write_then_solve
estimate = evaluate_mapping(*sys.argv[1:3], tempfile.mkdtemp(), sys.argv[3], SolveOptions(artifacts=False))
written = [open(os.path.join(models, name)).read() for name in sorted(os.listdir(models))]
result = AllocationIR.from_internal(estimate.context.get("allocation")).model_dump(exclude={"solve"})
assert estimate.context.get("allocation").solution.solve_stats.mip_gap == 0, "the latency level did not close"
print(hashlib.sha256(json.dumps([written, result], sort_keys=True).encode()).hexdigest())
"""


def test_the_model_and_its_solution_do_not_depend_on_the_hash_seed():
    """The fused SwiGLU builds the same variables and constraints, in the same order, and solves to the same result,
    its latency proven optimal."""
    mapping = "stream/inputs/examples/mapping/swiglu_tpu_v7_fused.yaml"
    solved = {
        subprocess.run(
            [sys.executable, "-c", _SOLVE, TPU_V7, SWIGLU, mapping],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()[-1]
        for seed in range(2)
    }
    assert len(solved) == 1


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


def test_a_one_input_op_is_costed_on_the_vector_unit_through_its_registers_and_vmem(tmp_path):
    """The VPU's registers and VMEM hold any operand; a ReLU or a max pool reads one of them, and ZigZag costs it
    streaming through those memories rather than leaving it to an ideal-cycle estimate."""
    workload = make_resnet_subgraph(ResNetSubgraphConfig(pattern=ResNetPattern.FRONTEND), str(tmp_path))
    estimate = evaluate_mapping(TPU_V7, workload, str(tmp_path), options=SolveOptions(artifacts=False))
    entries = {
        node.type: entry for node, costs in estimate.context.get("cost_lut").lut.items() for entry in costs.values()
    }
    assert {"Relu", "MaxPool"} <= entries.keys()
    assert all(entry.metadata["backend"] == "zigzag" for entry in entries.values())
    assert entries["Relu"].latency_total >= entries["Relu"].ideal_cycle > 0


def test_a_convolution_spreads_its_contraction_over_the_mxu_rows(tmp_path):
    """The MXU maps an im2col convolution's whole contraction (channels by kernel window) over its rows, as the
    silicon does, rather than one loop per array dimension: the ResNet stem's 3 x 7 x 7 window fills far more of the
    array than its 3 channels alone."""
    workload = make_resnet_subgraph(ResNetSubgraphConfig(pattern=ResNetPattern.FRONTEND), str(tmp_path))
    estimate = evaluate_mapping(TPU_V7, workload, str(tmp_path), options=SolveOptions(artifacts=False))
    conv = next(
        entry
        for node, costs in estimate.context.get("cost_lut").lut.items()
        if node.type == "Conv"
        for entry in costs.values()
    )
    assert conv.cme.ideal_temporal_cycle < 4 * conv.cme.ideal_cycle


def test_multi_head_attention_splits_the_heads_its_matrix_units_split(tmp_path):
    """The scores and context split the heads over the eight MXUs, their operands indexed by the heads alone; the
    softmax could split heads or query rows, which shrink it alike, and takes the heads too, so each vector unit
    normalizes the heads of its own TensorCore and no score crosses a chip. Splitting its rows instead sends every
    score to every vector unit and makes the fused attention slower than writing the scores out."""
    seq, d_model, heads = 128, 512, 8
    d_head = d_model // heads
    value = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.BFLOAT16, shape)  # noqa: E731
    nodes = [helper.make_node("MatMul", ["x", f"w{p}"], [p], name=f"proj_{p}") for p in "QKV"]
    nodes += [
        helper.make_node("Transpose", ["K"], ["Kt"], perm=[0, 2, 1], name="transpose_K"),
        helper.make_node("MatMul", ["Q", "Kt"], ["scores"], name="scores"),
        helper.make_node("Softmax", ["scores"], ["attn"], axis=-1, name="softmax"),
        helper.make_node("MatMul", ["attn", "V"], ["ctx"], name="context"),
    ]
    weights = [value(f"w{p}", [heads, d_model, d_head]) for p in "QKV"]
    graph = helper.make_graph(
        nodes, "mha", [value("x", [seq, d_model]), *weights], [value("ctx", [heads, seq, d_head])]
    )
    path = str(tmp_path / "mha.onnx")
    onnx.save(onnx.shape_inference.infer_shapes(helper.make_model(graph)), path)

    estimate = evaluate_mapping(TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    mapping, workload = estimate.context.get("mapping"), estimate.context.get("workload")
    splits = {
        n.name: {str(d) for d, _ in mapping.get(n).inter_core_tiling[0]}
        for n in workload.get_computation_nodes()
        if n.name in ("scores", "context") or n.name.startswith("softmax")
    }
    assert len(set(map(frozenset, splits.values()))) == 1, splits


@pytest.mark.slow
def test_attention_at_4096_tokens_runs_its_projections_and_scores_apart_and_the_softmax_with_the_context(tmp_path):
    """At 4096 tokens the projections and the scores fused hold a tile of every head's scores next to the input and the
    weights: as the tiler estimates them, they take longer fused than each apart. The softmax and the context fused
    keep the scores on chip. Solved, the step takes less than the 1160827 cycles of the projections and scores
    fused."""
    workload = make_multi_head_attention(tmp_path / "mha.onnx", seq=4096)
    estimate = evaluate_mapping(TPU_V7, workload, str(tmp_path), options=SolveOptions(artifacts=False))
    groups = [[node.name for node in group.get_computation_nodes()] for group in estimate.context.get("sub_workloads")]
    assert groups == [
        ["proj_K"],
        ["proj_Q"],
        ["proj_V"],
        ["scores"],
        ["softmax_max", "softmax_exp", "softmax_sum", "softmax_div", "context"],
    ]
    assert estimate.cycles < 1160827


def test_attention_too_large_for_the_vmems_is_cut_after_its_projections_only(tmp_path):
    """At 8192 tokens the projections and the attention do not fit fused, but the attention does on its own: its
    scores stay on chip rather than every operator going to HBM and back."""
    _, groups = _layers(TPU_V7, make_multi_head_attention(tmp_path / "mha.onnx", seq=8192))
    assert [group["layers"] for group in groups] == [["proj_Q", "proj_K", "proj_V"], ["scores", "softmax", "context"]]


def test_a_copy_in_memory_holds_what_every_reader_of_it_needs(tmp_path):
    """The attention head's input feeds the query projection, split by rows, and the key and value projections,
    split by head dimension and so reading all of it: each TensorCore's VMEM holds it whole, loaded from HBM, rather
    than a quarter the others gather across the chips."""
    path = "stream/inputs/testing/workload/attention_head.onnx"
    estimate = evaluate_mapping(TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    allocation = estimate.context.get("allocation")
    problem = allocation.problem
    into_vmem = next(n for n in problem.workload.get_transfer_nodes() if n.name == "Transfer(I)")
    assert problem.mapping.get(into_vmem).inter_core_tiling == ((),) * len(
        problem.mapping.get(into_vmem).inter_core_tiling
    )
    hbm = problem.accelerator.offchip_memory
    assert {link.sender for link in allocation.solution.transfer_routes[into_vmem].links_used} == {hbm}


def test_the_key_projection_sits_out_the_query_tiles(tmp_path):
    """Attention cut along its queries, its VMEM holding one copy of the input for every projection: the key and value
    projections do not depend on the queries, so they run in half the steady-state iterations rather than in each, and
    no node of the steady state has two loops along one dim."""
    path = make_multi_head_attention(tmp_path / "mha.onnx", seq=256, d_model=1024, heads=8)
    accelerator, workload = _parse(TPU_V7, path)
    generated, _ = GenericMappingGenerator(accelerator, workload, tempfile.mkdtemp()).generate_all_groups()
    mapping = yaml.safe_load(open(generated[0]))
    mapping["fused_groups"][0]["intra_core_tiling"] = [{"dim": "proj_Q.D1", "tile": 128}]
    mapping_path = tmp_path / "queries_cut.yaml"
    mapping_path.write_text(yaml.safe_dump(mapping, sort_keys=False))
    estimate = evaluate_mapping(TPU_V7, path, str(tmp_path), str(mapping_path), SolveOptions(artifacts=False))
    problem = estimate.context.get("allocation").problem
    steady = problem.workload
    for node in steady.get_iteration_space_nodes():
        loops = [steady.leading_dim(d)[0] for d in steady.get_dims(node)]
        assert len(set(loops)) == len(loops), node.name
    nodes = {n.name: n for n in steady.get_computation_nodes()}
    (queries,) = problem.ssis[nodes["proj_Q"]].get_temporal_variables()
    assert queries.effect is LoopEffect.VARYING
    for name in ("proj_K", "proj_V"):
        (loop,) = problem.ssis[nodes[name]].get_temporal_variables()
        assert (loop.dimension, loop.effect) == (queries.dimension, LoopEffect.ABSENT)
        assert active_fraction(nodes[name], problem.ssis) == 0.5


@pytest.fixture(scope="module")
def tensor_parallel_swiglu():
    """A 256 x 2048 x 8192 SwiGLU on the TensorCores, fused in 16-token, 128-wide and 32-hidden tiles: its down
    projection's 16 x 128 output tile is summed over the four VMEMs and reduce-scattered, a quarter to each."""
    tiling = [
        {"dim": "Gemm_Left.D1", "tile": 128},
        {"dim": "Gemm_Down.D2", "tile": 128},
        {"dim": "Gemm_Left.D2", "tile": 32},
        {"dim": "Gemm_Left.D0", "tile": 16},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        workload = make_small_swiglu_workload(seq_len=256, embedding_dim=2048, hidden_dim=8192, output_dir=tmp)
        options = SolveOptions(artifacts=False, stage_options={"intra_core_tiling": tiling})
        return evaluate_mapping(TPU_V7, workload, tmp, options=options).context.get("allocation")


def _route(allocation, name: str):
    tr = next(tr for tr in allocation.solution.transfer_routes if tr.name == name)
    return tr, allocation.solution.transfer_routes[tr]


def test_each_vmem_writes_every_partial_sum_it_receives_and_reads_those_it_adds_and_sends(tensor_parallel_swiglu):
    """The VMEMs link as a path 0-1-3-2 and each gathers a quarter of the 16 x 128 sum, 512 partial sums sent at
    bf16, the output's type, as the MXUs declare no partial-sum format: 8192 bits. Each VMEM adds its two TensorCores'
    partial sums per quarter, 32768 bits read and written. vmem_0 receives one quarter (8192 written, 8192 read to add
    it) and sends three (24576 read): 40960 written, 65536 read. vmem_1 receives five and sends three on: 73728
    written, 98304 read. The ports and the estimate agree."""
    space = tensor_parallel_swiglu.space
    tr, route = _route(tensor_parallel_swiglu, "Transfer(output_1)")
    moved = space.moved_bits(tr, route)
    charged: dict[tuple[str, str], float] = {}
    for side in _sides(SimpleNamespace(space=space), tr, route):
        key = (side.port.key.memory, side.direction)
        charged[key] = charged.get(key, 0.0) + side.share * moved
    expected = {
        "vmem_0": (40960, 65536),
        "vmem_1": (73728, 98304),
        "vmem_2": (40960, 65536),
        "vmem_3": (73728, 98304),
    }
    for memory in dict.fromkeys(route.targets):
        estimated = tuple(space.memory_share(tr, route, memory, leaving) * moved for leaving in (False, True))
        ported = (charged[(memory.name, WRITE)], charged[(memory.name, READ)])
        assert estimated == pytest.approx(expected[memory.name]), memory.name
        assert ported == pytest.approx(expected[memory.name]), memory.name


def test_the_off_chip_traffic_counts_a_copy_per_link_a_multicast_leaves_hbm_on(tensor_parallel_swiglu):
    """A 16 x 128 bf16 input tile, 32768 bits, read by all four VMEMs: over the four HBM links at once HBM sends a copy
    on each, 3 x 32768 = 98304 bits beyond once; on the other routes it sends it once, as the solve chooses."""
    space = tensor_parallel_swiglu.space
    tr, chosen = _route(tensor_parallel_swiglu, "Transfer(input)")
    repeated = {route: duplicate_offchip_bits(space, tr, route, None) for route in space.path_choices[tr]}
    direct = [route for route in repeated if len(set(route.targets)) == 4 and len(route.hops) == 4]
    per_link = [route for route in direct if all(u.offchip for u, _, _ in route.hops)]
    assert len(per_link) == 1
    assert {route: bits for route, bits in repeated.items() if bits} == {per_link[0]: 3 * 32768}
    assert repeated[chosen] == 0
