"""Elementwise layers folded into the GEMM producing their input, run on its cores at its output tile."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from stream.frontends import load_workload
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.stages.generation.placement_generation import PlacementGenerationStage
from stream.stages.parsing.accelerator_parser import parse_accelerator
from stream.workload.post_ops import fold_post_ops
from stream.workload.utils import unfinished_reductions

ROOT = Path(__file__).parents[2]
STRIX = ROOT / "stream/inputs/aie/hardware/whole_array_strix.yaml"
LIBRARY = ROOT / "stream/inputs/aie/kernels/aie2p.toml"


@pytest.fixture(scope="module")
def strix():
    return parse_accelerator(STRIX)


@pytest.fixture(scope="module")
def swiglu(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("swiglu"))
    return load_workload(make_swiglu_workload(256, 512, 2048, "bf16", "bf16", last_gemm_down=True, output_dir=out))


def _node(workload, name):
    return workload.get_node_by_name(name)


def test_a_unary_post_op_writes_its_layers_output(swiglu):
    folded, into = fold_post_ops(swiglu, ["Silu"])
    gate = _node(folded, "Gemm_Left")
    assert into == {"Silu": "Gemm_Left"}
    assert [t.name for t in gate.outputs] == ["left_swished"]
    assert gate.inputs == _node(swiglu, "Gemm_Left").inputs
    assert "Silu" not in {n.name for n in folded.node_list}


def test_a_binary_post_op_folds_into_the_later_gemm_and_reads_its_other_input_at_the_output_index(swiglu):
    """The product reads the SiLU's output and the up projection's: it goes with the up projection, the later of the
    two, so the other input is ready before the up projection's sum is done."""
    folded, into = fold_post_ops(swiglu, ["Silu", "Elt_Mul"])
    up = _node(folded, "Gemm_Right")
    assert into == {"Silu": "Gemm_Left", "Elt_Mul": "Gemm_Right"}
    assert [t.name for t in up.inputs] == ["input", "weights_2", "left_swished"]
    assert up.operand_mapping[2] == up.operand_mapping[-1]
    assert [t.name for t in up.outputs] == ["intermediate"]


def test_only_an_elementwise_layer_over_a_sum_folds(swiglu):
    with pytest.raises(ValueError, match="Gemm_Down is not elementwise"):
        fold_post_ops(swiglu, ["Gemm_Down"])


def test_a_post_op_input_is_read_once_the_readers_own_sum_is_done(swiglu):
    """The up projection reads the gate's output only with its own output, after the sum over the embedding both run in
    step: the group may cut that sum into blocks. Unfolded, the product walks the gate's output across it."""
    folded, _ = fold_post_ops(swiglu, ["Silu", "Elt_Mul"])
    group = folded.split_fusion_groups(cut_points=["Gemm_Right"])[0]
    assert not unfinished_reductions(group)


def test_a_post_op_pair_shares_memory_row_by_row(swiglu):
    folded, _ = fold_post_ops(swiglu, ["Silu", "Elt_Mul"])
    gate, up = _node(folded, "Gemm_Left"), _node(folded, "Gemm_Right")
    assert PlacementGenerationStage._post_op_pair([gate, up], [2, 2])
    assert not PlacementGenerationStage._post_op_pair([gate, up], [1, 3])
    plain_gate, plain_up = _node(swiglu, "Gemm_Left"), _node(swiglu, "Gemm_Right")
    assert not PlacementGenerationStage._post_op_pair([plain_gate, plain_up], [2, 2])


def _gemm(**fields):
    from stream.compiler.kernels.library import KernelLibrary

    registry = pytest.importorskip("stream.compiler.kernels.registry", reason="the AIE dialects")

    library = KernelLibrary.load(LIBRARY)
    return registry.AIE_KERNELS["gemm"](m=64, k=32, n=64, layout="default", library=library, **fields)


def test_a_binary_post_op_takes_its_input_in_the_output_tiling():
    plain, gated = _gemm(), _gemm(post_ops=("eltwise_mul",))
    layouts = gated.operand_layouts()
    assert len(layouts) == 4 and layouts[2] == layouts[3] == plain.operand_layouts()[2]
    assert plain.unique_name != gated.unique_name


def test_post_ops_keep_a_tile_of_their_own_and_cost_their_kernels():
    """The sum is cast into a bf16 tile the first post-op reads; with no measured call, the vector family's 16
    operations a cycle price it."""
    assert _gemm().SCRATCH_BITS == 0
    assert _gemm(post_ops=("silu",)).SCRATCH_BITS == 64 * 64 * 16
    assert _gemm(post_ops=("silu",)).post_op_cycles(4096) == 4096 // 16


def test_post_ops_are_priced_once_per_finished_sum(swiglu):
    """A call adds into the output; only the trip of the fused loop over the sum that finishes it runs the post-ops, so
    their cost spreads over the trips of that loop."""
    from stream.stages.estimation.aie_cost_estimator import AIECostEstimator

    folded, _ = fold_post_ops(swiglu, ["Silu"])
    gate = _node(folded, "Gemm_Left")
    dims = folded.get_dims(gate)
    sizes = [folded.get_dimension_size(d) for d in dims]
    m, k, n = sizes
    estimator = AIECostEstimator(folded, None, {dims[1]: 16})
    assert estimator._finished_outputs(gate, sizes, m * k * n) == m * n / 16


def _memory(accelerator, name):
    return accelerator.memories[name]


def test_a_transfer_to_two_memory_tiles_drives_a_shim_channel_in_each_column(strix):
    """Code generation gives each memory tile its own fifo through its own column's shim, so a read that the route
    brings in over one shim link and forwards to the second tile still takes a channel of each column's port."""
    from stream.opt.allocation.constraint_optimization.families.dma import _port_channels
    from stream.opt.allocation.constraint_optimization.hardware import build_hardware_facts

    dram, left, right = _memory(strix, "dram"), _memory(strix, "tile_0_1"), _memory(strix, "tile_1_1")

    def memory_links(memory, leaving):
        edges = strix.network.out_edges if leaving else strix.network.in_edges
        return tuple(dict.fromkeys(data["cl"] for *_, data in edges(memory, data=True)))

    entry = next(link for link in memory_links(dram, True) if left in link.ports)

    class Hashable(SimpleNamespace):
        __hash__ = object.__hash__

    route = Hashable(
        sources=(dram,),
        targets=(left, right),
        link_shares=((entry, 1.0),),
        links_at=lambda memory, leaving: (entry,) if memory is dram and leaving else (),
        pairs=((0, 0), (0, 1)),
    )
    transfer = Hashable(inputs="in", outputs="out")
    space = SimpleNamespace(
        accelerator=strix,
        transfer_nodes=[transfer],
        path_choices={transfer: [route]},
        walk_shares=lambda tr, choice: {},
        operand_role=lambda tr, read_side: "input1",
        memory_links=memory_links,
        hardware=build_hardware_facts(strix, 8),
        overlaps=lambda tr: False,
        placement_width=lambda tensors: 1,
        distinct_slice_width=lambda tensors: 1 if tensors == "in" else 2,
    )
    ctx = SimpleNamespace(space=space, vars=SimpleNamespace(y={(transfer, route): SimpleNamespace(_raw=1)}))
    _, out = _port_channels(ctx)
    assert sorted((port.key.port, fifos) for (_, port), fifos in out.items()) == [("mm2s_c0", 1), ("mm2s_c1", 1)]


def test_a_folded_gemm_is_priced_as_the_gemm_and_its_post_ops(swiglu, strix):
    from stream.mapping.mapping import FusedGroup, Mapping, NodeMapping
    from stream.stages.estimation.aie_cost_estimator import AIECostEstimator

    folded, _ = fold_post_ops(swiglu, ["Silu"])
    gate = _node(folded, "Gemm_Left")
    core = next(c for c in strix.core_list if getattr(c, "type", None) == "compute")
    dims = folded.get_dims(gate)
    m, k, n = (folded.get_dimension_size(d) for d in dims)

    def entry(kernel):
        mapping = Mapping(
            {gate: NodeMapping(resource_allocation=((core,),), kernel=kernel)}, [FusedGroup("g", ("Gemm_Left",))]
        )
        return AIECostEstimator(folded, mapping, {dims[1]: 16}).estimate(gate, core).latency_total

    with_silu = _gemm(post_ops=("silu",))
    assert entry(with_silu) == entry(_gemm()) + with_silu.post_op_cycles(m * n / 16)


def test_a_post_op_pair_takes_alternate_rows_and_pairs_cores_beside_each_other(swiglu, strix):
    """Each up projection core sits right above the gate core computing the same output tile, so it reads the gate's
    tile from the memory they share."""
    from stream.mapping.mapping import FusedGroup, Mapping, NodeMapping
    from stream.stages.generation.placement_generation import PlacementGenerationStage

    folded, _ = fold_post_ops(swiglu, ["Silu", "Elt_Mul"])
    group = folded.split_fusion_groups(cut_points=["Gemm_Right"])[0]
    gate, up = _node(group, "Gemm_Left"), _node(group, "Gemm_Right")
    kernels = {gate: _gemm(post_ops=("silu",)), up: _gemm(post_ops=("eltwise_mul",))}
    stage = PlacementGenerationStage.__new__(PlacementGenerationStage)
    stage.workload = group
    stage.mapping = Mapping(
        {node: NodeMapping(kernel=kernel) for node, kernel in kernels.items()},
        [FusedGroup("g", ("Gemm_Left", "Gemm_Right"))],
    )
    stage._pending_fallbacks, stage._pending_reserves = [], []
    grid = {(c.col_id, c.row_id): c for c in strix.core_list if c.type == "compute" and c.col_id is not None}
    stage._place_group([gate, up], grid)
    (gate_cores,), (up_cores,) = (stage.mapping.get(node).resource_allocation for node in (gate, up))
    rows = sorted({row for _, row in grid})
    assert {c.row_id for c in gate_cores} == set(rows[0::2]) and {c.row_id for c in up_cores} == set(rows[1::2])
    assert [(u.col_id, u.row_id - 1) for u in up_cores] == [(g.col_id, g.row_id) for g in gate_cores]
    assert stage.mapping.get(gate).inter_core_tiling == stage.mapping.get(up).inter_core_tiling


def _post_op_mapping(post_ops=("Silu", "Elt_Mul"), layers=("Gemm_Left", "Gemm_Right", "Silu", "Elt_Mul")):
    kernels = {
        name: {"name": "gemm", "kwargs": {"m": 32, "k": 32, "n": 64, "layout": "default"}}
        for name in ("Gemm_Left", "Gemm_Right", "Gemm_Down")
    }
    kernels |= {"Silu": {"name": "silu", "kwargs": {}}, "Elt_Mul": {"name": "eltwise_mul", "kwargs": {}}}
    return {
        "layers": [{"name": name, "core_allocation": [], "kernel": kernel} for name, kernel in kernels.items()],
        "fused_groups": [
            {"name": "front", "layers": list(layers), "post_ops": list(post_ops)},
            {"name": "down", "layers": ["Gemm_Down"]},
        ],
    }


def test_the_validator_takes_a_groups_post_ops():
    from stream.parser.mapping_validator import MappingValidator

    validator = MappingValidator(_post_op_mapping())
    assert validator.validate(), validator.errors
    assert validator.normalized_data["fused_groups"][0]["post_ops"] == ["Silu", "Elt_Mul"]


def _fold(workload, accelerator, mapping):
    from stream.stages.generation.fixed_mapping_generation import FixedMappingGenerationStage

    stage = FixedMappingGenerationStage.__new__(FixedMappingGenerationStage)
    stage.workload, stage.accelerator = workload, accelerator
    stage._fold_post_ops(mapping)
    return stage


def test_the_stage_folds_a_groups_post_ops_into_its_gemms(swiglu, strix):
    pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")
    mapping = _post_op_mapping()
    stage = _fold(swiglu, strix, mapping)
    kernels = {layer["name"]: layer["kernel"]["kwargs"].get("post_ops") for layer in mapping["layers"]}
    assert kernels["Gemm_Left"] == ("silu",) and kernels["Gemm_Right"] == ("eltwise_mul",)
    assert kernels["Gemm_Down"] is None
    assert mapping["fused_groups"][0]["layers"] == ["Gemm_Left", "Gemm_Right"]
    assert {n.name for n in stage.workload.node_list} >= {"Gemm_Left", "Gemm_Right"}
    assert not {"Silu", "Elt_Mul"} & {n.name for n in stage.workload.node_list}


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (_post_op_mapping(post_ops=("Sigmoid",)), "fused group front: post-op Sigmoid must be one of its layers"),
        (_post_op_mapping(post_ops=("Gemm_Left",)), "fused group front: post-op Gemm_Left must be .* not gemm"),
        (
            _post_op_mapping(layers=("Gemm_Left", "Silu")),
            "fused group front: post-op Elt_Mul must be one of its layers",
        ),
        (
            _post_op_mapping(post_ops=("Silu",), layers=("Gemm_Right", "Silu", "Elt_Mul")),
            "fused group front: post-op Silu folds into Gemm_Left, not a GEMM of it",
        ),
    ],
)
def test_a_misnamed_post_op_is_refused_before_anything_changes(swiglu, strix, mapping, message):
    pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")
    from copy import deepcopy

    from stream.stages.generation.fixed_mapping_generation import FixedMappingGenerationStage

    before = deepcopy(mapping)
    stage = FixedMappingGenerationStage.__new__(FixedMappingGenerationStage)
    stage.workload, stage.accelerator = swiglu, strix
    with pytest.raises(ValueError, match=message):
        stage._fold_post_ops(mapping)
    assert mapping == before and stage.workload is swiglu


def test_post_ops_are_refused_off_aie(swiglu):
    tpu = parse_accelerator(ROOT / "stream/inputs/examples/hardware/tpu_like_quad_core.yaml")
    with pytest.raises(ValueError, match="only AIE cores price and run"):
        _fold(swiglu, tpu, _post_op_mapping())


def test_a_fixed_mapping_runs_its_groups_on_the_folded_workload(swiglu, strix, tmp_path):
    pytest.importorskip("snaxc", reason="the AIE dialects are a separate install, via stream-setup-aie")
    import yaml

    from stream.stages.context import StageContext
    from stream.stages.generation.fixed_mapping_generation import FixedMappingGenerationStage
    from stream.stages.stage import Stage

    seen = {}

    class Record(Stage):
        reads = ("sub_workloads",)

        def is_leaf(self) -> bool:
            return True

        def run(self):
            seen["groups"] = [[n.name for n in w.get_computation_nodes()] for w in self.ctx.get("sub_workloads")]
            yield self.ctx

    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump(_post_op_mapping()))
    ctx = StageContext.from_kwargs(accelerator=strix, workload=swiglu, mapping_path=str(path))
    list(FixedMappingGenerationStage([Record], ctx).run())
    assert seen["groups"] == [["Gemm_Left", "Gemm_Right"], ["Gemm_Down"]]
