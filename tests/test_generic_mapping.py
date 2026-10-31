"""Tests for GenericMappingGenerator and the generic CO pipeline.

The two-conv workload is used as a fast single-group proxy (< 60s).
The Conv-Relu-Flatten-Gemm workload tests multi-group pipeline mechanics.
"""

import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from stream.allocation.allocation import Allocation
from stream.api import SolveOptions, evaluate_mapping
from stream.cost_model.core_cost import IDEAL_CYCLE_BACKEND
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.inputs.testing.workload.make_conv_relu_flatten_gemm import (
    make_conv_relu_flatten_gemm_workload,
)
from stream.inputs.testing.workload.make_resnet_subgraph import (
    ResNetPattern,
    ResNetSubgraphConfig,
    make_resnet_subgraph,
)
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.parser.mapping_factory import MappingFactory
from stream.parser.mapping_validator import MappingValidator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage
from stream.workload.normalization import expand_normalizations

_ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
_TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
_FUSEMAX = "stream/inputs/examples/hardware/fusemax.yaml"
_WORKLOAD_CONFIG = TwoConvWorkloadConfig(
    batch_size=1,
    in_channels=8,
    height=32,
    width=32,
    out_channels_1=16,
    out_channels_2=32,
    kernel_size=3,
    in_dtype="bf16",
    weight_dtype="bf16",
)


def _parse_workload_and_accelerator(workload_path: str | None = None, accelerator: str = _ACCELERATOR):
    """Helper: parse hardware + workload into objects via the pipeline stages."""
    if workload_path is None:
        workload_path = make_2_conv_workload(_WORKLOAD_CONFIG)
    ctx = StageContext.from_kwargs(
        accelerator=accelerator,
        workload_path=workload_path,
        output_path=tempfile.mkdtemp(),
    )
    stages = [AcceleratorParserStage, ONNXModelParserStage, LeafStage]
    mainstage = MainStage(stages, ctx)
    ctxs = mainstage.run()
    assert len(ctxs) == 1
    return ctxs[0].get("accelerator"), ctxs[0].get("workload")


def test_generator_produces_all_fields():
    """GenericMappingGenerator produces mapping with core_allocation,
    inter_core_tiling, fused_groups, and intra_core_tiling populated."""
    accelerator, workload = _parse_workload_and_accelerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = GenericMappingGenerator(accelerator, workload, tmpdir)
        paths, sub_workloads = gen.generate_all_groups()

        assert len(paths) >= 1, "Expected at least one group mapping YAML"

        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert "layers" in data, "Missing 'layers' key"
            assert len(data["layers"]) > 0, "Empty layers list"
            assert "fused_groups" in data, "Missing 'fused_groups' key"

            for layer in data["layers"]:
                assert "name" in layer, "Layer missing 'name'"
                assert "core_allocation" in layer, f"Layer {layer['name']} missing 'core_allocation'"
                assert isinstance(layer["core_allocation"], list), "core_allocation not a list"
                assert all(isinstance(slot, list) for slot in layer["core_allocation"]), (
                    "core_allocation entries must be lists (nested format)"
                )


def test_single_fused_group():
    """Generated mapping contains exactly one FusedGroup with valid intra_core_tiling."""
    accelerator, workload = _parse_workload_and_accelerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = GenericMappingGenerator(accelerator, workload, tmpdir)
        paths, sub_workloads = gen.generate_all_groups()

        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert len(data["fused_groups"]) == 1, f"Expected 1 fused group, got {len(data['fused_groups'])}"
            fg = data["fused_groups"][0]
            assert "layers" in fg and len(fg["layers"]) > 0, "FusedGroup has empty layers"
            assert "intra_core_tiling" in fg and len(fg["intra_core_tiling"]) > 0, (
                "FusedGroup has empty intra_core_tiling"
            )


def test_mapping_validates():
    """Generated mapping passes MappingValidator validation."""
    accelerator, workload = _parse_workload_and_accelerator()

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = GenericMappingGenerator(accelerator, workload, tmpdir)
        paths, sub_workloads = gen.generate_all_groups()

        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            validator = MappingValidator(data)
            is_valid = validator.validate()
            assert is_valid, f"MappingValidator errors: {validator.errors}"


def test_pipeline_end_to_end():
    """A generated mapping of the two-conv workload solves end-to-end on the TPU-like hardware.

    Two-conv is used as a fast single-group proxy (< 60s) for quick iteration.
    test_pipeline_multi_group covers multi-group pipeline mechanics.
    """
    workload_path = make_2_conv_workload(_WORKLOAD_CONFIG)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = evaluate_mapping(_ACCELERATOR, workload_path, tmpdir).context

    allocation: Allocation = ctx.get("allocation")
    assert allocation is not None, "No allocation in context"
    latency = allocation.solution.latency.total
    assert latency > 0, f"Expected positive latency, got {latency}"
    assert allocation.problem.iterations > 0, f"Expected positive iterations, got {allocation.problem.iterations}"

    # Verify total_latency was set by FusionGroupIterationStage
    total_latency = ctx.get("total_latency")
    assert total_latency is not None, "total_latency not set in context"
    assert total_latency > 0, f"Expected positive total_latency, got {total_latency}"

    # Verify per-group latencies are tracked
    group_latencies = ctx.get("group_latencies")
    assert group_latencies is not None, "group_latencies not set in context"
    assert len(group_latencies) >= 1, f"Expected at least 1 group, got {len(group_latencies)}"
    assert all(lat > 0 for lat in group_latencies.values()), "All group latencies must be positive"


@pytest.mark.slow  # multi-group MILP run; too slow / near-timeout on CI runners
@pytest.mark.timeout(120)
def test_pipeline_multi_group():
    """A generated mapping of a multi-group workload solves end-to-end.

    Uses a synthetic Conv->Relu->Flatten->Gemm workload cut after its Relu into 2 fusion groups. Tests the same
    multi-group pipeline mechanics as ResNet18 (cut points, per-group iteration, latency aggregation) in seconds.
    """
    workload_path = make_conv_relu_flatten_gemm_workload()
    options = SolveOptions(stage_options={"fusion_cut_points": ["Relu1"]})

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = evaluate_mapping(_ACCELERATOR, workload_path, tmpdir, options=options).context

    allocation: Allocation = ctx.get("allocation")
    assert allocation is not None, "No allocation in context"
    latency = allocation.solution.latency.total
    assert latency > 0, f"Expected positive latency, got {latency}"

    # Verify total_latency aggregated across all groups
    total_latency = ctx.get("total_latency")
    assert total_latency is not None, "total_latency not set in context"
    assert total_latency > 0, f"Expected positive total_latency, got {total_latency}"

    # Verify per-group latencies (positive latency; multi-group handling)
    group_latencies = ctx.get("group_latencies")
    assert group_latencies is not None, "group_latencies not set in context"
    assert len(group_latencies) == 2, f"Expected 2 groups (Conv+Relu and Gemm), got {len(group_latencies)}"
    assert all(lat > 0 for lat in group_latencies.values()), (
        f"All group latencies must be positive, got {group_latencies}"
    )
    # Verify aggregation: total = sum of per-group
    assert abs(total_latency - sum(group_latencies.values())) < 1e-6, (
        f"total_latency ({total_latency}) != sum of group_latencies ({sum(group_latencies.values())})"
    )


def test_pool_ops_saturate_the_vpus_on_tpu_v7():
    """A pooling op has no matmul owner, so on TPU7x it must run on the four VPUs -- the vector unit
    that already owns the reductions -- fully split across all four, not smeared across the 36-core
    MXU+VPU fallback set (an MXU cannot price a pool). Guards the tpu_v7 VPU operator_types routing:
    without the pooling ops in that list, the pool falls back to every compute core and, worse, lands
    partly on the systolic arrays."""
    onnx_path = make_resnet_subgraph(ResNetSubgraphConfig(pattern=ResNetPattern.FRONTEND))
    accelerator, workload = _parse_workload_and_accelerator(onnx_path, accelerator=_TPU_V7)

    vpu_ids = {c.id for c in accelerator.core_list if getattr(c, "name", "") == "tpu_v7_vpu"}
    assert len(vpu_ids) == 4, f"Expected 4 VPUs on TPU7x, got {sorted(vpu_ids)}"

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = GenericMappingGenerator(accelerator, workload, tmpdir)
        paths, _ = gen.generate_all_groups()

        pool_layers = []
        for path in paths:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert MappingValidator(data).validate(), "Generated tpu_v7 mapping failed validation"
            pool_layers += [layer for layer in data["layers"] if "Pool" in layer["name"]]

        assert pool_layers, "No pooling layer found in the frontend subgraph"
        for layer in pool_layers:
            core_ids = {core for slot in layer["core_allocation"] for core in slot}
            assert core_ids == vpu_ids, (
                f"{layer['name']} must run on all 4 VPUs {sorted(vpu_ids)}, got {sorted(core_ids)}"
            )
            # And it must be split across every one of them, not parked on a subset.
            split = layer["inter_core_tiling"][0]
            cores_used = 1
            for entry in split:
                cores_used *= entry["split"]
            assert cores_used == len(vpu_ids), (
                f"{layer['name']} inter-core split {split} uses {cores_used} cores, expected {len(vpu_ids)}"
            )


def test_fusemax_runs_convs_on_its_array_and_the_rest_on_its_vector_unit():
    """FuseMax's 256x1 vector unit serves the elementwise and pooling ops; the 256x256 array takes the convs."""
    onnx_path = make_resnet_subgraph(ResNetSubgraphConfig(pattern=ResNetPattern.FRONTEND))
    accelerator, workload = _parse_workload_and_accelerator(onnx_path, accelerator=_FUSEMAX)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths, _ = GenericMappingGenerator(accelerator, workload, tmpdir).generate_all_groups()
        layers = []
        for path in paths:
            with open(path) as f:
                layers += yaml.safe_load(f)["layers"]

    cores = {layer["name"]: {core for slot in layer["core_allocation"] for core in slot} for layer in layers}
    assert {name: ids for name, ids in cores.items() if "Conv" in name} == {
        name: {0} for name in cores if "Conv" in name
    }
    assert all(ids == {1} for name, ids in cores.items() if "Conv" not in name)


@pytest.mark.parametrize(
    ("units", "kept"),
    [
        ([65536, 256], [65536]),
        ([1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024]),
        ([1024, 1024, 1024, 600], [1024, 1024, 1024]),
        ([1024, 1024, 900], [1024, 1024, 900]),
    ],
)
def test_an_even_split_keeps_the_cores_that_finish_it_first(units, kept):
    """An even split runs at the pace of its smallest array, so a small core joins only when it adds MACs."""
    cores = [SimpleNamespace(operational_array=SimpleNamespace(total_unit_count=u)) for u in units]
    chosen = GenericMappingGenerator._fastest_even_split(cores)
    assert [core.operational_array.total_unit_count for core in chosen] == kept


def test_cores_without_an_operational_array_split_a_node_evenly():
    """A core that declares no MAC array, such as an AIE tile, gives no ground to prefer one over another."""
    cores = [SimpleNamespace(), SimpleNamespace()]
    assert GenericMappingGenerator._fastest_even_split(cores) == cores


def test_a_mapping_cannot_place_an_operator_on_a_core_that_does_not_execute_it():
    """A conv pinned to FuseMax's vector unit is rejected when the mapping is read, not priced."""
    onnx_path = make_resnet_subgraph(ResNetSubgraphConfig(pattern=ResNetPattern.FRONTEND))
    accelerator, workload = _parse_workload_and_accelerator(onnx_path, accelerator=_FUSEMAX)
    layers = [{"name": node.type, "core_allocation": [[1]]} for node in workload.get_computation_nodes()]

    with pytest.raises(ValueError, match="only executes"):
        MappingFactory({"layers": layers}, workload, accelerator).create()


@pytest.mark.parametrize("mapping", ["swiglu_tpu_v7_fused", "swiglu_tpu_v7_layer_by_layer"])
def test_swiglu_example_mappings_solve_on_tpu_v7(mapping, tmp_path):
    """Both SwiGLU example mappings for TPU7x allocate, every fused group included."""
    estimate = evaluate_mapping(
        _TPU_V7,
        "stream/inputs/examples/workload/swiglu_256_4096_14336.onnx",
        str(tmp_path),
        f"stream/inputs/examples/mapping/{mapping}.yaml",
        SolveOptions(artifacts=False),
    )
    assert estimate.group_cycles and all(cycles > 0 for cycles in estimate.group_cycles)


def _save(nodes, inputs, output, path, value_info=()):
    graph = helper.make_graph(nodes, "graph", inputs, [output], value_info=list(value_info))
    opsets = [helper.make_opsetid("", 19), helper.make_opsetid("com.example", 1)]
    onnx.save(helper.make_model(graph, opset_imports=opsets), path)
    return str(path)


def _flash_attention(path):
    """Attention as IRON's flash workload exports it: the scores, one online-softmax step per block, the context."""
    q = helper.make_tensor_value_info("q", TensorProto.BFLOAT16, [256, 64])
    k_t = helper.make_tensor_value_info("k_t", TensorProto.BFLOAT16, [64, 256])
    v = helper.make_tensor_value_info("v", TensorProto.BFLOAT16, [256, 64])
    out = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [256, 64])
    scores = [helper.make_tensor_value_info(n, TensorProto.BFLOAT16, [256, 256]) for n in ("scores", "probs")]
    nodes = [
        helper.make_node("MatMul", ["q", "k_t"], ["scores"], name="Attn_Scores"),
        helper.make_node("PartialSoftmax", ["scores"], ["probs"], name="Attn_Softmax", domain="com.example"),
        helper.make_node("MatMul", ["probs", "v"], ["out"], name="Attn_Context"),
    ]
    return _save(nodes, [q, k_t, v], out, path, scores)


def _one_op(op_type, path, **attributes):
    x = helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [64, 256])
    out = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [64, 256])
    return _save([helper.make_node(op_type, ["x"], ["out"], name=op_type, **attributes)], [x], out, path)


def _placement(accelerator, workload):
    with tempfile.TemporaryDirectory() as tmpdir:
        paths, _ = GenericMappingGenerator(accelerator, workload, tmpdir).generate_all_groups()
        layers = [layer for path in paths for layer in yaml.safe_load(Path(path).read_text())["layers"]]
    return {layer["name"]: {core for slot in layer["core_allocation"] for core in slot} for layer in layers}


def _vpus(accelerator):
    return [core for core in accelerator.core_list if getattr(core, "name", "") == "tpu_v7_vpu"]


def test_a_core_that_runs_softmax_runs_its_online_step(tmp_path):
    path = _flash_attention(tmp_path / "flash.onnx")
    accelerator, workload = _parse_workload_and_accelerator(path, accelerator=_TPU_V7)
    assert _placement(accelerator, workload)["Attn_Softmax"] == {core.id for core in _vpus(accelerator)}
    estimate = evaluate_mapping(_TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    assert estimate.cycles > 0


def _ideal_cycles(path, tmp_path):
    estimate = evaluate_mapping(_TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    lut = estimate.context.get("cost_lut").lut
    return {node.type: entry.ideal_cycle for node, costs in lut.items() for entry in costs.values()}


def test_partial_softmax_costs_its_core_the_cycles_of_the_softmax_it_is_part_of(tmp_path):
    part = _ideal_cycles(_one_op("PartialSoftmax", tmp_path / "part.onnx", domain="com.example"), tmp_path / "part")
    whole = _ideal_cycles(_one_op("Softmax", tmp_path / "softmax.onnx", axis=1), tmp_path / "softmax")
    assert set(whole) == {"ReduceMax", "Exp", "ReduceSum", "Div"}
    assert part["PartialSoftmax"] == sum(whole.values())


def _fixed_costs(path, cores, tmp_path):
    layers = [{"name": name, "core_allocation": [ids], "inter_core_tiling": []} for name, ids in cores.items()]
    mapping = {"layers": layers, "fused_groups": [{"name": "all", "layers": list(cores), "intra_core_tiling": []}]}
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(mapping))
    options = SolveOptions(artifacts=False)
    estimate = evaluate_mapping(_TPU_V7, path, str(tmp_path), str(tmp_path / "mapping.yaml"), options)
    lut = estimate.context.get("cost_lut").lut
    return {node.type: entry for node, costs in lut.items() for entry in costs.values()}


def _one_op_on_a_vpu(op_type, tmp_path, **attributes):
    (tmp_path / op_type).mkdir()
    path = _one_op(op_type, tmp_path / op_type / "model.onnx", **attributes)
    return _fixed_costs(path, {op_type: [2]}, tmp_path / op_type)[op_type]


def test_a_fixed_mapping_prices_a_softmax_as_the_operators_it_expands_into(tmp_path):
    """Max, exp at four cycles, sum and divide: seven cycles and four operations per element of a Relu's one."""
    relu = _one_op_on_a_vpu("Relu", tmp_path)
    softmax = _one_op_on_a_vpu("Softmax", tmp_path, axis=1)
    partial = _one_op_on_a_vpu("PartialSoftmax", tmp_path, domain="com.example")
    for priced in (softmax, partial):
        assert priced.ideal_cycle == 7 * relu.ideal_cycle
        assert priced.cme.mac_energy == pytest.approx(4 * relu.cme.mac_energy)


def test_an_operator_expanded_over_fewer_points_costs_only_those(tmp_path):
    """An L2 normalization's square root runs once per row of 256 elements, not on every element."""
    relu = _one_op_on_a_vpu("Relu", tmp_path)
    norm = _one_op_on_a_vpu("LpNormalization", tmp_path, axis=1)
    assert norm.cme.mac_energy == pytest.approx((2 + 1 / 256) * relu.cme.mac_energy)


def test_a_fixed_mapping_places_partial_softmax_on_a_core_that_lists_softmax(tmp_path):
    cores = {"Attn_Scores": [0], "Attn_Softmax": [2], "Attn_Context": [0]}
    assert _fixed_costs(_flash_attention(tmp_path / "flash.onnx"), cores, tmp_path)["PartialSoftmax"].ideal_cycle > 0


def test_a_core_that_runs_softmax_runs_the_operators_it_is_expanded_into(tmp_path):
    path = _one_op("Softmax", tmp_path / "softmax.onnx", axis=1)
    accelerator, workload = _parse_workload_and_accelerator(path, accelerator=_TPU_V7)
    for vpu in _vpus(accelerator):
        vpu.operator_types = ["Softmax"]
    placement = _placement(accelerator, expand_normalizations(workload))
    assert len(placement) == 4
    assert all(cores == {core.id for core in _vpus(accelerator)} for cores in placement.values())


def test_a_layer_no_core_runs_is_refused_naming_its_operator(tmp_path):
    accelerator, workload = _parse_workload_and_accelerator(_flash_attention(tmp_path / "flash.onnx"), _TPU_V7)
    for vpu in _vpus(accelerator):
        vpu.operator_types = [op for op in vpu.operator_types if op != "Softmax"]
    served = sorted({op for core in accelerator.core_list for op in getattr(core, "operator_types", None) or ()})
    refusal = f"no core of this design runs PartialSoftmax (Attn_Softmax); its cores run: {', '.join(served)}"
    with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
        _placement(accelerator, workload)


def _layer_norm(path):
    x = helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [64, 256])
    out = helper.make_tensor_value_info("out", TensorProto.BFLOAT16, [64, 256])
    params = [helper.make_tensor(name, TensorProto.BFLOAT16, [256], [0] * 256) for name in ("scale", "bias")]
    node = helper.make_node("LayerNormalization", ["x", "scale", "bias"], ["out"], name="LayerNorm", axis=1)
    graph = helper.make_graph([node], "graph", [x], [out], params)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), path)
    return str(path)


def test_a_layer_normalization_reads_its_scale_and_shift(tmp_path):
    path = _layer_norm(tmp_path / "layernorm.onnx")
    estimate = evaluate_mapping(_TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    read = {t.name for node in estimate.context.get("cost_lut").lut for t in node.inputs}
    assert {"scale", "bias"} <= read and estimate.cycles > 0


def test_a_whole_layer_normalization_on_a_vector_core_is_priced_by_its_cost_model(tmp_path):
    """Its scale and shift are read with the output, so the core's two input operands suffice."""
    entry = _fixed_costs(_layer_norm(tmp_path / "layernorm.onnx"), {"LayerNorm": [2]}, tmp_path)["LayerNormalization"]
    assert entry.metadata.get("backend") != IDEAL_CYCLE_BACKEND
