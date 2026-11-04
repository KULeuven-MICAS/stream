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
from onnx import TensorProto, helper

from stream.api import SolveOptions, evaluate_mapping
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.inputs.testing.workload.make_attention_head import make_multi_head_attention
from stream.inputs.testing.workload.make_decode_attention import make_decode_attention
from stream.mapping.capacity_tiler import CapacityTiler, _divisors_desc
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.mapping.mapping import whole_bits
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


def _parse(hardware: str, workload: str):
    ctx = StageContext.from_kwargs(accelerator=hardware, workload_path=workload, output_path=tempfile.mkdtemp())
    ctxs = MainStage([AcceleratorParserStage, ONNXModelParserStage, LeafStage], ctx).run()
    return ctxs[0].get("accelerator"), ctxs[0].get("workload")


def _worst_core_ratio(gen: GenericMappingGenerator, sub, cns, tiling) -> float:
    """Worst memory footprint / (capacity * fill) the tiler measures for ``tiling``."""
    tiler = CapacityTiler(sub, gen.accelerator)
    cores = {cn: gen._select_cores_for_node(cn) for cn in cns}
    unroll = gen._inter_core_unrolling(sub, cns)
    dims = {sub.leading_dim(d)[0] for cn in cns for d in sub.get_dims(cn)}
    per_core = {d: sub.get_dimension_size(d) // unroll.get(d, 1) for d in dims}
    resident = per_core | tiler._seed_resident(cns, tiling, per_core)
    held = tiler.memory_bits(cns, cores, unroll, resident)
    capacities = tiler._capacities(cns, cores)
    return max(bits / (capacities[m] * tiler.fill_fraction) for m, bits in held.items())


def test_divisors_desc():
    assert _divisors_desc(12) == [12, 6, 4, 3, 2, 1]
    assert _divisors_desc(1) == [1]
    assert _divisors_desc(14336)[0] == 14336  # sqrt enumeration returns the whole dim first


def test_streams_output_axes_when_weight_overflows():
    """A large Gemm whose resident weight overflows streams its output axes until it fits, keeping the contraction
    whole so no partial sum has to stay resident."""
    acc, w = _parse(_TPU_QUAD, _GEMM)
    gen = GenericMappingGenerator(acc, w, tempfile.mkdtemp())
    subs = w.split_fusion_groups(cut_points=gen._cut_points(None))
    refined_any = False
    for sub in subs:
        cns = tuple(sub.get_computation_nodes())
        if not cns:
            continue
        seed = gen._auto_fusion_tiling(sub, cns) or gen._whole_layer_tiling(sub, cns)
        refined = gen._capacity_refine(sub, cns, seed)
        if _worst_core_ratio(gen, sub, cns, seed) > 1.0:
            refined_any = True
            # the trivial mapper overflowed; the refined tiling must fit and must tile a contraction axis
            assert _worst_core_ratio(gen, sub, cns, refined) <= 1.0
            assert refined and not any(".D1" in e["dim"] for e in refined), f"expected output-axis tiles, got {refined}"
    assert refined_any, "the Gemm was expected to overflow the trivial mapping"


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
    """The overflowing SwiGLU streams two dims, in the order its nodes walk them, whatever order a set iterates in."""
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
    assert plans == {str([{"dim": "Gemm_Left.D0", "tile": 8}, {"dim": "Gemm_Left.D2", "tile": 128}])}


def test_a_node_left_on_one_core_holds_its_tensors_whole():
    """Split over four cores, each Gemm core streams 32 of its 512 output columns per iteration; left on one core it
    holds the columns of all four, so the group's per-core tile shrinks to 8 for the same 32 columns there."""
    acc, w = _parse(_TPU_QUAD, _GEMM)
    (cn,) = w.get_computation_nodes()
    unroll = {w.get_dims(cn)[2]: 4}
    cores = [c for c in acc.core_list if c.id in (0, 1, 2, 3)]
    tiler = CapacityTiler(w, acc)
    assert tiler.plan((cn,), {cn: cores}, unroll, set()) == [
        {"dim": "Gemm.D0", "tile": 16},
        {"dim": "Gemm.D2", "tile": 32},
    ]
    assert tiler.plan((cn,), {cn: cores[:1]}, unroll, set()) == [
        {"dim": "Gemm.D0", "tile": 16},
        {"dim": "Gemm.D2", "tile": 8},
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


def test_a_group_no_tiling_fits_is_cut_where_the_operators_before_stop_fitting(tmp_path):
    """A 4096-wide attention fused on four Eyeriss-like cores cannot hold its projections' weights next to the softmax
    of every head: it is cut after the longest runs that fit, two projections, the third, then the attention with
    all the sub-ops its softmax expands into."""
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
    assert operators == [{"proj_K", "proj_Q"}, {"proj_V"}, {"scores", "softmax", "context"}]
    assert all(gen._fits(group) for group in groups)


@pytest.mark.parametrize(
    ("hardware", "workload", "node", "layer_by_layer", "innermost"),
    [
        # The key projection alone: its 1 MiB of outputs per 256 KiB chiplet cannot stay resident over its 4096-wide
        # contraction
        ("simba_small", "mha", "proj_K", True, "proj_K.D3"),
        # The fused SwiGLU's down projection keeps its outputs while the d_ff its weights stream along goes past them
        ("eyeriss_like_quad_core", _SWIGLU_FITS, "Gemm_Down", False, "Gemm_Left.D0"),
    ],
)
def test_a_reduction_runs_innermost_only_where_its_outputs_cannot_stay_while_it_accumulates(
    tmp_path, hardware, workload, node, layer_by_layer, innermost
):
    path = make_multi_head_attention(tmp_path / "mha.onnx", seq=512) if workload == "mha" else workload
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
operand_precision: {input: int8, accumulator: int32}
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
    """The chip and an int8 ``m x k`` by ``k x n`` Gemm on it: one packed core with an L1 of ``l1_kib``, reaching the
    off-chip memory through the SRAM of ``sram_kib`` of a memory core, over a link of ``link`` bits a cycle, or directly
    where there is none."""
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
        ((256, 1024, 4096, 32, 96), None, ["Gemm.D1", "Gemm.D2", "Gemm.D0"]),
        # Rows innermost keep the 128 x 128 partial sums in the L1, so the 2 MiB of weights are read once.
        ((128, 2048, 1024, 64, 0), None, ["Gemm.D0", "Gemm.D1", "Gemm.D2"]),
        # On one column the SRAM's memory core is not the allocation's, so the default order stays.
        ((256, 1024, 4096, 32, 96), 1, ["Gemm.D1", "Gemm.D0", "Gemm.D2"]),
        # The tiles passing through the 32 KiB SRAM fill it, so it holds no block of rows: the default order stays.
        ((128, 512, 8192, 64, 32), None, ["Gemm.D0", "Gemm.D1", "Gemm.D2"]),
        # Held in the SRAM, a would reach the L1 once per column tile over a link no faster than off chip.
        ((64, 512, 1024, 16, 32), 2, ["Gemm.D0", "Gemm.D1", "Gemm.D2"]),
    ],
    ids=["rows-outermost", "rows-innermost", "one-column", "staging-full", "l1-takes-more"],
)
def test_a_tile_loop_order_is_the_one_moving_the_least_data(tmp_path, shape, nb_cols_to_use, order):
    assert _order(tmp_path, *shape, nb_cols_to_use) == order


def test_a_wide_link_from_the_staging_memory_takes_the_rows_outermost(tmp_path):
    """The last case above with the SRAM's link to the L1 16 times as wide: the L1 takes a again per column tile in
    less time than a is read from off chip again, so the rows go outermost."""
    assert _order(tmp_path, 64, 512, 1024, 16, 32, nb_cols_to_use=2, link=4096) == ["Gemm.D1", "Gemm.D2", "Gemm.D0"]


def test_a_memory_behind_an_on_chip_buffer_takes_no_more_than_before(tmp_path):
    """The last case above with a buffer no core uses between the SRAM and the L1, over a link of 64 bits a cycle: the
    L1 has no link from off chip or a staging memory to time it by, so it may take no more than in the default order,
    which stays."""
    chip, gemm = _gemm(tmp_path, 64, 512, 1024, l1_kib=16, sram_kib=32)
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
    assert _order(tmp_path, 256, 1024, 4096, 32, 96) == ["Gemm.D1", "Gemm.D0", "Gemm.D2"]


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
