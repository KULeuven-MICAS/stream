"""The capacity-aware intra-core tiler streams a resident weight on overflow, leaves fitting groups alone."""

import math
import os
import subprocess
import sys
import tempfile

import pytest

from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.mapping.capacity_tiler import CapacityTiler, _divisors_desc
from stream.mapping.generic_generator import GenericMappingGenerator
from stream.parser.mapping_validator import MappingValidator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage

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


def test_a_tiling_over_several_dims_does_not_depend_on_the_hash_seed():
    """The overflowing SwiGLU streams two dims, in the order its nodes walk them, whatever order a set iterates in."""
    workload = make_swiglu_workload(512, 1024, 4096, "bf16", "bf16")
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
