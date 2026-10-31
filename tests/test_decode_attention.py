"""Decode attention, plain and flash-decoding, computes attention over the KV cache and runs on TPU7x as a stream of
the cache from HBM: the contractions on the MXUs, the softmax and the combine on the VPUs."""

import numpy as np
import pytest
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator

from stream.api import SolveOptions, evaluate_mapping
from stream.cost_model.roofline import OFFCHIP_LINKS, OFFCHIP_MEMORY, group_point, hardware_roofline
from stream.inputs.testing.workload.make_decode_attention import make_decode_attention, make_flash_decoding

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
MXU_TYPES = {"Einsum"}
HBM_BITS_PER_CYCLE = 53600


def _attention(q, k, v):
    scores = np.einsum("bgrd,bgsd->bgrs", q, k)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    return np.einsum("bgrs,bgsd->bgrd", probs / probs.sum(-1, keepdims=True), v)


def test_both_graphs_compute_attention_over_the_cache(tmp_path):
    batch, context, parts = 2, 64, 4
    rng = np.random.default_rng(0)
    q = rng.standard_normal((batch, 8, 4, 128), dtype=np.float32)
    k, v = (rng.standard_normal((batch, 8, context, 128), dtype=np.float32) for _ in range(2))
    expected = _attention(q, k, v)
    plain = make_decode_attention(tmp_path / "plain.onnx", batch, context, dtype=TensorProto.FLOAT)
    flash = make_flash_decoding(tmp_path / "flash.onnx", batch, context, parts, dtype=TensorProto.FLOAT)
    split = (batch, 8, parts, context // parts, 128)
    (out,) = ReferenceEvaluator(plain).run(None, {"q": q, "K": k, "V": v})
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
    (out,) = ReferenceEvaluator(flash).run(None, {"q": q, "K": k.reshape(split), "V": v.reshape(split)})
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("make", [make_decode_attention, make_flash_decoding])
@pytest.mark.parametrize("batch", [1, 16])
def test_decode_streams_the_cache_from_hbm(tmp_path, make, batch):
    path = make(tmp_path / "decode.onnx", batch, 256)
    estimate = evaluate_mapping(TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    allocation = estimate.context.get("allocation")
    problem = allocation.problem
    for node in problem.workload.get_computation_nodes():
        cores = {core for cores in problem.mapping.get(node).resource_allocation for core in cores}
        array = "mxu" if node.type in MXU_TYPES else "vpu"
        assert {core.name for core in cores} == {f"tpu_v7_{array}"}, node.name
    point = group_point(allocation, hardware_roofline(problem.accelerator))
    assert point["bound"] in (OFFCHIP_MEMORY, OFFCHIP_LINKS)
    cache_bytes = 2 * batch * 8 * 256 * 128 * 2
    offchip = next(level for level in point["levels"] if level["name"] == OFFCHIP_MEMORY)
    assert offchip["bytes"] >= cache_bytes


def test_plain_decode_over_a_long_context_streams_the_cache_at_the_hbm_rate(tmp_path):
    path = make_decode_attention(tmp_path / "decode.onnx", 1, 32768)
    estimate = evaluate_mapping(TPU_V7, path, str(tmp_path), options=SolveOptions(artifacts=False))
    cache_bits = 2 * 8 * 32768 * 128 * 16
    assert estimate.cycles < 1.3 * cache_bits / HBM_BITS_PER_CYCLE
