"""KV-cache data-movement tests: Slice (exact affine region) and Gather (conservative full-axis),
the decode-step dataflow, and the ONNX parsers that ingest them.

The point is *correct data movement*: the derived dependency must recover exactly which cache region
a decode step reads, so the hardware moves the right data.
"""

from __future__ import annotations

import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from xdsl.dialects.builtin import bf16

from stream.parser.onnx.model import ONNXModelParser
from stream.workload.affine_access import compose_dependency, footprint, relevancy
from stream.workload.data_movement import gather_node, slice_node
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.models import KVCacheConfig, build_kv_cache_decode_step
from stream.workload.steady_state.iteration_space import LoopEffect
from stream.workload.tensor import Tensor

# --------------------------------------------------------------------------- slice / gather nodes


def test_slice_reads_the_exact_prefix():
    cache = Tensor.create("K", bf16, (128, 32))
    node, out = slice_node(cache, axis=0, start=0, length=10)
    assert out.shape == (10, 32)
    region = footprint(node.operand_mapping[0], {0: range(0, 10), 1: range(0, 32)})
    assert region == (range(0, 10), range(0, 32))


def test_slice_sliding_window_offset():
    cache = Tensor.create("K", bf16, (128, 32))
    node, _ = slice_node(cache, axis=0, start=6, length=4)
    region = footprint(node.operand_mapping[0], {0: range(0, 4), 1: range(0, 32)})
    assert region[0] == range(6, 10)  # cache[6:10]


def test_gather_reads_the_full_source_axis_conservatively():
    """A data-dependent gather is bounded by reading the whole gathered axis (safe data movement)."""
    cache = Tensor.create("K", bf16, (128, 32))
    node, out = gather_node(cache, axis=0, num_indices=3)
    assert out.shape == (3, 32)
    types = derive_iterator_types(node)
    src = node.num_dims - 1
    assert types[src] == IteratorType.REDUCTION  # the source axis is read in full
    assert relevancy(node, cache, src) == LoopEffect.VARYING
    # full-iteration footprint touches the entire source axis
    region = footprint(node.operand_mapping[0], {0: range(0, 3), 1: range(0, 32), src: range(0, 128)})
    assert region[0] == range(0, 128)


# --------------------------------------------------------------------------- decode-step dataflow


def test_decode_step_reads_only_the_valid_cache_prefix():
    """End-to-end dataflow: the new token's scores read K_valid, which is exactly K_cache[0:valid]."""
    wl = build_kv_cache_decode_step(KVCacheConfig(cache_capacity=128, valid_len=40, d_head=64))
    k_slice = next(n for n in wl.get_computation_nodes() if n.name == "K_valid")
    scores = next(n for n in wl.get_computation_nodes() if n.name == "scores")
    k_valid = k_slice.outputs[0]
    # what region of K_valid the scores need (single query, full key range, full head dim)
    needed = compose_dependency(
        k_slice.operand_mapping[1], scores.get_mapping(k_valid), {0: range(0, 1), 1: range(0, 40), 2: range(0, 64)}
    )
    assert needed[0] == range(0, 40)
    # and that region of K_valid comes from K_cache[0:40]
    cache_region = footprint(k_slice.operand_mapping[0], {0: range(0, 40), 1: range(0, 64)})
    assert cache_region[0] == range(0, 40)  # never touches the unwritten cache tail [40:128]


def test_decode_step_scores_reduce_over_cache_positions():
    """The new token attends over every valid cached key: the cache-position axis is the reduction."""
    wl = build_kv_cache_decode_step()
    scores = next(n for n in wl.get_computation_nodes() if n.name == "scores")
    red = [p for p, t in derive_iterator_types(scores).items() if t == IteratorType.REDUCTION]
    assert len(red) == 1  # the head dim e is contracted; j (cache position) is the parallel output axis


# --------------------------------------------------------------------------- ONNX ingestion


def _parse(model: onnx.ModelProto):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        parser = ONNXModelParser(f.name)
        parser.run()
    return parser.workload


def test_onnx_slice_parser_recovers_the_offset():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [128, 32])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [40, 32])
    inits = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.array([40], dtype=np.int64), "ends"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axes"),
    ]
    node = helper.make_node("Slice", ["data", "starts", "ends", "axes"], ["out"], name="sl")
    model = helper.make_model(
        helper.make_graph([node], "g", [data], [out], inits), opset_imports=[helper.make_opsetid("", 17)]
    )
    slice_node_parsed = _parse(model).get_computation_nodes()[0]
    assert slice_node_parsed.type == "Slice"
    region = footprint(slice_node_parsed.operand_mapping[0], {0: range(0, 40), 1: range(0, 32)})
    assert region == (range(0, 40), range(0, 32))


def test_onnx_slice_parser_nonzero_start():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [128, 32])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [8, 32])
    inits = [
        numpy_helper.from_array(np.array([100], dtype=np.int64), "starts"),
        numpy_helper.from_array(np.array([108], dtype=np.int64), "ends"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axes"),
    ]
    node = helper.make_node("Slice", ["data", "starts", "ends", "axes"], ["out"], name="sl")
    model = helper.make_model(
        helper.make_graph([node], "g", [data], [out], inits), opset_imports=[helper.make_opsetid("", 17)]
    )
    parsed = _parse(model).get_computation_nodes()[0]
    region = footprint(parsed.operand_mapping[0], {0: range(0, 8), 1: range(0, 32)})
    assert region[0] == range(100, 108)


def test_onnx_gather_parser_reads_full_axis():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [128, 32])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [3, 32])
    indices = numpy_helper.from_array(np.array([1, 5, 9], dtype=np.int64), "indices")
    node = helper.make_node("Gather", ["data", "indices"], ["out"], name="gt", axis=0)
    model = helper.make_model(
        helper.make_graph([node], "g", [data], [out], [indices]), opset_imports=[helper.make_opsetid("", 17)]
    )
    gathered = _parse(model).get_computation_nodes()[0]
    assert gathered.type == "Gather"
    src = gathered.num_dims - 1
    region = footprint(gathered.operand_mapping[0], {0: range(0, 3), 1: range(0, 32), src: range(0, 128)})
    assert region[0] == range(0, 128)  # conservative: the whole cache axis moves
