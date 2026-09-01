"""The state a kernel carries, modelled as what it is: an operand resident on its core.

An online softmax keeps its running maximum and sum from one key block to the next. The
graph the design is exported from drops that -- the flash softmax is one opaque node whose
fake is ``empty_like`` -- so the buffer standing in for the reduction was invisible to every
constraint. These pin the three things that make it visible and keep it honest: that it
reads as a recurrence, that it is not moved, and above all that the bytes charged to a core
are the bytes that core holds.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.datatypes import LayerDim
from stream.mapping.mapping import Mapping, NodeMapping
from stream.workload.iterator_type import IteratorType, derive_iterator_types, is_state_operand, sequential_dims
from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

pytest.importorskip("snaxc", reason="the AIE kernels are a separate install, via stream-setup-aie")

from stream.compiler.kernels.aie_kernel import StateOperand  # noqa: E402
from stream.stages.generation.kernel_state import state_tensor  # noqa: E402

QUERY, KEY, ROWS = 128, 256, 4
IDENTITY = AffineMap.from_callable(lambda q, k: (q, k))


def _softmax_with_state():
    scores = Tensor.create("scores", bf16, (QUERY, KEY))
    probs = Tensor.create("probs", bf16, (QUERY, KEY))
    node = ComputationNode(
        type="PartialSoftmax",
        name="Attn_Softmax",
        inputs=(scores,),
        outputs=(probs,),
        operand_mapping=(IDENTITY, IDENTITY),
    )
    state, mapping = state_tensor(node, StateOperand("flash_state", ROWS, carried_over=1, indexed_by=0))
    node = replace(node, inputs=(scores, state), operand_mapping=(IDENTITY, mapping, IDENTITY))
    workload = Workload([InEdge(name="scores_in", outputs=(scores,)), node, OutEdge(name="probs_out", inputs=(probs,))])
    return workload, node, state, probs


def test_the_carried_state_reads_as_a_recurrence():
    """The same predicate the linear-attention state uses, so the key stops looking parallel."""
    _, node, state, _ = _softmax_with_state()
    assert is_state_operand(node, state)
    assert sequential_dims(node) == frozenset({1})
    assert derive_iterator_types(node) == {0: IteratorType.PARALLEL, 1: IteratorType.SEQUENTIAL}


def test_the_carried_state_is_resident_rather_than_produced():
    """Nothing produces it and nothing carries it between nodes, so the graph still sorts.

    An edge here would be the node waiting on itself, which is why the producer lookup skips
    a state operand rather than the workload growing a self-loop.
    """
    workload, node, _, _ = _softmax_with_state()
    assert [n.name for n in workload.dataflow_sort()] == ["scores_in", "Attn_Softmax", "probs_out"]


def _per_core_bytes(workload, node, tensor, split: tuple[LayerDim, int] | None):
    mapping = Mapping({})
    mapping.set(node, NodeMapping(inter_core_tiling=((split,),) if split else ()))
    return workload.get_tensor_single_core(tensor, node, mapping).size_bits() // 8


def test_splitting_the_query_divides_the_state_with_it():
    """Each core keeps the rows it works on, which is what its own buffer holds."""
    workload, node, state, _ = _softmax_with_state()
    query, _ = workload.get_dims(node)
    whole = _per_core_bytes(workload, node, state, None)
    assert whole == ROWS * QUERY * 2
    assert _per_core_bytes(workload, node, state, (query, 4)) == whole // 4
    assert _per_core_bytes(workload, node, state, (query, 16)) == whole // 16


def test_splitting_the_key_leaves_the_state_whole():
    """The key is what the state is carried over. Splitting a carry does not halve it: both
    cores would need all of it, which is the same reason the dimension is SEQUENTIAL."""
    workload, node, state, _ = _softmax_with_state()
    _, key = workload.get_dims(node)
    assert _per_core_bytes(workload, node, state, (key, 4)) == _per_core_bytes(workload, node, state, None)


def test_the_state_keeps_its_declared_rows_whatever_the_key_is():
    """The row extent survives because the reach of ``key - 1`` is clipped to the tensor's own
    shape. That clip is what makes the size right, so it is asserted rather than assumed."""
    workload, node, state, _ = _softmax_with_state()
    assert state.shape == (ROWS, QUERY)
    _, key = workload.get_dims(node)
    mapping = Mapping({})
    mapping.set(node, NodeMapping(inter_core_tiling=(((key, 4),),)))
    assert workload.get_tensor_single_core(state, node, mapping).shape[0] == ROWS
