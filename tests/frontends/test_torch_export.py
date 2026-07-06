"""Tests for the torch.export frontend (plan/08, plan Phase 3).

The conversion is torch-free and tested with hand-built ATen calls (structural equivalence to the
IR twin); torch itself is optional, so ``can_load`` declines without it and an e2e export test is
skipped. This mirrors the islpy-present/absent split.
"""

from __future__ import annotations

import pytest
from xdsl.ir.affine import AffineMap

from stream.frontends import available_frontends, frontend_for
from stream.frontends.torch_export import (
    AtenCall,
    AtenTensor,
    TorchExportFrontend,
    convert_aten_calls,
    register_aten_op,
)
from stream.workload.fusion.proposer import propose_fusion_regions
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode, NormalizationNode


def _mlp_calls():
    graph_inputs = [AtenTensor("x", (4, 8)), AtenTensor("W1", (16, 8)), AtenTensor("W2", (8, 16))]
    calls = [
        AtenCall("aten::linear", ("x", "W1"), AtenTensor("h", (4, 16))),
        AtenCall("aten::silu", ("h",), AtenTensor("act", (4, 16))),
        AtenCall("aten::linear", ("act", "W2"), AtenTensor("y", (4, 8))),
    ]
    return graph_inputs, calls


def test_conversion_is_a_valid_affine_workload_structurally_like_the_ir_twin():
    workload, report = convert_aten_calls(*_mlp_calls())
    assert report.is_complete
    nodes = workload.get_computation_nodes()
    assert [n.type for n in nodes] == ["MatMul", "aten::silu", "MatMul"]  # op provenance preserved
    # the linears contract k (a REDUCTION); the elementwise activation is pure PARALLEL
    assert IteratorType.REDUCTION in derive_iterator_types(nodes[0]).values()
    assert set(derive_iterator_types(nodes[1]).values()) == {IteratorType.PARALLEL}
    # and the affine structure fuses as one region (same as the SwiGLU/MLP IR twin)
    assert len(propose_fusion_regions(workload, 2**60)) == 1


def test_softmax_lowers_to_a_fusible_normalization_node():
    graph_inputs = [AtenTensor("scores", (2, 4, 4))]
    calls = [AtenCall("aten::softmax", ("scores",), AtenTensor("probs", (2, 4, 4)), attrs={"dim": -1})]
    workload, report = convert_aten_calls(graph_inputs, calls)
    assert report.is_complete
    softmax = workload.get_computation_nodes()[0]
    assert isinstance(softmax, NormalizationNode)
    assert softmax.reduction_axes == (2,)  # the last axis


def test_unsupported_op_is_reported_not_fatal():
    graph_inputs = [AtenTensor("x", (4, 8))]
    calls = [AtenCall("aten::scaled_dot_product_attention", ("x",), AtenTensor("y", (4, 8)))]
    workload, report = convert_aten_calls(graph_inputs, calls)
    assert not report.is_complete
    assert report.counts["aten::scaled_dot_product_attention"] == 1
    assert workload.number_of_nodes() > 0  # did not crash


def test_overlay_can_extend_the_op_table():
    """The moat seam: a plugin registers coverage for an op the public table lacks."""

    def build_custom(inputs, output, call):
        rank = len(output.shape)
        maps = tuple(AffineMap.identity(rank) for _ in range(len(inputs) + 1))
        return ComputationNode(
            type="Custom", name=call.output.name, inputs=inputs, outputs=(output,), operand_mapping=maps
        )

    register_aten_op("aten::my_custom_op", build_custom)
    graph_inputs = [AtenTensor("x", (4, 8))]
    calls = [AtenCall("aten::my_custom_op", ("x",), AtenTensor("y", (4, 8)))]
    _, report = convert_aten_calls(graph_inputs, calls)
    assert report.is_complete


def test_torch_frontend_registered_but_declines_without_torch():
    assert any(f.name == "torch_export" for f in available_frontends())
    # torch is not installed in the base env: the frontend must decline an arbitrary source
    assert TorchExportFrontend().can_load(object()) is False


def test_torch_export_end_to_end():
    torch = pytest.importorskip("torch")  # skipped unless the [torch] extra is installed

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(8, 16)
            self.fc2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            return self.fc2(torch.nn.functional.silu(self.fc1(x)))

    exported = torch.export.export(MLP(), (torch.randn(4, 8),))
    frontend = frontend_for(exported)
    assert frontend.name == "torch_export"
    workload = frontend.load(exported)
    assert any(n.type == "MatMul" for n in workload.get_computation_nodes())
