"""The ``accelforge`` backend without AccelForge: its options and cache identity, the cores it claims, how a mapping
becomes a cost entry, and the ZigZag fallback for a node no Einsum can express."""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.cost_model.core_cost import IDEAL_CYCLE_BACKEND
from stream.hardware.architecture.core import Core
from stream.hardware.ports import OUTPUT, READ_BY_DATAPATH, WRITE_BY_DATAPATH, input_role
from stream.stages.estimation.accelforge import backend as backend_module
from stream.stages.estimation.accelforge.backend import (
    ACCELFORGE_BACKEND,
    MODEL_REVISION,
    AccelForgeEstimator,
    AccelForgeOptions,
    MappedNode,
    port_traffic,
)
from stream.stages.estimation.accelforge.tree import COMPUTE, PJ, zigzag_core
from stream.stages.estimation.accelforge.workload import UnsupportedConversion
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator
from stream.workload.node import ComputationNode, InEdge, OutEdge
from stream.workload.tensor import Tensor
from stream.workload.workload import Workload

from ..cost_stage_runs import inputs
from .cores import CORES, example_core, load_core


@pytest.mark.parametrize(
    "options, message",
    [({"mapper": "ffm"}, "unexpected keyword argument 'mapper'"), ({"memory_order": "any"}, "'free' or 'forced'")],
)
def test_an_option_it_does_not_take_or_a_wrong_value_is_refused(options, message):
    with pytest.raises(ValueError, match=message):
        AccelForgeOptions.parse(options)


def test_the_cache_identity_holds_the_options_and_the_model_revision():
    identity = ACCELFORGE_BACKEND.cache_identity({})
    assert identity == ACCELFORGE_BACKEND.cache_identity({"memory_order": "free", "imperfect_spatial_loops": False})
    assert identity != ACCELFORGE_BACKEND.cache_identity({"memory_order": "forced"})
    assert identity["model_revision"] == MODEL_REVISION


def _fake_core(core_type: str) -> Core:
    core = Core.__new__(Core)
    core.id, core.core_type, core.type = 0, core_type, core_type.rsplit(".", maxsplit=1)[-1]
    return core


def test_it_claims_zigzag_compute_cores_only_when_accelforge_is_installed(monkeypatch):
    monkeypatch.setattr(backend_module, "accelforge_installed", lambda: True)
    assert ACCELFORGE_BACKEND.claims(example_core("tpu_like"))
    assert not ACCELFORGE_BACKEND.claims(example_core("tpu_v7_vmem"))
    assert not ACCELFORGE_BACKEND.claims(_fake_core("aie2.compute"))
    monkeypatch.setattr(backend_module, "accelforge_installed", lambda: False)
    assert not ACCELFORGE_BACKEND.claims(example_core("tpu_like"))


def _gemm(op_type: str = "Gemm") -> ComputationNode:
    a, b, c = Tensor.create("A", bf16, (8, 4)), Tensor.create("B", bf16, (4, 2)), Tensor.create("C", bf16, (8, 2))
    maps = tuple(
        AffineMap.from_callable(fn) for fn in (lambda m, k, n: (m, k), lambda m, k, n: (k, n), lambda m, k, n: (m, n))
    )
    return ComputationNode(type=op_type, name="g", inputs=(a, b), outputs=(c,), operand_mapping=maps)


def _mapped(**changes) -> MappedNode:
    """The mapping of an 8x4x2 matmul on 16 units over two temporal loops, A and C kept in ``sram``."""
    fields = {
        "latency": 10.0,
        "energy_j": 100 * PJ,
        "actions": {
            (COMPUTE, "None", "compute"): 64.0,
            ("sram", "A", "read"): 4.0,
            ("sram", "C", "read"): 2.0,
            ("sram", "C", "write"): 3.0,
        },
        "bits_per_action": {("sram", "read"): 64.0, ("sram", "write"): 32.0},
        "spatial_unrolling": {0: 8, 2: 2},
        "temporal_cycles": 4,
        "top_storage": {"A": "sram", "C": "sram"},
        "units": 16,
        "mapping": ("[A,C in sram]", "MAC computes g"),
        "memory_order": "free",
        "fallback": None,
        "wall_s": 0.1,
    }
    return MappedNode(**(fields | changes))


def test_port_traffic_is_each_operands_top_memory_actions_at_their_width():
    assert port_traffic(_gemm(), example_core("simd"), _mapped()) == (
        (input_role(1), READ_BY_DATAPATH, 4 * 64.0),
        (OUTPUT, READ_BY_DATAPATH, 2 * 64.0),
        (OUTPUT, WRITE_BY_DATAPATH, 3 * 32.0),
    )


def test_partial_sums_move_at_the_accumulator_precision_the_core_states():
    core = example_core("tpu_like")  # int32 accumulator, bf16 output: partial sums twice as wide
    actions = _mapped().actions | {("sram", "C", "write"): 6.0}  # 192 bits written, 128 of them read back
    traffic = {(role, way): bits for role, way, bits in port_traffic(_gemm(), core, _mapped(actions=actions))}
    assert traffic[(OUTPUT, READ_BY_DATAPATH)] == 2 * 128.0
    assert traffic[(OUTPUT, WRITE_BY_DATAPATH)] == 192.0 + 128.0  # the writes later read back widen too


def test_a_point_that_expands_into_several_operators_costs_their_cycles():
    workload, accelerator, mapping = inputs()
    estimator = AccelForgeEstimator(ZigZagCostEstimator(workload, accelerator, mapping), AccelForgeOptions())
    core = accelerator.get_core(0)
    plain = estimator.entry(_gemm(), core, _mapped())
    assert (plain.latency_total, plain.ideal_temporal_cycle, plain.ideal_cycle) == (10.0, 4.0, 4.0)
    silu = estimator.entry(_gemm("Silu"), core, _mapped())  # four cycles per point, one operation
    assert (silu.latency_total, silu.ideal_temporal_cycle, silu.ideal_cycle) == (16.0, 16.0, 16.0)
    assert plain.energy_total == silu.energy_total == pytest.approx(100.0)


def test_a_node_no_einsum_can_express_is_costed_by_zigzag_and_says_why():
    x, w, o = Tensor.create("I", bf16, (4, 23)), Tensor.create("W", bf16, (8, 4, 2)), Tensor.create("O", bf16, (8, 8))
    maps = tuple(
        AffineMap.from_callable(fn)
        for fn in (lambda k, c, ox, fx: (c, 3 * ox + fx), lambda k, c, ox, fx: (k, c, fx), lambda k, c, ox, fx: (k, ox))
    )
    node = ComputationNode(type="Conv", name="gapped", inputs=(x, w), outputs=(o,), operand_mapping=maps)
    workload = Workload(
        [InEdge(name="i", outputs=(x,)), InEdge(name="w", outputs=(w,)), node, OutEdge(name="o", inputs=(o,))]
    )
    estimator = AccelForgeEstimator(ZigZagCostEstimator(workload, None, None), AccelForgeOptions())
    entry = estimator.estimate(node, example_core("tpu_like"))
    assert entry.metadata["backend"] in ("zigzag", IDEAL_CYCLE_BACKEND)
    assert "leaves gaps" in entry.metadata["accelforge_refused"]


LOAD_STREAM = """
import sys
import stream.api
from stream.stages.estimation.core_cost_backends import discover_backends
print("accelforge" in {backend.name for backend in discover_backends()}, "accelforge" in sys.modules)
"""


def test_loading_stream_and_its_backends_does_not_import_accelforge():
    result = subprocess.run([sys.executable, "-c", LOAD_STREAM], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["True", "False"]


def _invert_output_memories(data: dict) -> None:
    data["memories"]["rf_16B"]["served_dimensions"] = ["D1", "D2"]
    data["memories"]["sram_8KB"]["served_dimensions"] = []


def _empty_array(data: dict) -> None:
    data["operational_array"]["sizes"] = [0, 12]


def _drop_read_port(data: dict) -> None:
    memory = data["memories"]["rf_64B_A"]
    memory["ports"] = [p for p in memory["ports"] if p["type"] != "read"]


@pytest.mark.parametrize(
    "change, reason",
    [
        (_invert_output_memories, "operand O's memories are copied more often further from the array"),
        (_empty_array, "its operational array has no units"),
        (_drop_read_port, "a memory needs a read and a write port"),
    ],
)
def test_cores_a_tree_cannot_express_are_refused(change, reason):
    data = yaml.safe_load((CORES / "eyeriss_like.yaml").read_text())
    change(data)
    with pytest.raises(UnsupportedConversion, match=reason):
        zigzag_core(load_core(data))


def test_a_memory_core_is_refused():
    with pytest.raises(UnsupportedConversion, match="only zigzag compute cores convert"):
        zigzag_core(example_core("tpu_v7_vmem"))
