"""Tests for the end-to-end MAC roofline metric."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from zigzag.utils import open_yaml

from stream.cost_model.core_cost import IDEAL_CYCLE_BACKEND, CoreCostEntry
from stream.opt.allocation.constraint_optimization.report import (
    end_to_end_mac_utilization,
    mac_roofline_peak,
    node_utilization,
)
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.utils import is_mac_operator_type

HARDWARE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "stream", "inputs", "examples", "hardware")
TPU_V7 = os.path.abspath(os.path.join(HARDWARE_DIR, "tpu_v7_ironwood.yaml"))

# A SwiGLU 256x512x2048 (805M MACs) finishing in 2863 cycles, against an ideal 768 on the eight 256x256x2 MXUs.
SWIGLU_REF_MAC_OPS = 805_306_368
SWIGLU_REF_LATENCY = 2863
TPU_V7_MXU_PEAK = 8 * 256 * 256 * 2


def load_accelerator(path: str):
    data = open_yaml(path)
    validator = AcceleratorValidator(data, path)
    assert validator.validate()
    return AcceleratorFactory(validator.normalized_data).create()


class TestIsMacOperatorType:
    @pytest.mark.parametrize("op", ["MatMul", "Gemm", "Conv", "matmul", "Linear", "ConvTranspose", "MatMulInteger"])
    def test_mac_ops(self, op: str) -> None:
        assert is_mac_operator_type(op)

    @pytest.mark.parametrize("op", ["Mul", "Add", "Silu", "Softmax", "MaxPool", "Div", "Exp"])
    def test_non_mac_ops(self, op: str) -> None:
        """``Mul`` must not match ``matmul`` -- that would wrongly admit the VPU into the MAC roofline."""
        assert not is_mac_operator_type(op)


class TestMacRooflinePeak:
    def test_tpu_v7_counts_only_the_mxus(self) -> None:
        """TPU7x has 8 MXU + 4 VPU + 4 VMEM + 1 HBM core. Only the MXUs admit MatMul/Gemm/Conv."""
        accelerator = load_accelerator(TPU_V7)
        peak, n_cores = mac_roofline_peak(accelerator)
        assert n_cores == 8
        assert peak == TPU_V7_MXU_PEAK == 1048576

    def test_vector_cores_are_excluded_from_the_peak(self) -> None:
        """The peak excludes the VPUs; fails if they ever creep back into the roofline denominator."""
        accelerator = load_accelerator(TPU_V7)
        offchip_id = accelerator.offchip_core_id
        all_cores_peak = sum(
            getattr(getattr(c, "operational_array", None), "total_unit_count", 0) or 0
            for c in accelerator.core_list
            if c.id != offchip_id
        )
        peak, _ = mac_roofline_peak(accelerator)
        assert all_cores_peak == 1048576 + 4 * 8 * 128 * 4  # + the four (8, 128, 4) VPUs
        assert peak < all_cores_peak

    def test_unrestricted_cores_count_but_specialised_non_mac_cores_do_not(self) -> None:
        """The four unrestricted cores stay in the peak; the two non-MAC specialised cores do not."""
        accelerator = load_accelerator(os.path.join(HARDWARE_DIR, "tpu_like_quad_core.yaml"))
        offchip_id = accelerator.offchip_core_id
        unrestricted = [
            c for c in accelerator.core_list if c.id != offchip_id and getattr(c, "operator_types", None) is None
        ]
        peak, n_cores = mac_roofline_peak(accelerator)
        assert n_cores == len(unrestricted) == 4
        assert peak == sum(c.operational_array.total_unit_count for c in unrestricted)
        assert peak < sum(c.operational_array.total_unit_count for c in accelerator.core_list if c.id != offchip_id)


class TestEndToEndMacUtilization:
    def test_swiglu_ref_matches_the_hand_computed_roofline(self) -> None:
        """805,306,368 MACs over 8x(256x256x2) is an ideal 768 cycles; done in 2863 -> ~26.8% util."""
        accelerator = load_accelerator(TPU_V7)
        agg = end_to_end_mac_utilization(accelerator, SWIGLU_REF_MAC_OPS, SWIGLU_REF_LATENCY)

        ideal_cycles = SWIGLU_REF_MAC_OPS / TPU_V7_MXU_PEAK
        assert ideal_cycles == 768
        assert agg["peak_macs_per_cycle"] == TPU_V7_MXU_PEAK
        assert agg["mac_capable_cores"] == 8
        assert agg["total_mac_ops"] == SWIGLU_REF_MAC_OPS
        assert agg["end_to_end_mac_utilization"] == pytest.approx(ideal_cycles / SWIGLU_REF_LATENCY)
        assert agg["end_to_end_mac_utilization"] == pytest.approx(0.26825, abs=1e-5)

    def test_no_mac_work_reports_none_not_zero(self) -> None:
        """A workload with no matmul/conv has no MAC roofline: None, not a misleading measured 0.0."""
        accelerator = load_accelerator(TPU_V7)
        assert end_to_end_mac_utilization(accelerator, 0, SWIGLU_REF_LATENCY)["end_to_end_mac_utilization"] is None


def cost(node_type: str, backend: str, ideal: float = 98.0) -> CoreCostEntry:
    layer = SimpleNamespace(type=node_type)
    return CoreCostEntry(0.0, ideal, ideal, ideal, layer=layer, metadata={"backend": backend})


class TestNodeUtilization:
    def test_efficiency_compares_one_call_with_its_ideal(self) -> None:
        """A node idle on 15 of 16 iterations runs 6 cycles per iteration, yet each call meets its ideal."""
        row = node_utilization(cost("Relu", "zigzag"), n_cores=1, runtime=98, active=6)
        assert (row["latency_cycles"], row["compute_efficiency"]) == (6, 1.0)

    @pytest.mark.parametrize(
        ("node_type", "backend", "fallback"),
        [("Conv", IDEAL_CYCLE_BACKEND, True), ("Relu", IDEAL_CYCLE_BACKEND, False), ("Conv", "aie", False)],
    )
    def test_only_a_matmul_or_conv_costed_at_ideal_cycles_fell_back(
        self, node_type: str, backend: str, fallback: bool
    ) -> None:
        """A kernel library prices AIE cores without a ZigZag evaluation; that is a cost, not a fallback."""
        assert node_utilization(cost(node_type, backend), n_cores=1, runtime=98, active=98)["fallback"] is fallback
