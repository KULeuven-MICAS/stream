"""Tests for the end-to-end MAC roofline metric."""

from __future__ import annotations

import os

import pytest
from zigzag.utils import open_yaml

from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.parser.accelerator_factory import AcceleratorFactory
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.workload.utils import is_mac_operator_type

HARDWARE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "stream", "inputs", "examples", "hardware")
TPU_V7 = os.path.abspath(os.path.join(HARDWARE_DIR, "tpu_v7_ironwood.yaml"))

# Verified TPU7x reference: SwiGLU 256x512x2048 fused solves at 2863 cycles vs an ideal 384 on 32 MXUs.
SWIGLU_REF_MAC_OPS = 805_306_368
SWIGLU_REF_LATENCY = 2863
TPU_V7_MXU_PEAK = 32 * 256 * 256


def load_accelerator(path: str):
    data = open_yaml(path)
    validator = AcceleratorValidator(data, path)
    assert validator.validate()
    return AcceleratorFactory(validator.normalized_data).create()


def make_scheduler(accelerator, total_mac_ops: int, latency_total: int) -> SteadyStateScheduler:
    """A scheduler stub carrying only what the roofline reads, so the test needs no solve."""
    scheduler = object.__new__(SteadyStateScheduler)
    scheduler.accelerator = accelerator
    scheduler.total_mac_ops = total_mac_ops
    scheduler.latency_total = latency_total
    scheduler.performance_stats = {"aggregate": {}}
    return scheduler


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
        """TPU7x has 32 MXU + 4 VPU + 4 VMEM + 1 HBM core. Only the MXUs admit MatMul/Gemm/Conv."""
        accelerator = load_accelerator(TPU_V7)
        scheduler = make_scheduler(accelerator, SWIGLU_REF_MAC_OPS, SWIGLU_REF_LATENCY)
        peak, n_cores = scheduler._mac_roofline_peak()
        assert n_cores == 32
        assert peak == TPU_V7_MXU_PEAK == 2097152

    def test_vector_cores_are_excluded_from_the_peak(self) -> None:
        """The peak excludes the VPUs; fails if they ever creep back into the roofline denominator."""
        accelerator = load_accelerator(TPU_V7)
        offchip_id = accelerator.offchip_core_id
        all_cores_peak = sum(
            getattr(getattr(c, "operational_array", None), "total_unit_count", 0) or 0
            for c in accelerator.core_list
            if c.id != offchip_id
        )
        scheduler = make_scheduler(accelerator, SWIGLU_REF_MAC_OPS, SWIGLU_REF_LATENCY)
        peak, _ = scheduler._mac_roofline_peak()
        assert all_cores_peak == 2097152 + 4 * 8 * 128  # + the four (8, 128) VPUs
        assert peak < all_cores_peak

    def test_unrestricted_cores_count_but_specialised_non_mac_cores_do_not(self) -> None:
        """The four unrestricted cores stay in the peak; the two non-MAC specialised cores do not."""
        accelerator = load_accelerator(os.path.join(HARDWARE_DIR, "tpu_like_quad_core.yaml"))
        offchip_id = accelerator.offchip_core_id
        unrestricted = [
            c for c in accelerator.core_list if c.id != offchip_id and getattr(c, "operator_types", None) is None
        ]
        scheduler = make_scheduler(accelerator, 1, 1)
        peak, n_cores = scheduler._mac_roofline_peak()
        assert n_cores == len(unrestricted) == 4
        assert peak == sum(c.operational_array.total_unit_count for c in unrestricted)
        assert peak < sum(c.operational_array.total_unit_count for c in accelerator.core_list if c.id != offchip_id)


class TestEndToEndMacUtilization:
    def test_swiglu_ref_matches_the_hand_computed_roofline(self) -> None:
        """805,306,368 MACs over 32x(256x256) is an ideal 384 cycles; solved 2863 -> ~13.4% util."""
        accelerator = load_accelerator(TPU_V7)
        scheduler = make_scheduler(accelerator, SWIGLU_REF_MAC_OPS, SWIGLU_REF_LATENCY)
        scheduler._augment_performance_stats_end_to_end()
        agg = scheduler.performance_stats["aggregate"]

        ideal_cycles = SWIGLU_REF_MAC_OPS / TPU_V7_MXU_PEAK
        assert ideal_cycles == 384
        assert agg["peak_macs_per_cycle"] == TPU_V7_MXU_PEAK
        assert agg["mac_capable_cores"] == 32
        assert agg["total_mac_ops"] == SWIGLU_REF_MAC_OPS
        assert agg["end_to_end_mac_utilization"] == pytest.approx(ideal_cycles / SWIGLU_REF_LATENCY)
        assert agg["end_to_end_mac_utilization"] == pytest.approx(0.13413, abs=1e-5)

    def test_no_mac_work_reports_none_not_zero(self) -> None:
        """A workload with no matmul/conv has no MAC roofline: None, not a misleading measured 0.0."""
        accelerator = load_accelerator(TPU_V7)
        scheduler = make_scheduler(accelerator, 0, SWIGLU_REF_LATENCY)
        scheduler._augment_performance_stats_end_to_end()
        assert scheduler.performance_stats["aggregate"]["end_to_end_mac_utilization"] is None

    def test_missing_aggregate_is_a_no_op(self) -> None:
        """Observability must never break a solved run."""
        accelerator = load_accelerator(TPU_V7)
        scheduler = make_scheduler(accelerator, SWIGLU_REF_MAC_OPS, SWIGLU_REF_LATENCY)
        scheduler.performance_stats = None
        scheduler._augment_performance_stats_end_to_end()
        assert scheduler.performance_stats is None
