from pathlib import Path

import pytest
import yaml

from stream.api import evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"


@pytest.mark.slow
def test_slot_latency_breakdown_reports_solved_reuse_factor_and_contribution(
    tmp_path: Path, two_conv: TwoConvWorkloadConfig
) -> None:
    """One iteration, no fused loop: the fill is what the run adds to its iteration, and no node waits absent."""
    workload = make_2_conv_workload(two_conv, str(tmp_path))
    estimate = evaluate_mapping(ACCELERATOR, workload, str(tmp_path), make_2_conv_mapping(two_conv, str(tmp_path)))
    (path,) = tmp_path.rglob("slot_latency_breakdown.yaml")
    breakdown = yaml.safe_load(path.read_text())
    slots, totals = breakdown["slots"], breakdown["totals"]
    (group,) = estimate.context.data["group_allocations"].values()
    assert group["steady_state"]["iterations"] == 1
    assert totals["fill_cycles"] == totals["total_latency_cycles"] - totals["iteration_latency_cycles"]
    assert totals["absent_wait_cycles"] == 0
    transfers = [tr for slot in slots for tr in slot["transfer_contributors"]]
    assert transfers
    assert all(tr["reuse_factor"] is not None and tr["latency_contribution_cycles"] is not None for tr in transfers)
