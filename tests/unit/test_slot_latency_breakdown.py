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
    evaluate_mapping(ACCELERATOR, make_2_conv_workload(two_conv), str(tmp_path), make_2_conv_mapping(two_conv))
    (path,) = tmp_path.rglob("slot_latency_breakdown.yaml")
    slots = yaml.safe_load(path.read_text())["slots"]
    transfers = [tr for slot in slots for tr in slot["transfer_contributors"]]
    assert transfers
    assert all(tr["reuse_factor"] is not None and tr["latency_contribution_cycles"] is not None for tr in transfers)
