from pathlib import Path

import pytest
import yaml

from stream.api import evaluate_mapping
from stream.inputs.testing.mapping.make_2_conv_mapping import make_2_conv_mapping
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload

ACCELERATOR = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
TWO_CONV = TwoConvWorkloadConfig(
    batch_size=1,
    in_channels=8,
    height=32,
    width=32,
    out_channels_1=16,
    out_channels_2=32,
    kernel_size=3,
    in_dtype="bf16",
    weight_dtype="bf16",
)


@pytest.mark.slow
def test_slot_latency_breakdown_reports_solved_reuse_factor_and_contribution(tmp_path: Path) -> None:
    evaluate_mapping(ACCELERATOR, make_2_conv_workload(TWO_CONV), str(tmp_path), make_2_conv_mapping(TWO_CONV))
    (path,) = tmp_path.rglob("slot_latency_breakdown.yaml")
    slots = yaml.safe_load(path.read_text())["slots"]
    transfers = [tr for slot in slots for tr in slot["transfer_contributors"]]
    assert transfers
    assert all(tr["reuse_factor"] is not None and tr["contribution"] is not None for tr in transfers)
