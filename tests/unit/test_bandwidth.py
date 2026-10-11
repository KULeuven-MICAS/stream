import pytest

from stream.cost_model.bandwidth import BandwidthModel
from stream.cost_model.layout import Layout

STRIX = {
    "ceiling": 296.6,
    "contiguous": 148.3,
    "strided": {
        "read": {32: 19.0, 64: 35.2, 128: 53.5, 256: 57.2, 512: 93.5},
        "write": {32: 17.1, 64: 28.3, 128: 53.2, 256: 54.2, 512: 93.5},
    },
}


def test_efficiency_is_the_measured_ratio_at_a_measured_span():
    model = BandwidthModel.from_description(STRIX)
    assert model.efficiency(128, "read") == pytest.approx(53.5 / 148.3)
    assert model.efficiency(32, "write") == pytest.approx(17.1 / 148.3)


def test_efficiency_rises_with_span_and_saturates_at_contiguous():
    model = BandwidthModel.from_description(STRIX)
    spans = [16, 32, 64, 128, 256, 512, 1024, 4096]
    efficiencies = [model.efficiency(s, "read") for s in spans]
    assert efficiencies == sorted(efficiencies)
    assert model.efficiency(16, "read") == model.efficiency(32, "read")
    assert model.efficiency(4096, "read") == 1.0


def test_span_is_the_inner_extent_until_a_dimension_is_whole():
    assert Layout.row_major(2).contiguous_bytes((64, 512), (1024, 4096), 16) == 1024
    assert Layout.row_major(2).contiguous_bytes((8, 4096), (2048, 4096), 16) == 8 * 4096 * 2
    assert Layout.row_major(2).contiguous_bytes((256, 64), (256, 64), 16) == 256 * 64 * 2
