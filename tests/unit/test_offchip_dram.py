import pytest

from stream.cost_model.offchip_dram import DramProfile, contiguous_span_bytes

# The Strix profile as whole_array_strix.yaml declares it, bits per cycle.
STRIX = {
    "ceiling": 296.6,
    "contiguous": 148.3,
    "strided": {
        "read": {32: 19.0, 64: 35.2, 128: 53.5, 256: 57.2, 512: 93.5},
        "write": {32: 17.1, 64: 28.3, 128: 53.2, 256: 54.2, 512: 93.5},
    },
}


def test_a_description_without_a_profile_has_none():
    assert DramProfile.from_description(None) is None
    assert DramProfile.from_description({}) is None


def test_efficiency_is_the_measured_ratio_at_a_measured_span():
    p = DramProfile.from_description(STRIX)
    assert p.efficiency(128, "read") == pytest.approx(53.5 / 148.3)
    assert p.efficiency(32, "write") == pytest.approx(17.1 / 148.3)


def test_efficiency_rises_with_span_and_saturates_at_contiguous():
    p = DramProfile.from_description(STRIX)
    spans = [16, 32, 64, 128, 256, 512, 1024, 4096]
    effs = [p.efficiency(s, "read") for s in spans]
    assert effs == sorted(effs)
    assert p.efficiency(16, "read") == p.efficiency(32, "read")  # clamped below the table
    assert p.efficiency(4096, "read") == 1.0


def test_contiguous_bytes_at_the_ceiling():
    p = DramProfile.from_description(STRIX)
    # 16 MB contiguous each way, both directions: the benchmark that set the ceiling.
    bits = 16 * 2**20 * 8
    assert p.cycles([(bits, 1 << 20, "read"), (bits, 1 << 20, "write")]) == pytest.approx(2 * bits / 296.6)


def test_span_is_the_inner_extent_until_a_dimension_is_whole():
    # A (64, 512) block of a (1024, 4096) bf16 tensor: 512 elements a row.
    assert contiguous_span_bytes((64, 512), (1024, 4096), 16) == 1024
    # Whole rows run on into the next one: (8, 4096) of (2048, 4096) is one 64 KB run.
    assert contiguous_span_bytes((8, 4096), (2048, 4096), 16) == 8 * 4096 * 2
    # Without the enclosing tensor, only the inner extent is known.
    assert contiguous_span_bytes((256, 64), None, 16) == 128
