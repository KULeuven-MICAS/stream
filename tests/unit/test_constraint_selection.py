"""Unit tests for ConstraintSelection frozen dataclass.

Covers all behavior from Phase 5 Plan 01 requirements:
  - Defaults all fields to True
  - Partial override works correctly
  - Immutability (FrozenInstanceError on mutation)
  - WARNING logged for nonsensical combination (memory_capacity=False + object_fifo_depth=True)
  - No warning for sensible combinations
  - Importable from stream.opt.solver public API
"""

import dataclasses
import logging

import pytest

from stream.opt.solver import ConstraintSelection


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_defaults_all_true():
    """ConstraintSelection() has all four bool fields defaulting to True."""
    cs = ConstraintSelection()
    assert cs.memory_capacity is True
    assert cs.object_fifo_depth is True
    assert cs.buffer_descriptors is True
    assert cs.dma_channels is True


# ---------------------------------------------------------------------------
# Partial override
# ---------------------------------------------------------------------------


def test_partial_override():
    """ConstraintSelection(memory_capacity=False) sets that field only; others remain True."""
    cs = ConstraintSelection(memory_capacity=False)
    assert cs.memory_capacity is False
    assert cs.object_fifo_depth is True
    assert cs.buffer_descriptors is True
    assert cs.dma_channels is True


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_immutable():
    """Assigning to any field on ConstraintSelection raises FrozenInstanceError."""
    cs = ConstraintSelection()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cs.memory_capacity = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Warning behavior
# ---------------------------------------------------------------------------


def test_warning_nonsensical(caplog):
    """ConstraintSelection(memory_capacity=False, object_fifo_depth=True) logs a WARNING containing 'nonsensical'."""
    with caplog.at_level(logging.WARNING, logger="stream.opt.solver.solver"):
        ConstraintSelection(memory_capacity=False, object_fifo_depth=True)
    assert any(
        "nonsensical" in record.message.lower()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ), f"Expected WARNING with 'nonsensical', got records: {caplog.records}"


def test_no_warning_sensible(caplog):
    """ConstraintSelection(memory_capacity=False, object_fifo_depth=False) produces NO warning."""
    with caplog.at_level(logging.WARNING, logger="stream.opt.solver.solver"):
        ConstraintSelection(memory_capacity=False, object_fifo_depth=False)
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 0, f"Expected no WARNING, got: {warning_records}"


def test_no_warning_all_true(caplog):
    """ConstraintSelection() (all True) produces NO warning."""
    with caplog.at_level(logging.WARNING, logger="stream.opt.solver.solver"):
        ConstraintSelection()
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 0, f"Expected no WARNING, got: {warning_records}"


# ---------------------------------------------------------------------------
# Public import
# ---------------------------------------------------------------------------


def test_importable():
    """ConstraintSelection is importable from stream.opt.solver public API."""
    from stream.opt.solver import ConstraintSelection as CS  # noqa: F401

    assert CS is ConstraintSelection
