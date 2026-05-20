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
        "nonsensical" in record.message.lower() for record in caplog.records if record.levelno == logging.WARNING
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
    from stream.opt.solver import ConstraintSelection as ConstraintSelectionImported  # noqa: F401

    assert ConstraintSelectionImported is ConstraintSelection


# ---------------------------------------------------------------------------
# Guard verification tests (Phase 5 Plan 02)
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock  # noqa: E402


def _make_tta_stub(constraint_selection, *, bind_objective=False):
    """Create a minimal TTA-like object with constraint_selection set.

    We cannot instantiate a real TTA without a Workload, so we create
    a mock that has the constraint_selection attribute and delegates
    to the real _create_constraints / _overlap_and_objective methods.

    bind_objective: if True, also bind the real _set_total_latency_and_objective
    so its DMA conditional logic executes. Only needed for the objective test.
    """
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
        TransferAndTensorAllocator,
    )

    tta = MagicMock(spec=TransferAndTensorAllocator)
    tta.constraint_selection = constraint_selection
    # Bind the real dispatch methods so if-guards execute
    tta._create_constraints = TransferAndTensorAllocator._create_constraints.__get__(tta)
    tta._overlap_and_objective = TransferAndTensorAllocator._overlap_and_objective.__get__(tta)
    if bind_objective:
        tta._set_total_latency_and_objective = TransferAndTensorAllocator._set_total_latency_and_objective.__get__(tta)
    return tta


def test_memory_capacity_guard():
    """TTA with memory_capacity=False does not call _memory_capacity_constraints."""
    cs = ConstraintSelection(memory_capacity=False)
    tta = _make_tta_stub(cs)
    tta._create_constraints()
    tta._memory_capacity_constraints.assert_not_called()
    # Structural constraints should still be called
    tta._tensor_placement_constraints.assert_called_once()


def test_memory_capacity_enabled():
    """TTA with memory_capacity=True calls _memory_capacity_constraints."""
    cs = ConstraintSelection(memory_capacity=True)
    tta = _make_tta_stub(cs)
    tta._create_constraints()
    tta._memory_capacity_constraints.assert_called_once()


def test_object_fifo_guard():
    """TTA with object_fifo_depth=False does not call _object_fifo_depth_constraints."""
    cs = ConstraintSelection(object_fifo_depth=False)
    tta = _make_tta_stub(cs)
    tta._create_constraints()
    tta._object_fifo_depth_constraints.assert_not_called()


def test_buffer_descriptor_guard():
    """TTA with buffer_descriptors=False does not call _buffer_descriptor_constraints."""
    cs = ConstraintSelection(buffer_descriptors=False)
    tta = _make_tta_stub(cs)
    tta._create_constraints()
    tta._buffer_descriptor_constraints.assert_not_called()


def test_dma_guard():
    """TTA with dma_channels=False does not call _add_dma_usage_constraints."""
    cs = ConstraintSelection(dma_channels=False)
    tta = _make_tta_stub(cs)
    tta.max_slot = 0
    tta.big_m = 10
    tta._overlap_and_objective()
    tta._add_dma_usage_constraints.assert_not_called()


def test_dma_enabled():
    """TTA with dma_channels=True calls _add_dma_usage_constraints."""
    cs = ConstraintSelection(dma_channels=True)
    tta = _make_tta_stub(cs)
    tta.max_slot = 0
    tta.big_m = 10
    tta._overlap_and_objective()
    tta._add_dma_usage_constraints.assert_called_once()


def test_dma_objective_no_dma_terms():
    """When dma_channels=False, primary objective = total_lat only (no DMA vars)."""
    cs = ConstraintSelection(dma_channels=False)
    tta = _make_tta_stub(cs, bind_objective=True)
    # Set up minimal mocks for _set_total_latency_and_objective
    mock_model = MagicMock()
    mock_total_lat = MagicMock()
    mock_total_lat._raw = "total_lat_raw"
    mock_model.add_var.return_value = mock_total_lat
    mock_model.quicksum.return_value = MagicMock(_raw=0)
    tta.model = mock_model
    tta.overlap = MagicMock()
    tta.iterations = 1
    tta.slot_latency = {}
    tta.tensors_to_optimize_reuse_for = []
    tta._set_total_latency_and_objective()
    # Verify lexicographic objectives were set with primary = total_lat only
    mock_model.set_lexicographic_objectives.assert_called_once()
    objectives = mock_model.set_lexicographic_objectives.call_args[0][0]
    primary = next(o for o in objectives if o.name == "latency")
    assert primary.expr == "total_lat_raw"
    assert primary.priority == 2


def test_skip_warnings(caplog):
    """Each disabled group produces a WARNING log containing the group name."""
    cs = ConstraintSelection(
        memory_capacity=False,
        object_fifo_depth=False,
        buffer_descriptors=False,
        dma_channels=False,
    )
    tta = _make_tta_stub(cs)
    with caplog.at_level(logging.WARNING):
        tta._create_constraints()
        tta.max_slot = 0
        tta.big_m = 10
        tta._overlap_and_objective()
    assert "memory_capacity" in caplog.text.lower()
    assert "object_fifo_depth" in caplog.text.lower()
    assert "buffer_descriptors" in caplog.text.lower()
    assert "dma_channels" in caplog.text.lower()


def test_all_enabled_calls_all():
    """TTA with default ConstraintSelection() (all True) calls all constraint methods."""
    cs = ConstraintSelection()  # all True
    tta = _make_tta_stub(cs)
    tta._create_constraints()
    tta._memory_capacity_constraints.assert_called_once()
    tta._object_fifo_depth_constraints.assert_called_once()
    tta._buffer_descriptor_constraints.assert_called_once()
    tta.max_slot = 0
    tta.big_m = 10
    tta._overlap_and_objective()
    tta._add_dma_usage_constraints.assert_called_once()
