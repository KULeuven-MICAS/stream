"""Validation tests for the Phase 39 metrics-capture hook (CAP-01/02/03).

Three tests:
- test_conditional_write_skips_on_empty_store  (fast, default suite)
- test_conditional_write_on_populated_store    (fast, default suite)
- test_capture_emits_file_and_fields           (slow, @pytest.mark.slow)
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

# Load the project-local tests/conftest.py by absolute path so that the import
# is CWD- and sys.path-independent.  A bare `import tests.conftest` resolves to
# the xdsl installed-package `tests` namespace when pytest is invoked as
# `pytest tests/` (no CWD prepend to sys.path), shadowing the local module.
_conftest_path = pathlib.Path(__file__).resolve().parent / "conftest.py"
_spec = importlib.util.spec_from_file_location("_local_conftest", _conftest_path)
assert _spec is not None and _spec.loader is not None
conftest_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest_module)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Fast unit tests: conditional-write contract (CAP-01 / CAP-02)
# ---------------------------------------------------------------------------
# These tests drive pytest_terminal_summary directly (no subprocess, no solver).
# They monkeypatch _metrics_store and conftest_module.__file__ to redirect the
# absolute output path (Path(__file__).parent.parent / "metrics_current.json")
# into tmp_path.  monkeypatch.chdir alone is NOT sufficient because the writer
# derives the path from conftest's __file__, not from CWD.


def test_conditional_write_skips_on_empty_store(tmp_path, monkeypatch):
    """D-01 / CAP-01: empty store -> no file written, no clobber."""
    # Redirect the writer's output to tmp_path.
    # conftest.py: out = Path(__file__).parent.parent / "metrics_current.json"
    # So we make __file__ = tmp_path / "pkg" / "conftest.py"
    # => parent = tmp_path / "pkg", parent.parent = tmp_path
    fake_conftest = tmp_path / "pkg" / "conftest.py"
    fake_conftest.parent.mkdir(parents=True)
    monkeypatch.setattr(conftest_module, "__file__", str(fake_conftest))

    monkeypatch.setattr(conftest_module, "_metrics_store", {})

    conftest_module.pytest_terminal_summary(None, 0, None)

    assert (tmp_path / "metrics_current.json").exists() is False


def test_conditional_write_on_populated_store(tmp_path, monkeypatch):
    """D-01/D-02/D-03/D-04 / CAP-01/CAP-02: populated store -> sorted node-ID-keyed file with 5 fields."""
    fake_conftest = tmp_path / "pkg" / "conftest.py"
    fake_conftest.parent.mkdir(parents=True)
    monkeypatch.setattr(conftest_module, "__file__", str(fake_conftest))

    node_id = "tests/test_hardware_combinations.py::test_hardware_two_conv[eyeriss_like_single_core]"
    populated: dict[str, dict] = {
        node_id: {
            "total_latency": 114999.0,
            "group_latencies_max": 114999.0,
            "objective": 10.0,
            "mip_gap": None,
            "solve_time_s": 0.004,
        }
    }
    monkeypatch.setattr(conftest_module, "_metrics_store", populated)

    conftest_module.pytest_terminal_summary(None, 0, None)

    out = tmp_path / "metrics_current.json"
    assert out.exists(), "expected metrics_current.json to be written"

    data = json.loads(out.read_text())
    assert list(data.keys()) == [node_id], f"unexpected outer keys: {list(data.keys())}"

    entry = data[node_id]
    expected_fields = {"total_latency", "group_latencies_max", "objective", "mip_gap", "solve_time_s"}
    assert set(entry.keys()) == expected_fields, f"field mismatch: {set(entry.keys())}"

    # D-02: sort_keys=True -> inner keys must be alphabetically ordered
    assert list(entry.keys()) == sorted(entry.keys()), f"inner keys not alphabetically sorted: {list(entry.keys())}"


# ---------------------------------------------------------------------------
# Slow subprocess integration test: real matrix-cell capture (CAP-01/02/03)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_capture_emits_file_and_fields(tmp_path):
    """CAP-01/02/03: a real subprocess matrix-cell run emits metrics_current.json.

    Runs one matrix cell out-of-process so the real pytest_terminal_summary path
    (including absolute __file__ resolution and the D-02 fixed path) is exercised.

    The exact node-ID key assertion also covers CAP-03 '-x survival key format':
    a single completed cell yields a partial file with a valid full node-ID key --
    the same shape a -x abort would leave after the first cell completes.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    out = repo_root / "metrics_current.json"

    # Remove any stale file so we get a clean assertion (metrics_current.json is
    # a gitignored regenerated artifact -- safe to remove before re-running).
    if out.exists():
        out.unlink()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_hardware_combinations.py",
        "-k",
        "two_conv and eyeriss_like_single_core",
        "-m",
        "not slow",
        "-q",
        "--no-header",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    assert out.exists(), "metrics_current.json not written by the subprocess run"

    data = json.loads(out.read_text())
    assert len(data) == 1, f"expected exactly 1 captured cell, got {len(data)}"

    (key,) = data.keys()
    expected_key = "tests/test_hardware_combinations.py::test_hardware_two_conv[eyeriss_like_single_core]"
    assert key == expected_key, f"unexpected node-ID key: {key!r}"

    entry = data[key]
    assert set(entry) == {
        "total_latency",
        "group_latencies_max",
        "objective",
        "mip_gap",
        "solve_time_s",
    }, f"D-04 field mismatch: {set(entry)}"

    assert isinstance(entry["total_latency"], int | float) and entry["total_latency"] > 0, (
        f"total_latency must be a positive float, got {entry['total_latency']!r}"
    )
    assert entry["mip_gap"] is None, f"expected mip_gap=None (OR-Tools), got {entry['mip_gap']!r}"
    assert entry["solve_time_s"] is not None and entry["solve_time_s"] >= 0, (
        f"solve_time_s must be >= 0, got {entry['solve_time_s']!r}"
    )
