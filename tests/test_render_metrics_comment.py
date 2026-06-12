"""Unit tests for render_metrics_comment.py (BASE-01..03, DIFF-01, DIFF-02).

All tests use synthetic JSON dicts — no solver, no matrix run.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

# Load script by absolute path (mirrors test_metrics_capture.py pattern — avoids xdsl
# installed-package `tests` namespace shadowing a bare module import).
_script_path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "analysis" / "render_metrics_comment.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("render_metrics_comment", _script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

_ONE_CELL_BASE = {
    "_meta": {
        "python_version": "3.12.3",
        "baseline_sha": "abc123",
        "baseline_date": "2026-01-01",
        "backend": "ortools_gscip",
    },
    "tests/test_hw.py::test_two_conv[eyeriss]": {
        "group_latencies_max": 1000.0,
        "mip_gap": None,
        "objective": 10.0,
        "solve_time_s": 0.1,
        "total_latency": 1000.0,
    },
}

# solve_time_s differs (non-gated) but total_latency is identical -> no flag
_ONE_CELL_CURRENT_SAME = {
    "tests/test_hw.py::test_two_conv[eyeriss]": {
        "group_latencies_max": 1000.0,
        "mip_gap": None,
        "objective": 10.0,
        "solve_time_s": 0.12,
        "total_latency": 1000.0,
    },
}

# total_latency 1050 vs 1000 = 5% delta > tol=0.001
_ONE_CELL_CURRENT_FLAGGED = {
    "tests/test_hw.py::test_two_conv[eyeriss]": {
        "group_latencies_max": 1100.0,
        "mip_gap": None,
        "objective": 11.0,
        "solve_time_s": 0.1,
        "total_latency": 1050.0,
    },
}


# ---------------------------------------------------------------------------
# Test 1: within-tol -> no FLAGGED status
# ---------------------------------------------------------------------------


def test_within_tol_no_flag():
    """Within-tol cell -> no FLAGGED status; status line shows no-changes / ✅."""
    mod = _load_script()
    rows, captured, total = mod.compute_diffs(_ONE_CELL_CURRENT_SAME, _ONE_CELL_BASE, tol=0.001)
    assert all(r["status"] != "FLAGGED" for r in rows), f"unexpected FLAGGED: {rows}"
    comment = mod.render_comment(rows, captured, total, _ONE_CELL_BASE["_meta"], tol=0.001)
    assert "no changes" in comment.lower() or "✅" in comment


# ---------------------------------------------------------------------------
# Test 2: over-tol total_latency -> FLAGGED; objective/solve_time_s alone never FLAGGED
# ---------------------------------------------------------------------------


def test_over_tol_flagged():
    """total_latency |delta|/|base| > tol -> at least one FLAGGED row."""
    mod = _load_script()
    rows, captured, total = mod.compute_diffs(_ONE_CELL_CURRENT_FLAGGED, _ONE_CELL_BASE, tol=0.001)
    flagged = [r for r in rows if r["status"] == "FLAGGED"]
    assert len(flagged) >= 1, f"expected at least one FLAGGED row, got {rows}"

    # Prove objective/solve_time_s differences alone cannot produce FLAGGED
    # Use a current where total_latency matches but objective/solve_time_s differ wildly
    current_non_gated_diff = {
        "tests/test_hw.py::test_two_conv[eyeriss]": {
            "group_latencies_max": 1000.0,
            "mip_gap": None,
            "objective": 999.0,  # huge difference — but NOT gated
            "solve_time_s": 9999.0,  # huge difference — but NOT gated
            "total_latency": 1000.0,  # identical -> no flag
        },
    }
    rows2, _, _ = mod.compute_diffs(current_non_gated_diff, _ONE_CELL_BASE, tol=0.001)
    assert all(r["status"] != "FLAGGED" for r in rows2), (
        f"objective/solve_time_s differences alone should never set FLAGGED, got: {rows2}"
    )


# ---------------------------------------------------------------------------
# Test 3: baseline cell absent from current -> NO_DATA + X-of-Y banner
# ---------------------------------------------------------------------------


def test_no_data_banner():
    """Baseline key missing from current -> NO_DATA row; comment never reports a regression."""
    mod = _load_script()
    current_empty: dict = {}  # no cells captured (simulates -x abort before first cell)
    rows, captured, total = mod.compute_diffs(current_empty, _ONE_CELL_BASE, tol=0.001)
    no_data = [r for r in rows if r["status"] == "NO_DATA"]
    assert len(no_data) >= 1, f"expected NO_DATA row, got {rows}"
    # Baseline-only cells are NEVER classified as REMOVED in v1
    assert not any(r["status"] == "REMOVED" for r in rows), "REMOVED must not be emitted in v1"
    comment = mod.render_comment(rows, captured, total, _ONE_CELL_BASE["_meta"], tol=0.001)
    # X-of-Y banner: 0 of 1 captured
    assert "0 of 1" in comment or "NO DATA" in comment.upper()
    # Status line must NOT say there is a regression for the NO_DATA row
    assert "flagged" not in comment.lower() or "no changes" in comment.lower() or "✅" in comment


# ---------------------------------------------------------------------------
# Test 4: stable HTML marker present
# ---------------------------------------------------------------------------


def test_marker_present():
    """Stable HTML marker is present in every rendered comment."""
    mod = _load_script()
    rows, captured, total = mod.compute_diffs(_ONE_CELL_CURRENT_SAME, _ONE_CELL_BASE, tol=0.001)
    comment = mod.render_comment(rows, captured, total, _ONE_CELL_BASE["_meta"], tol=0.001)
    assert "<!-- stream-aie-metrics-regression-guard-v1 -->" in comment


# ---------------------------------------------------------------------------
# Test 5: subprocess invocation always exits 0 even on flagged diff
# ---------------------------------------------------------------------------


def test_exit_zero_on_diff(tmp_path):
    """Script exits 0 even when cells are flagged (advisory-only, D-11)."""
    base_file = tmp_path / "base.json"
    cur_file = tmp_path / "cur.json"
    out_file = tmp_path / "out.md"
    base_file.write_text(json.dumps(_ONE_CELL_BASE, indent=2))
    cur_file.write_text(json.dumps(_ONE_CELL_CURRENT_FLAGGED, indent=2))
    result = subprocess.run(
        [
            sys.executable,
            str(_script_path),
            "--current",
            str(cur_file),
            "--baseline",
            str(base_file),
            "--output",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"expected exit 0, got {result.returncode}: {result.stderr}"


# ---------------------------------------------------------------------------
# Test 6: stdlib-only (no tabulate import)
# ---------------------------------------------------------------------------


def test_stdlib_only():
    """Script source contains no tabulate import of any form."""
    source = _script_path.read_text()
    assert "import tabulate" not in source, "found 'import tabulate' in script"
    assert "from tabulate" not in source, "found 'from tabulate' in script"


# ---------------------------------------------------------------------------
# Test 7: --update-baseline roundtrip writes correct _meta
# ---------------------------------------------------------------------------


def test_update_baseline_roundtrip(tmp_path):
    """--update-baseline writes a baseline JSON with all 4 required _meta fields."""
    cur_file = tmp_path / "metrics_current.json"
    base_file = tmp_path / "golden_metrics.json"
    cur_data = {
        "tests/test_hw.py::test_two_conv[eyeriss]": {
            "group_latencies_max": 1000.0,
            "mip_gap": None,
            "objective": 10.0,
            "solve_time_s": 0.1,
            "total_latency": 1000.0,
        },
    }
    cur_file.write_text(json.dumps(cur_data, indent=2))
    result = subprocess.run(
        [
            sys.executable,
            str(_script_path),
            "--current",
            str(cur_file),
            "--baseline",
            str(base_file),
            "--update-baseline",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert base_file.exists(), "baseline file not written"
    base = json.loads(base_file.read_text())
    assert "_meta" in base, "missing _meta key"
    meta = base["_meta"]
    for field in ("python_version", "baseline_sha", "baseline_date", "backend"):
        assert field in meta, f"_meta missing field: {field}"
    # Metric body must mirror current (cell key must be present)
    cell_key = "tests/test_hw.py::test_two_conv[eyeriss]"
    assert cell_key in base, "cell key missing from baseline after --update-baseline"


# ---------------------------------------------------------------------------
# Test 8: the GFM delta table is well-formed (header/separator/data cell counts match)
# ---------------------------------------------------------------------------


def test_table_rows_have_equal_cell_counts():
    """Every Markdown table row (header, separator, data) has the same number of
    pipe-delimited cells — otherwise GitHub renders the header jammed into one cell.

    Regression guard for the header-join bug (header used "".join instead of " |".join).
    """
    mod = _load_script()
    rows, captured, total = mod.compute_diffs(_ONE_CELL_CURRENT_SAME, _ONE_CELL_BASE, tol=0.001)
    comment = mod.render_comment(rows, captured, total, _ONE_CELL_BASE["_meta"], tol=0.001)

    # The table rows are the lines that start with "|" (header, separator, data).
    table_lines = [ln for ln in comment.splitlines() if ln.lstrip().startswith("|")]
    assert len(table_lines) >= 3, f"expected header + separator + >=1 data row, got {table_lines}"

    # Cell count for a GFM row = pieces after stripping the outer pipes.
    def _cells(line: str) -> int:
        return len(line.strip().strip("|").split("|"))

    counts = [_cells(ln) for ln in table_lines]
    # Per-workload table: Hardware | total_latency (base → cur) | Δ% | array fill | MAC eff (e2e) | note = 6
    assert all(c == 6 for c in counts), (
        f"all table rows must have 6 cells; got {counts}. Header may be missing inner pipe separators."
    )


# ---------------------------------------------------------------------------
# Test 9: per-workload grouping — one collapsible table per test function
# ---------------------------------------------------------------------------


def test_per_workload_grouping():
    """Cells from two workloads render as two separate collapsible tables titled by workload,
    with the hardware as the row key (fixes the lost-workload-identity problem)."""
    mod = _load_script()
    base = {
        "_meta": _ONE_CELL_BASE["_meta"],
        "tests/test_hw.py::test_hardware_two_conv[simba]": {"total_latency": 4316.0, "objective": 10.0},
        "tests/test_hw.py::test_hardware_swiglu_small[simba]": {"total_latency": 131.0, "objective": 19.0},
    }
    current = {
        "tests/test_hw.py::test_hardware_two_conv[simba]": {"total_latency": 4316.0, "objective": 10.0},
        "tests/test_hw.py::test_hardware_swiglu_small[simba]": {"total_latency": 131.0, "objective": 19.0},
    }
    rows, captured, total = mod.compute_diffs(current, base, tol=0.001)
    comment = mod.render_comment(rows, captured, total, base["_meta"], tol=0.001)
    # Two <details> blocks, one per workload, each titled by the (test_-stripped) function name.
    assert comment.count("<details>") == 2, f"expected one table per workload:\n{comment}"
    assert "hardware_two_conv —" in comment and "hardware_swiglu_small —" in comment
    # Hardware is the row key; the workload is no longer ambiguous between the two tables.
    assert "| simba |" in comment


# ---------------------------------------------------------------------------
# Test 10: degenerate (ZigZag-fallback) cells are surfaced and never gate the verdict
# ---------------------------------------------------------------------------


def test_degenerate_cell_surfaced():
    """A current cell flagged degenerate is called out, annotated in its row note and summary,
    but does NOT by itself set the gated FLAGGED verdict (total_latency unchanged)."""
    mod = _load_script()
    base = {
        "_meta": _ONE_CELL_BASE["_meta"],
        "tests/test_hw.py::test_hardware_two_conv[meta_prototype]": {
            "total_latency": 2956432.0,
            "mac_spatial_utilization": None,
            "degenerate": True,
        },
    }
    current = {
        "tests/test_hw.py::test_hardware_two_conv[meta_prototype]": {
            "total_latency": 2956432.0,  # identical -> not a numeric regression
            "mac_spatial_utilization": None,
            "degenerate": True,
        },
    }
    rows, captured, total = mod.compute_diffs(current, base, tol=0.001)
    assert all(r["status"] != "FLAGGED" for r in rows), "degenerate alone must not gate FLAGGED"
    comment = mod.render_comment(rows, captured, total, base["_meta"], tol=0.001)
    assert "degenerate cell(s)" in comment, f"degenerate callout missing:\n{comment}"
    assert "ZigZag fallback" in comment, "degenerate row note missing"
    # Gated verdict still clean (no total_latency change)
    assert "no changes" in comment.lower() or "✅" in comment


def test_workload_hparams_caption():
    """The per-workload hyperparameter string renders as a code-span caption under the summary
    (code span, not italics, so the underscores in seq_len/embedding_dim are shown literally)."""
    mod = _load_script()
    hp = "seq_len=256, embedding_dim=2048, hidden_dim=8192, bf16; layer-fused tiles seq=16/embedding=128/hidden=32"
    base = {
        "_meta": _ONE_CELL_BASE["_meta"],
        "tests/test_hw.py::test_hardware_swiglu[simba]": {"total_latency": 100.0, "workload_hparams": hp},
    }
    rows, captured, total = mod.compute_diffs(base, base, tol=0.001)
    comment = mod.render_comment(rows, captured, total, base["_meta"], tol=0.001)
    assert f"`{hp}`" in comment, f"hparams code-span caption missing:\n{comment}"


def test_end_to_end_mac_utilization_column():
    """The end-to-end MAC utilization renders in its own column (sub-1% kept to 2 decimals), and the
    column legend is present so it is not confused with the per-layer array-fill column."""
    mod = _load_script()
    base = {
        "_meta": _ONE_CELL_BASE["_meta"],
        "tests/test_hw.py::test_hardware_swiglu[simba]": {
            "total_latency": 1000.0,
            "mac_spatial_utilization": 0.683,  # array fill (dataflow quality)
            "end_to_end_mac_utilization": 0.0034,  # true chip utilization (mostly-idle mesh)
        },
    }
    rows, captured, total = mod.compute_diffs(base, base, tol=0.001)
    comment = mod.render_comment(rows, captured, total, base["_meta"], tol=0.001)
    assert "MAC eff (e2e)" in comment, "e2e column header missing"
    assert "68%" in comment and "0.34%" in comment, f"array-fill / e2e values not both rendered:\n{comment}"
    assert "**Columns:**" in comment, "column legend missing"


def test_low_utilization_note():
    """A non-degenerate cell with very low MAC utilization gets a 'low array utilization' note
    (e.g. a tiny workload on an oversized array) — distinct from the degenerate fallback note."""
    mod = _load_script()
    base = {
        "_meta": _ONE_CELL_BASE["_meta"],
        "tests/test_hw.py::test_hardware_two_conv[fusemax]": {
            "total_latency": 186148.0,
            "mac_spatial_utilization": 0.0140625,
            "degenerate": False,
        },
    }
    rows, captured, total = mod.compute_diffs(base, base, tol=0.001)
    comment = mod.render_comment(rows, captured, total, base["_meta"], tol=0.001)
    assert "low array utilization" in comment, f"low-util note missing:\n{comment}"
    assert "degenerate cell(s)" not in comment, "low util must not be reported as degenerate"
