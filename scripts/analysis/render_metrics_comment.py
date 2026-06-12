#!/usr/bin/env python3
"""Diff metrics_current.json against tests/golden_metrics.json and render a Markdown PR comment."""

import argparse
import datetime
import json
import pathlib
import platform
import subprocess
import sys

METRICS_GATED = ("total_latency",)
METRICS_INFO = ("objective", "solve_time_s")
METRICS_ALL = METRICS_GATED + METRICS_INFO
META_KEY = "_meta"
MARKER = "<!-- stream-aie-metrics-regression-guard-v1 -->"


def load_json(path: pathlib.Path) -> dict:
    """Read and parse a JSON file; raise FileNotFoundError with a clear message."""
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text())


def _is_flagged(cur: float | None, base: float | None, tol: float) -> bool:
    """Return True if the cell should be flagged as changed (38-DETERMINISM-RECORD §7)."""
    if base is None or cur is None:
        return False  # NO_DATA — never a numeric regression
    if base == 0.0:
        return cur != 0.0  # baseline-zero guard: flag if current is nonzero
    return abs(cur - base) / abs(base) > tol


def compute_diffs(
    current: dict,
    baseline: dict,
    tol: float,
) -> tuple[list[dict], int, int]:
    """Diff current vs baseline metrics, skipping _meta.

    Returns (rows, captured_count, total_count).

    Each row dict has:
      - node_id: str
      - status: "OK" | "FLAGGED" | "NEW" | "NO_DATA"
        ("REMOVED" is a RESERVED future label — NOT emitted in v1)
      - per-metric sub-dicts {baseline, current, delta, delta_pct, flagged}
        for each metric in METRICS_ALL (only for OK/FLAGGED rows)

    Classification:
      - baseline-only cell (in baseline, NOT in current) -> "NO_DATA"
        (a capture failure / partial -x run; never a numeric regression in v1)
      - current-only cell (in current, NOT in baseline) -> "NEW"
      - both present -> "FLAGGED" if any METRICS_GATED metric exceeds tol, else "OK"

    captured = baseline cells that also appear in current (actually measured this run).
    total = baseline cells (excluding _meta).
    """
    baseline_keys = set(baseline) - {META_KEY}
    current_keys = set(current) - {META_KEY}
    all_keys = sorted(baseline_keys | current_keys)

    total = len(baseline_keys)
    captured = len(baseline_keys & current_keys)

    rows: list[dict] = []
    for node_id in all_keys:
        in_base = node_id in baseline_keys
        in_cur = node_id in current_keys

        if in_base and not in_cur:
            # Baseline-only: capture failure / partial run — not a regression in v1
            rows.append({"node_id": node_id, "status": "NO_DATA"})
            continue

        if in_cur and not in_base:
            # Current-only: newly added cell
            rows.append({"node_id": node_id, "status": "NEW"})
            continue

        # Both present — compute per-metric sub-dicts
        base_entry = baseline[node_id]
        cur_entry = current[node_id]
        metrics: dict[str, dict] = {}
        any_flagged = False

        for m in METRICS_ALL:
            base_val = base_entry.get(m)
            cur_val = cur_entry.get(m)
            if base_val is None and cur_val is None:
                delta = None
                delta_pct = None
            elif base_val is None or cur_val is None:
                delta = None
                delta_pct = None
            else:
                delta = cur_val - base_val
                delta_pct = (delta / abs(base_val) * 100.0) if base_val != 0.0 else None

            gated = m in METRICS_GATED
            flagged = gated and _is_flagged(cur_val, base_val, tol)
            if flagged:
                any_flagged = True

            metrics[m] = {
                "baseline": base_val,
                "current": cur_val,
                "delta": delta,
                "delta_pct": delta_pct,
                "flagged": flagged,
            }

        status = "FLAGGED" if any_flagged else "OK"
        rows.append({"node_id": node_id, "status": status, **metrics})

    return rows, captured, total


def _fmt_val(v: float | None, precision: int = 2) -> str:
    """Format a numeric value for table display."""
    if v is None:
        return "NO DATA"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _fmt_delta(delta: float | None, delta_pct: float | None, precision: int = 2) -> tuple[str, str]:
    """Format delta and delta_pct with direction arrows."""
    if delta is None or delta_pct is None:
        return "—", "—"
    sign = "+" if delta >= 0 else ""
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "")
    return f"{sign}{delta:.{precision}f}", f"{arrow}{sign}{delta_pct:.2f}%"


def _metric_precision(metric: str) -> int:
    """Display precision per metric. solve_time_s is sub-0.01s wall-time, so 2 decimals
    round it to 0.00 while its Δ% (computed on the true value) reads ~95% — internally
    contradictory. 4 decimals keeps the value, Δ, and Δ% consistent."""
    return 4 if metric == "solve_time_s" else 2


def _short_node_id(node_id: str) -> str:
    """Truncate a long pytest node ID to its [param] suffix for readability."""
    bracket = node_id.rfind("[")
    if bracket != -1:
        return node_id[bracket:]
    # Fallback: keep the last component after ::
    colon = node_id.rfind("::")
    if colon != -1:
        return node_id[colon + 2 :]
    return node_id


def render_comment(  # noqa: PLR0912, PLR0915
    rows: list[dict],
    captured: int,
    total: int,
    meta: dict | None,
    tol: float,
) -> str:
    """Assemble the full Markdown comment body per D-07.

    Order: marker, header, status line, X-of-Y banner, provenance, collapsible table, regen hint.
    Always returns a str (never raises).
    """
    lines: list[str] = []

    # (a) Stable HTML marker
    lines.append(MARKER)
    lines.append("")

    # (b) Header
    lines.append("## Stream AIE Metrics Regression Guard")
    lines.append("")

    # (c) Status line
    flagged_rows = [r for r in rows if r.get("status") == "FLAGGED"]
    if flagged_rows:
        node_ids_flagged = ", ".join(_short_node_id(r["node_id"]) for r in flagged_rows)
        tol_pct = tol * 100.0
        lines.append(
            f"**⚠ {len(flagged_rows)} cell(s) flagged** (total_latency > {tol_pct:.1f}% tol): {node_ids_flagged}"
        )
    else:
        lines.append("**✅ no changes — all gated cells within tol**")
    lines.append("")

    # (d) X-of-Y captured banner
    no_data_count = sum(1 for r in rows if r.get("status") == "NO_DATA")
    banner = f"**{captured} of {total} cells captured**"
    if no_data_count > 0:
        banner += f" ({no_data_count} cell(s) reported NO DATA — not counted as regressions)"
    lines.append(banner)
    lines.append("")

    # (e) Provenance line from _meta + mip_gap note
    if meta:
        sha = meta.get("baseline_sha", "unknown")
        date = meta.get("baseline_date", "unknown")
        pyver = meta.get("python_version", "unknown")
        backend = meta.get("backend", "unknown")
        lines.append(f"**Provenance:** baseline `{sha}` | date `{date}` | Python `{pyver}` | backend `{backend}`")
    else:
        lines.append("**Provenance:** unknown")
    lines.append("**Note:** mip_gap: null — OR-Tools GSCIP")
    lines.append("")

    # (f) Delta table in collapsible <details>
    lines.append("<details>")
    lines.append("<summary><strong>Metric delta table (click to expand)</strong></summary>")
    lines.append("")

    # Table header: node-id + 3 metrics × 4 sub-columns
    header_parts = ["| Node ID"]
    sep_parts = ["|---"]
    for m in METRICS_ALL:
        label = "objective (lex-last-phase)" if m == "objective" else m
        header_parts.append(f" {label} (base)")
        header_parts.append(f" {label} (cur)")
        header_parts.append(f" {label} Δ")
        header_parts.append(f" {label} Δ%")
        sep_parts.extend(["|---", "|---", "|---", "|---"])
    lines.append(" |".join(header_parts) + " |")
    lines.append("".join(sep_parts) + "|")

    # Table rows
    for row in rows:
        status = row.get("status", "OK")
        node_short = _short_node_id(row["node_id"])
        if status == "NO_DATA":
            row_cells = f"| {node_short} |"
            for _ in METRICS_ALL:
                row_cells += " NO DATA | NO DATA | — | — |"
            lines.append(row_cells)
            continue
        if status == "NEW":
            row_cells = f"| {node_short} (NEW) |"
            for m in METRICS_ALL:
                cur_entry = row.get(m, {})
                cur_val = cur_entry.get("current") if isinstance(cur_entry, dict) else None
                row_cells += f" — | {_fmt_val(cur_val, _metric_precision(m))} | — | — |"
            lines.append(row_cells)
            continue

        # OK or FLAGGED
        row_parts = [f"| {node_short}"]
        for m in METRICS_ALL:
            m_data = row[m]
            prec = _metric_precision(m)
            base_str = _fmt_val(m_data["baseline"], prec)
            cur_str = _fmt_val(m_data["current"], prec)
            d_str, dp_str = _fmt_delta(m_data["delta"], m_data["delta_pct"], prec)
            if m_data.get("flagged"):
                # Prepend warning to the delta% cell
                dp_str = f"⚠ {dp_str}"
            row_parts.extend([f" {base_str}", f" {cur_str}", f" {d_str}", f" {dp_str}"])
        lines.append(" |".join(row_parts) + " |")

    lines.append("")
    lines.append("</details>")
    lines.append("")

    # (g) Regen hint
    lines.append("**To regenerate baseline:** `python scripts/analysis/render_metrics_comment.py --update-baseline`")
    lines.append("")

    return "\n".join(lines)


def update_baseline(
    current_path: pathlib.Path,
    baseline_path: pathlib.Path,
) -> None:
    """Read current_path, attach fresh _meta, write baseline_path with sort_keys+indent.

    _meta fields: python_version, baseline_sha (git rev-parse HEAD, fallback "unknown"),
                  baseline_date (today ISO), backend ("ortools_gscip").
    """
    current = load_json(current_path)

    # Get baseline_sha from git; fall back to "unknown" on any failure
    baseline_sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            baseline_sha = result.stdout.strip()
    except Exception:  # noqa: BLE001
        baseline_sha = "unknown"

    meta = {
        "backend": "ortools_gscip",
        "baseline_date": datetime.date.today().isoformat(),
        "baseline_sha": baseline_sha,
        "python_version": platform.python_version(),
    }

    baseline_data = {**current, META_KEY: meta}
    baseline_path.write_text(json.dumps(baseline_data, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current", default="metrics_current.json", type=pathlib.Path)
    ap.add_argument("--baseline", default="tests/golden_metrics.json", type=pathlib.Path)
    ap.add_argument("--output", default=None, type=pathlib.Path)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--tolerance", default=0.001, type=float)
    args = ap.parse_args()

    if args.update_baseline:
        update_baseline(args.current, args.baseline)
        sys.exit(0)

    current = load_json(args.current)
    baseline = load_json(args.baseline)
    rows, captured, total = compute_diffs(current, baseline, args.tolerance)
    meta = baseline.get(META_KEY)
    comment = render_comment(rows, captured, total, meta, args.tolerance)

    if args.output:
        args.output.write_text(comment)
    else:
        print(comment)

    sys.exit(0)


if __name__ == "__main__":
    main()
