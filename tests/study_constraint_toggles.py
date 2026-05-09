"""Constraint toggle study script for TETRA.

Enumerates all 16 combinations of 4 boolean constraint groups
(memory_capacity, object_fifo_depth, buffer_descriptors, dma_channels),
runs the optimizer for each combination, and produces a terminal comparison
table plus matplotlib plots showing which constraints have the largest impact
on solution quality and solve time.

Usage:
    PYTHONPATH=. python tests/study_constraint_toggles.py --workload gemm
    PYTHONPATH=. python tests/study_constraint_toggles.py --workload swiglu
    PYTHONPATH=. python tests/study_constraint_toggles.py --workload gemm --output-yaml /tmp/results.yaml
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
import tempfile
import time
import traceback
import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np

from stream.api import optimize_allocation_co
from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping
from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.opt.solver import ConstraintSelection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCELERATOR = os.path.join(
    os.path.dirname(__file__),
    "../stream/inputs/aie/hardware/whole_array_strix.yaml",
)

CONSTRAINT_FIELDS = ["memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels"]
CONSTRAINT_LABELS = {
    "memory_capacity": "Memory Capacity",
    "object_fifo_depth": "Object FIFO Depth",
    "buffer_descriptors": "Buffer Descriptors",
    "dma_channels": "DMA Channels",
}

# Color scheme for plots
COLOR_BASELINE = "#2ecc71"  # green — all-enabled baseline
COLOR_DEFAULT = "#3498db"  # blue — other combinations
COLOR_FAILED = "#e74c3c"  # red — failed runs


# ---------------------------------------------------------------------------
# Combination enumeration
# ---------------------------------------------------------------------------


def _all_combinations() -> list[tuple[str, ConstraintSelection]]:
    """Generate all 16 constraint toggle combinations with human-readable labels.

    Iterates over the Cartesian product of {True, False} for the 4 constraint
    fields. The first combination (all True) is the baseline.

    Returns:
        List of (label, ConstraintSelection) tuples, length 16.
        Label is the human-readable comma-joined list of enabled groups,
        or "None (all disabled)" when all fields are False.
    """
    combos = []
    for bits in itertools.product([True, False], repeat=4):
        cs = ConstraintSelection(
            memory_capacity=bits[0],
            object_fifo_depth=bits[1],
            buffer_descriptors=bits[2],
            dma_channels=bits[3],
        )
        enabled = [CONSTRAINT_LABELS[f] for f, on in zip(CONSTRAINT_FIELDS, bits, strict=True) if on]
        label = " + ".join(enabled) if enabled else "None (all disabled)"
        combos.append((label, cs))
    return combos


# ---------------------------------------------------------------------------
# Pipeline helpers (follow verify_backends.py pattern exactly)
# ---------------------------------------------------------------------------


def _run_gemm_pipeline(  # noqa: PLR0913, N803
    M: int,
    K: int,
    N: int,
    m: int,
    k: int,
    n: int,
    in_dtype: str,
    out_dtype: str,
    rows: int,
    cols: int,
    trace_size: int,
    npu: str,
    backend: str,
    output_path: str,
    constraint_selection: ConstraintSelection | None = None,
):
    """Run the TETRA GEMM optimization pipeline.

    Returns:
        ctx: Stage context with ``scheduler`` accessible via ``ctx.get``.
    """
    workload_path = make_gemm_workload(M, K, N, in_dtype, out_dtype)
    mapping_path = make_gemm_mapping(M, K, N, m, k, n, nb_rows_to_use=rows, nb_cols_to_use=cols)

    hw_name = ACCELERATOR.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    experiment_id = f"{hw_name}-{wl_name}-{rows}_row_{cols}_col"

    return optimize_allocation_co(
        hardware=ACCELERATOR,
        workload=workload_path,
        mapping=mapping_path,
        experiment_id=experiment_id,
        output_path=output_path,
        skip_if_exists=False,
        trace_size=trace_size,
        nb_cols_to_use=cols,
        npu=npu,
        backend=backend,
        constraint_selection=constraint_selection,
    )


def _run_swiglu_pipeline(  # noqa: PLR0913
    seq_len: int,
    embedding_dim: int,
    hidden_dim: int,
    in_dtype: str,
    out_dtype: str,
    rows: int,
    cols: int,
    trace_size: int,
    npu: str,
    backend: str,
    output_path: str,
    constraint_selection: ConstraintSelection | None = None,
):
    """Run the TETRA SwiGLU optimization pipeline.

    Returns:
        ctx: Stage context with ``scheduler`` accessible via ``ctx.get``.
    """
    workload_path = make_swiglu_workload(seq_len, embedding_dim, hidden_dim, in_dtype, out_dtype)
    mapping_path = make_swiglu_mapping(
        seq_len,
        embedding_dim,
        hidden_dim,
        True,  # last_gemm_down
        32,  # seq_len_tile_size (default from reference tests)
        32,  # embedding_tile_size
        64,  # hidden_tile_size
    )

    hw_name = ACCELERATOR.split("/")[-1].split(".")[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    experiment_id = f"{hw_name}-{wl_name}-{rows}_row_{cols}_col"

    return optimize_allocation_co(
        hardware=ACCELERATOR,
        workload=workload_path,
        mapping=mapping_path,
        experiment_id=experiment_id,
        output_path=output_path,
        skip_if_exists=False,
        trace_size=trace_size,
        nb_cols_to_use=cols,
        npu=npu,
        backend=backend,
        constraint_selection=constraint_selection,
    )


def _extract_latency_total(ctx) -> float:
    """Extract ``latency_total`` (solver objective) from a completed pipeline context."""
    scheduler = ctx.get("scheduler")
    return float(scheduler.latency_total)


# ---------------------------------------------------------------------------
# Run single combination
# ---------------------------------------------------------------------------


def _run_single(
    workload: str,
    label: str,
    cs: ConstraintSelection,
    args: argparse.Namespace,
    output_root: str,
) -> dict:
    """Run a single constraint combination and return a result dict.

    Returns:
        dict with keys:
            label (str): human-readable combination name
            constraint_selection (dict): field -> bool mapping
            status (str): "OPTIMAL" or "ERROR: <message>"
            objective (float | None): solver objective value, None on failure
            solve_time_s (float): wall-clock solve time in seconds
    """
    result: dict = {
        "label": label,
        "constraint_selection": {
            "memory_capacity": cs.memory_capacity,
            "object_fifo_depth": cs.object_fifo_depth,
            "buffer_descriptors": cs.buffer_descriptors,
            "dma_channels": cs.dma_channels,
        },
        "status": "FAILED",
        "objective": None,
        "solve_time_s": 0.0,
    }

    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if workload == "gemm":
                ctx = _run_gemm_pipeline(
                    M=args.M,
                    K=args.K,
                    N=args.N,
                    m=args.m,
                    k=args.k,
                    n=args.n,
                    in_dtype=args.in_dtype,
                    out_dtype=args.out_dtype,
                    rows=args.rows,
                    cols=args.cols,
                    trace_size=args.trace_size,
                    npu=args.npu,
                    backend=args.backend,
                    output_path=tmpdir,
                    constraint_selection=cs,
                )
            else:
                ctx = _run_swiglu_pipeline(
                    seq_len=args.seq_len,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    in_dtype=args.in_dtype,
                    out_dtype=args.out_dtype,
                    rows=args.rows,
                    cols=args.cols,
                    trace_size=args.trace_size,
                    npu=args.npu,
                    backend=args.backend,
                    output_path=tmpdir,
                    constraint_selection=cs,
                )
        result["objective"] = _extract_latency_total(ctx)
        result["status"] = "OPTIMAL"
    except Exception as exc:
        result["status"] = f"ERROR: {str(exc)[:100]}"
        traceback.print_exc(file=sys.stderr)

    result["solve_time_s"] = round(time.perf_counter() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Run all 16 combinations
# ---------------------------------------------------------------------------


def run_study(args: argparse.Namespace) -> list[dict]:
    """Run all 16 constraint toggle combinations and return result dicts.

    Iterates over all combinations from _all_combinations(), prints progress,
    and collects results.

    Returns:
        List of 16 result dicts (one per combination).
    """
    combos = _all_combinations()
    total = len(combos)
    results = []

    print("\n=== Constraint Toggle Study ===")
    print(f"Workload: {args.workload.upper()}")
    print(f"Backend: {args.backend}")
    print()

    for i, (label, cs) in enumerate(combos, start=1):
        print(f'[{i}/{total}] Running "{label}" ...', end=" ", flush=True)
        result = _run_single(args.workload, label, cs, args, output_root="")
        status_str = result["status"]
        obj_str = f"obj={result['objective']:,.0f}" if result["objective"] is not None else "obj=N/A"
        print(f"{status_str} {obj_str} t={result['solve_time_s']}s")
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Terminal table output
# ---------------------------------------------------------------------------


def _print_results_table(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout.

    Columns:
        # | Constraints Enabled | Objective | Delta % | Solve Time (s)

    The baseline is the first result (all constraints enabled). All other rows
    show delta relative to the baseline objective.
    """
    # Find baseline objective (first result = all enabled)
    baseline_obj: float | None = results[0]["objective"] if results else None

    print()
    print("=== Constraint Toggle Study ===")
    workload_name = results[0].get("workload", "unknown") if results else "unknown"
    print(f"Workload: {workload_name}")
    print()

    col_idx = 4
    col_label = 60
    col_obj = 15
    col_delta = 10
    col_time = 14

    header = (
        f"{'#':>{col_idx}}  "
        f"{'Constraints Enabled':<{col_label}}  "
        f"{'Objective':>{col_obj}}  "
        f"{'Delta %':>{col_delta}}  "
        f"{'Solve Time (s)':>{col_time}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for i, r in enumerate(results, start=1):
        label = r["label"]
        obj = r["objective"]
        solve_time = r["solve_time_s"]

        if obj is not None:
            obj_str = f"{obj:,.0f}"
        else:
            obj_str = "N/A"

        if obj is not None and baseline_obj is not None and baseline_obj != 0:
            delta_pct = (obj - baseline_obj) / baseline_obj * 100
            delta_str = f"{delta_pct:+.2f}%"
        else:
            delta_str = "N/A"

        time_str = f"{solve_time:.2f}"

        # Truncate label if too long
        if len(label) > col_label:
            label = label[: col_label - 3] + "..."

        print(
            f"{i:>{col_idx}}  "
            f"{label:<{col_label}}  "
            f"{obj_str:>{col_obj}}  "
            f"{delta_str:>{col_delta}}  "
            f"{time_str:>{col_time}}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def _save_yaml(results: list[dict], path: str) -> None:
    """Save results list to YAML file.

    Falls back gracefully if PyYAML is not installed.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        warnings.warn(
            "PyYAML not installed — skipping YAML output. Install with: pip install pyyaml",
            stacklevel=2,
        )
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"Results written to: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_objective_bars(results: list[dict], output_dir: str) -> None:  # noqa: PLR0912, PLR0915
    """Plot horizontal bar chart of objective value per constraint combination.

    Baseline (all enabled, index 0) bar is green, failed bars are red, all
    others are blue. Saved to ``{output_dir}/constraint_study_objective.png``.
    """
    labels = [r["label"] for r in results]
    objectives = [r["objective"] if r["objective"] is not None else 0.0 for r in results]
    colors = []
    for i, r in enumerate(results):
        if r["objective"] is None:
            colors.append(COLOR_FAILED)
        elif i == 0:
            colors.append(COLOR_BASELINE)
        else:
            colors.append(COLOR_DEFAULT)

    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.45 + 1.5)))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, objectives, color=colors, edgecolor="black", linewidth=0.5)

    # Value labels at end of each bar
    for bar, r in zip(bars, results, strict=True):
        if r["objective"] is not None:
            val_str = f"{r['objective']:,.0f}"
            ax.text(
                bar.get_width() * 1.005,
                bar.get_y() + bar.get_height() / 2,
                val_str,
                va="center",
                ha="left",
                fontsize=7,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # index 0 (baseline) at the top
    ax.set_xlabel("Objective (Latency Total, cycles)")
    ax.set_title("Constraint Toggle Study — Objective Value", fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch  # noqa: PLC0415

    legend_elements = [
        Patch(facecolor=COLOR_BASELINE, edgecolor="black", label="Baseline (all enabled)"),
        Patch(facecolor=COLOR_DEFAULT, edgecolor="black", label="Partial constraints"),
        Patch(facecolor=COLOR_FAILED, edgecolor="black", label="Failed"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "constraint_study_objective.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_solve_time_bars(results: list[dict], output_dir: str) -> None:  # noqa: PLR0912, PLR0915
    """Plot horizontal bar chart of solve time per constraint combination.

    Same color scheme as objective chart. Saved to
    ``{output_dir}/constraint_study_solve_time.png``.
    """
    labels = [r["label"] for r in results]
    times = [r["solve_time_s"] if r["solve_time_s"] is not None else 0.0 for r in results]
    colors = []
    for i, r in enumerate(results):
        if r["objective"] is None:
            colors.append(COLOR_FAILED)
        elif i == 0:
            colors.append(COLOR_BASELINE)
        else:
            colors.append(COLOR_DEFAULT)

    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.45 + 1.5)))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, times, color=colors, edgecolor="black", linewidth=0.5)

    # Value labels
    for bar, t in zip(bars, times, strict=True):
        if t > 0:
            ax.text(
                bar.get_width() * 1.005,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s",
                va="center",
                ha="left",
                fontsize=7,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Solve Time (seconds)")
    ax.set_title("Constraint Toggle Study — Solve Time", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch  # noqa: PLC0415

    legend_elements = [
        Patch(facecolor=COLOR_BASELINE, edgecolor="black", label="Baseline (all enabled)"),
        Patch(facecolor=COLOR_DEFAULT, edgecolor="black", label="Partial constraints"),
        Patch(facecolor=COLOR_FAILED, edgecolor="black", label="Failed"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "constraint_study_solve_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_constraint_heatmap(results: list[dict], output_dir: str) -> None:  # noqa: PLR0912, PLR0915
    """Plot a heatmap showing which constraints are ON/OFF per combination.

    The table has:
        - 16 rows: one per combination (index 0 = baseline at top)
        - 4 columns: one per constraint group (ON = green, OFF = red)
        - 2 extra columns: Objective and Delta %

    Saved to ``{output_dir}/constraint_study_heatmap.png``.
    """
    baseline_obj: float | None = results[0]["objective"] if results else None

    row_labels = [r["label"] for r in results]
    short_labels = [CONSTRAINT_LABELS[f] for f in CONSTRAINT_FIELDS]
    col_labels = short_labels + ["Objective", "Delta %"]

    table_data = []
    cell_colors = []

    for _i, r in enumerate(results):
        cs_dict = r.get("constraint_selection", {})
        obj = r["objective"]

        row = []
        row_colors = []

        # Constraint ON/OFF columns
        for field in CONSTRAINT_FIELDS:
            is_on = cs_dict.get(field, True)
            row.append("ON" if is_on else "OFF")
            row_colors.append("#C6EFCE" if is_on else "#FFC7CE")  # green / red

        # Objective column
        if obj is not None:
            row.append(f"{obj:,.0f}")
        else:
            row.append("FAILED")
        row_colors.append("#DDDDDD")

        # Delta % column
        if obj is not None and baseline_obj is not None and baseline_obj != 0:
            delta_pct = (obj - baseline_obj) / baseline_obj * 100
            delta_str = f"{delta_pct:+.2f}%"
            # Color: green for small delta, red for large
            abs_delta = abs(delta_pct)
            if abs_delta < 1.0:
                delta_color = "#C6EFCE"
            elif abs_delta < 5.0:
                delta_color = "#FFEB9C"
            else:
                delta_color = "#FFC7CE"
        else:
            delta_str = "N/A"
            delta_color = "#DDDDDD"
        row.append(delta_str)
        row_colors.append(delta_color)

        table_data.append(row)
        cell_colors.append(row_colors)

    n_rows = len(results)
    fig_height = max(5, n_rows * 0.38 + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")
    ax.set_title(
        "Constraint Toggle Impact — Which Constraints Matter?",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Style header row
    n_cols = len(col_labels)
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Style row labels column (make baseline row label bold)
    for i in range(n_rows):
        row_label_cell = table[i + 1, -1]  # row labels are at column index -1
        if i == 0:
            row_label_cell.set_text_props(fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "constraint_study_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_results(results: list[dict], output_dir: str) -> None:
    """Generate matplotlib plots for constraint toggle study results.

    Produces 3 PNG files in output_dir:
        - constraint_study_objective.png: horizontal bar chart of objectives
        - constraint_study_solve_time.png: horizontal bar chart of solve times
        - constraint_study_heatmap.png: ON/OFF heatmap with delta column
    """
    os.makedirs(output_dir, exist_ok=True)
    _plot_objective_bars(results, output_dir)
    _plot_solve_time_bars(results, output_dir)
    _plot_constraint_heatmap(results, output_dir)
    print(f"\nPlots saved to: {output_dir}/")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Constraint toggle study for TETRA. "
            "Runs all 16 combinations of 4 boolean constraint groups and "
            "produces a comparison table plus matplotlib plots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--workload",
        type=str,
        choices=["gemm", "swiglu"],
        required=True,
        help="Workload type to study",
    )

    # GEMM-specific arguments
    gemm_group = parser.add_argument_group("GEMM parameters (required when --workload gemm)")
    gemm_group.add_argument("--M", type=int, default=256, help="M dimension")
    gemm_group.add_argument("--K", type=int, default=8192, help="K dimension")
    gemm_group.add_argument("--N", type=int, default=2048, help="N dimension")
    gemm_group.add_argument("--m", type=int, default=32, help="m tile dimension")
    gemm_group.add_argument("--k", type=int, default=32, help="k tile dimension")
    gemm_group.add_argument("--n", type=int, default=32, help="n tile dimension")

    # SwiGLU-specific arguments
    swiglu_group = parser.add_argument_group("SwiGLU parameters (required when --workload swiglu)")
    swiglu_group.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    swiglu_group.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    swiglu_group.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension")

    # Common parameters
    parser.add_argument("--in_dtype", type=str, default="bf16", help="Input data type")
    parser.add_argument("--out_dtype", type=str, default="bf16", help="Output data type")
    parser.add_argument("--rows", type=int, default=4, help="Number of AIE rows")
    parser.add_argument("--cols", type=int, default=8, help="Number of AIE columns")
    parser.add_argument("--trace_size", type=int, default=1048576, help="Trace buffer size")
    parser.add_argument("--npu", type=str, default="npu2", help="NPU type to target")

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        default="ortools_gscip",
        choices=["gurobi", "ortools_gscip", "ortools_highs", "ortools_gurobi"],
        help="Solver backend to use for all combinations",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/constraint_study",
        dest="output_dir",
        help="Directory where plots and YAML are saved",
    )
    parser.add_argument(
        "--output-yaml",
        type=str,
        default=None,
        metavar="PATH",
        dest="output_yaml",
        help="Optional path to write YAML results file",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    results = run_study(args)
    # Attach workload name to results for table header
    for r in results:
        r["workload"] = args.workload
    _print_results_table(results)
    if args.output_yaml:
        _save_yaml(results, args.output_yaml)
    plot_results(results, args.output_dir)
