#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid", context="talk")

# =========================
# DEFAULT INPUT CONSTANTS
# (you can override via CLI)
# =========================
M, K, N = 128, 128, 128
NB_ROWS_SINGLE_COL = 2
BASE_A_DIR = f"outputs/single_core-gemm_{M}_{K}_{N}-1_row_1_col/traces"
BASE_B_DIR = f"outputs/single_col-gemm_{M}_{K}_{N}-{NB_ROWS_SINGLE_COL}_row_1_col/traces"
LABEL_A = "single_core"
LABEL_B = "single_col"

# Output plot folder
PLOT_OUT_DIR = "outputs/plots"
PLOT_FILENAME = "trace_vs_wallclock_gmacs.png"


def read_aggregate_perf_gmacs(base_dir: str) -> tuple[float | None, str]:
    """
    Read aggregate_perf.json and return 'best' gmacs_per_second.
    Returns (value, path_used). If missing/unreadable, returns (None, path).
    """
    agg_path = os.path.join(base_dir, "aggregate_perf.json")
    try:
        with open(agg_path) as f:
            data = json.load(f)
        best = data.get("best", None)
        if best is None:
            # Fallback: take max over per_iteration
            per_iter = data.get("per_iteration", [])
            vals = [float(x.get("gmacs_per_second", 0.0)) for x in per_iter]
            return (max(vals) if vals else None, agg_path)
        return (float(best.get("gmacs_per_second", 0.0)), agg_path)
    except Exception as e:
        print(f"Warning: failed reading {agg_path}: {e}")
        return (None, agg_path)


def read_wallclock_best_gmacs(base_dir: str) -> tuple[float | None, str]:
    """
    Read wall_clock_summary.json and return the max 'gmacs_per_second' over (avg,min,max).
    Returns (value, path_used). If missing/unreadable, returns (None, path).
    """
    wc_path = os.path.join(base_dir, "wall_clock_summary.json")
    try:
        with open(wc_path) as f:
            data = json.load(f)
        stats = data.get("stats", {})
        vals = []
        for k in ("avg", "min", "max"):
            if k in stats and "gmacs_per_second" in stats[k]:
                vals.append(float(stats[k]["gmacs_per_second"]))
        return (max(vals) if vals else None, wc_path)
    except Exception as e:
        print(f"Warning: failed reading {wc_path}: {e}")
        return (None, wc_path)


def bar_labels(ax, bars):
    """Write numeric value on top of each bar."""
    for b in bars:
        height = b.get_height()
        if np.isnan(height):
            continue
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )


def plot_two_panels(
    label_a: str,
    label_b: str,
    trace_a: float | None,
    trace_b: float | None,
    wc_a: float | None,
    wc_b: float | None,
    out_dir: str,
    out_name: str,
):
    os.makedirs(out_dir, exist_ok=True)

    # Prepare values, replace None with np.nan so matplotlib handles missing gracefully
    vals_trace = [trace_a if trace_a is not None else np.nan, trace_b if trace_b is not None else np.nan]
    vals_wc = [wc_a if wc_a is not None else np.nan, wc_b if wc_b is not None else np.nan]

    labels = [label_a, label_b]
    colors = sns.color_palette("deep", n_colors=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Left subplot: trace GMAC/s (higher is typical)
    ax0 = axes[0]
    x0 = np.arange(len(labels))
    bars0 = ax0.bar(x0, vals_trace, color=[colors[0], colors[1]])
    ax0.set_xticks(x0)
    ax0.set_xticklabels(labels, rotation=0)
    ax0.set_title("Trace throughput (GMAC/s)")
    ax0.set_ylabel("GMAC/s")

    bar_labels(ax0, bars0)

    # Right subplot: wall-clock GMAC/s
    ax1 = axes[1]
    x1 = np.arange(len(labels))
    bars1 = ax1.bar(x1, vals_wc, color=[colors[0], colors[1]])
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_title("Wall-clock throughput (GMAC/s)")

    bar_labels(ax1, bars1)

    # Single legend for both subplots
    # Use proxy artists with labels
    proxy = [
        plt.Rectangle((0, 0), 1, 1, color=colors[0]),
        plt.Rectangle((0, 0), 1, 1, color=colors[1]),
    ]
    fig.legend(proxy, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])  # leave space for legend

    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[write] {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare best GMAC/s from trace (aggregate_perf.json) vs wall-clock (wall_clock_summary.json) "
        "for two experiment directories."
    )
    p.add_argument(
        "--base_a",
        type=str,
        default=BASE_A_DIR,
        help="Base directory A pointing to the 'traces' folder (default: constant in script).",
    )
    p.add_argument(
        "--base_b",
        type=str,
        default=BASE_B_DIR,
        help="Base directory B pointing to the 'traces' folder (default: constant in script).",
    )
    p.add_argument("--label_a", type=str, default=LABEL_A, help="Label for directory A.")
    p.add_argument("--label_b", type=str, default=LABEL_B, help="Label for directory B.")
    p.add_argument("--out_dir", type=str, default=PLOT_OUT_DIR, help="Output directory for the plot.")
    p.add_argument("--out_name", type=str, default=PLOT_FILENAME, help="Output filename for the plot.")
    return p.parse_args()


def main():
    args = parse_args()

    # A: read values
    trace_a, trace_a_path = read_aggregate_perf_gmacs(args.base_a)
    wc_a, wc_a_path = read_wallclock_best_gmacs(args.base_a)
    print(f"[info] A trace file:      {trace_a_path} -> {trace_a if trace_a is not None else 'N/A'} GMAC/s")
    print(f"[info] A wall-clock file: {wc_a_path} -> {wc_a if wc_a is not None else 'N/A'} GMAC/s")

    # B: read values
    trace_b, trace_b_path = read_aggregate_perf_gmacs(args.base_b)
    wc_b, wc_b_path = read_wallclock_best_gmacs(args.base_b)
    print(f"[info] B trace file:      {trace_b_path} -> {trace_b if trace_b is not None else 'N/A'} GMAC/s")
    print(f"[info] B wall-clock file: {wc_b_path} -> {wc_b if wc_b is not None else 'N/A'} GMAC/s")

    # Plot
    plot_two_panels(
        label_a=args.label_a,
        label_b=args.label_b,
        trace_a=trace_a,
        trace_b=trace_b,
        wc_a=wc_a,
        wc_b=wc_b,
        out_dir=args.out_dir,
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
