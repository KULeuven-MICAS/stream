"""Generate comparison plots for cross-backend TETRA verification results.

Produces:
  outputs/backend_comparison.png — 2x2 panel with objective + solve time for GEMM and SwiGLU
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data from cross-backend verification runs (2026-05-07)
# ---------------------------------------------------------------------------

WORKLOADS = {
    "GEMM\n(256x8192x2048)": {
        "Gurobi": {"objective": 48_730_624, "solve_time": 11.27},
        "GSCIP": {"objective": 48_730_624, "solve_time": 14.25},
        "HiGHS": {"objective": 48_730_623, "solve_time": 13.36},
    },
    "SwiGLU\n(256x512x2048)": {
        "Gurobi": {"objective": 9_396_480, "solve_time": 27.37},
        "GSCIP": {"objective": 9_396_480, "solve_time": 110.68},
        "HiGHS": {"objective": 9_396_480, "solve_time": 64.85},
    },
}

SOLVERS = ["Gurobi", "GSCIP", "HiGHS"]
COLORS = {"Gurobi": "#E24A33", "GSCIP": "#348ABD", "HiGHS": "#988ED5"}
HATCHES = {"Gurobi": "", "GSCIP": "//", "HiGHS": ".."}


def main():  # noqa: PLR0915
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "TETRA Cross-Backend Solver Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    workload_names = list(WORKLOADS.keys())

    # --- Row 1: Objective values (latency_total) ---
    for col_idx, wl_name in enumerate(workload_names):
        ax = axes[0, col_idx]
        data = WORKLOADS[wl_name]
        objectives = [data[s]["objective"] for s in SOLVERS]

        x = np.arange(len(SOLVERS))
        bars = ax.bar(
            x,
            objectives,
            width=0.5,
            color=[COLORS[s] for s in SOLVERS],
            hatch=[HATCHES[s] for s in SOLVERS],
            edgecolor="black",
            linewidth=0.8,
        )

        # Add value labels
        for bar, val in zip(bars, objectives, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(f"{wl_name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latency Total (cycles)" if col_idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(SOLVERS, fontsize=10)

        # Tight y-range to show any differences
        min_obj = min(objectives)
        max_obj = max(objectives)
        margin = max(max_obj * 0.002, 1)
        ax.set_ylim(min_obj - margin * 50, max_obj + margin * 200)

        # Add "identical" annotation
        if max_obj - min_obj <= 1:
            ax.annotate(
                "All identical",
                xy=(0.5, 0.02),
                xycoords="axes fraction",
                ha="center",
                fontsize=10,
                color="green",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )

    # --- Row 2: Solve times ---
    for col_idx, wl_name in enumerate(workload_names):
        ax = axes[1, col_idx]
        data = WORKLOADS[wl_name]
        times = [data[s]["solve_time"] for s in SOLVERS]
        gurobi_time = data["Gurobi"]["solve_time"]

        x = np.arange(len(SOLVERS))
        bars = ax.bar(
            x,
            times,
            width=0.5,
            color=[COLORS[s] for s in SOLVERS],
            hatch=[HATCHES[s] for s in SOLVERS],
            edgecolor="black",
            linewidth=0.8,
        )

        # Add value labels with speedup/slowdown relative to Gurobi
        for bar, t, solver in zip(bars, times, SOLVERS, strict=False):
            ratio = t / gurobi_time
            label = f"{t:.1f}s"
            if solver != "Gurobi":
                label += f"\n({ratio:.1f}x)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(f"{wl_name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Solve Time (seconds)" if col_idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(SOLVERS, fontsize=10)
        ax.set_ylim(0, max(times) * 1.25)

    # Row labels
    axes[0, 0].annotate(
        "Optimization Result\n(lower is better)",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        fontsize=11,
        ha="center",
        va="center",
        rotation=90,
        fontweight="bold",
        color="#444444",
    )
    axes[1, 0].annotate(
        "Solver Runtime\n(lower is better)",
        xy=(-0.35, 0.5),
        xycoords="axes fraction",
        fontsize=11,
        ha="center",
        va="center",
        rotation=90,
        fontweight="bold",
        color="#444444",
    )

    plt.tight_layout(rect=[0.05, 0.02, 1, 0.95])
    out_path = "outputs/backend_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # --- Also create a summary table plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.axis("off")
    ax2.set_title(
        "Cross-Backend Verification Summary",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    table_data = []
    for wl_name, data in WORKLOADS.items():
        wl_short = wl_name.replace("\n", " ")
        for solver in SOLVERS:
            d = data[solver]
            ratio = d["solve_time"] / data["Gurobi"]["solve_time"]
            obj_match = (
                "identical"
                if abs(d["objective"] - data["Gurobi"]["objective"]) <= 1
                else f"{abs(d['objective'] - data['Gurobi']['objective']):.0f} diff"
            )
            table_data.append(
                [
                    wl_short,
                    solver,
                    f"{d['objective']:,.0f}",
                    obj_match,
                    f"{d['solve_time']:.1f}s",
                    f"{ratio:.2f}x" if solver != "Gurobi" else "baseline",
                ]
            )

    table = ax2.table(
        cellText=table_data,
        colLabels=["Workload", "Solver", "Objective", "vs Gurobi", "Solve Time", "Relative"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Color header
    for j in range(6):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color solver cells
    for i, row in enumerate(table_data, start=1):
        solver = row[1]
        if solver in COLORS:
            table[i, 1].set_facecolor(COLORS[solver] + "33")
        # Green for "identical"
        if row[3] == "identical":
            table[i, 3].set_facecolor("#C6EFCE")

    out_path2 = "outputs/backend_summary_table.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
