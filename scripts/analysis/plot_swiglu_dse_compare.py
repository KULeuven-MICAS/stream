#!/usr/bin/env python3
"""Compare SwiGLU DSE sweeps across multiple experiment directories.

Each directory must already have been processed by ``plot_swiglu_dse_sweep.py``
so that ``<dir>/_plots/summary.csv`` exists. Configure the experiments and
their display labels in ``EXPERIMENTS`` at the top of this file.

Outputs (written to ``OUT_DIR``):
    - ``best_overall_bar.{pdf,png}``: single bar per experiment showing the
      global best latency (min over all tile-size combinations).
    - ``best_per_tilesize_box.{pdf,png}``: box plot of best latency per
      tile-size combination, one box per experiment. Whiskers cover the full
      data range so there are no detached outlier dots.
    - ``best_per_tilesize_bar.{pdf,png}``: grouped bar chart over the tile-size
      combinations that succeeded in every experiment, one bar per experiment.
    - ``success_rate_bar.{pdf,png}``: aggregate succeeded vs. failed mapping
      counts per experiment.
    - ``compare_summary.csv``: per-experiment aggregate stats.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------------
# Configure experiments here. Add more (path, label) entries as needed.
# Order in this list determines plot order (left to right / top to bottom).
# ---------------------------------------------------------------------------

EXPERIMENTS: list[tuple[str, str]] = [
    ("outputs/dse-whole_array_strix-swiglu_256_2048_8192-4_row_8_col", "Baseline"),
    (
        "outputs/dse-inf-fifo-depth-whole_array_strix-swiglu_256_2048_8192-4_row_8_col",
        "Infinite OFIFO depth",
    ),
]

OUT_DIR = "outputs/_compare_swiglu_dse"
UNITS = "cycles"  # axis-label units; "" disables the suffix


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    path: str
    label: str
    rows: list[dict]  # raw rows from summary.csv (strings)

    @property
    def best_per_tilesize(self) -> np.ndarray:
        """Best latency per tile-size combination (only combos that succeeded)."""
        out: list[float] = []
        for r in self.rows:
            v = r.get("best_latency", "")
            if v:
                out.append(float(v))
        return np.asarray(out, dtype=float)

    @property
    def best_overall(self) -> float:
        arr = self.best_per_tilesize
        return float(arr.min()) if arr.size else float("nan")

    @property
    def best_combo_label(self) -> str:
        best: tuple[float, str] | None = None
        for r in self.rows:
            v = r.get("best_latency", "")
            if not v:
                continue
            lat = float(v)
            label = f"{r['seq_tile']}x{r['emb_tile']}x{r['hid_tile']}"
            if best is None or lat < best[0]:
                best = (lat, label)
        return best[1] if best else ""

    @property
    def total_mappings(self) -> int:
        return sum(int(r["total_mappings"]) for r in self.rows)

    @property
    def total_succeeded(self) -> int:
        return sum(int(r["succeeded"]) for r in self.rows)

    @property
    def total_failed(self) -> int:
        return sum(int(r["failed"]) for r in self.rows)

    @property
    def success_rate(self) -> float:
        return self.total_succeeded / self.total_mappings if self.total_mappings else 0.0

    def best_for(self, key: tuple[str, str, str]) -> float | None:
        """Best latency for a specific (seq, emb, hid) tile-size key."""
        for r in self.rows:
            if (r["seq_tile"], r["emb_tile"], r["hid_tile"]) == key:
                v = r.get("best_latency", "")
                return float(v) if v else None
        return None


def load_experiment(path: str, label: str) -> Experiment:
    csv_path = os.path.join(path, "_plots", "summary.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Missing summary.csv for experiment '{label}': {csv_path}\nRun plot_swiglu_dse_sweep.py on '{path}' first."
        )
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return Experiment(path=path, label=label, rows=rows)


# ---------------------------------------------------------------------------
# Style (mirrors plot_swiglu_dse_sweep.py)
# ---------------------------------------------------------------------------


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.axisbelow": True,
            "mathtext.fontset": "stix",
        }
    )


def _sci_formatter(powerlimits: tuple[int, int] = (-2, 3)) -> ScalarFormatter:
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits(powerlimits)
    return fmt


def _apply_sci_notation(ax: Axes, axis: str = "y") -> None:
    if axis in ("both", "x"):
        ax.xaxis.set_major_formatter(_sci_formatter())
    if axis in ("both", "y"):
        ax.yaxis.set_major_formatter(_sci_formatter())


def _save_fig(fig: Figure, out_path_no_ext: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path_no_ext}.{ext}")
    plt.close(fig)


def _experiment_colors(n: int) -> list:
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")  # noqa: PLR2004
    return [cmap(i % cmap.N) for i in range(n)]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_best_overall_bar(experiments: list[Experiment], out_path_no_ext: str, units: str) -> None:
    valid = [e for e in experiments if not np.isnan(e.best_overall)]
    if not valid:
        return
    labels = [e.label for e in valid]
    bests = np.array([e.best_overall for e in valid])
    colors = _experiment_colors(len(valid))

    width = max(5.5, 1.4 * len(valid) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 5))
    x = np.arange(len(valid))
    ax.bar(x, bests, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Best latency{units}")
    _apply_sci_notation(ax, axis="y")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)

    # Annotate each bar with the best tile-size combo.
    ymax = bests.max()
    for xi, e, b in zip(x, valid, bests, strict=True):
        ax.text(
            float(xi),
            float(b) + 0.02 * ymax,
            e.best_combo_label,
            ha="center",
            va="bottom",
            fontsize=11,
            color="#333333",
        )
    ax.set_ylim(0, ymax * 1.15)
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def plot_best_per_tilesize_box(experiments: list[Experiment], out_path_no_ext: str, units: str) -> None:
    data = [e.best_per_tilesize for e in experiments]
    if not any(d.size for d in data):
        return
    labels = [e.label for e in experiments]
    colors = _experiment_colors(len(experiments))

    width = max(5.5, 1.4 * len(experiments) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 5.5))
    bp = ax.boxplot(
        data,
        vert=True,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),  # whiskers extend to min/max so no flier dots
        boxprops={"edgecolor": "#1f3f70"},
        medianprops={"color": "#c14b3a", "linewidth": 2.0},
        whiskerprops={"color": "#1f3f70"},
        capprops={"color": "#1f3f70"},
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Overlay individual best-per-tilesize points.
    for i, arr in enumerate(data, start=1):
        if arr.size == 0:
            continue
        jitter = np.random.RandomState(0).uniform(-0.12, 0.12, size=arr.size)
        ax.scatter(
            np.full(arr.size, i) + jitter,
            arr,
            s=22,
            color="#222222",
            alpha=0.7,
            zorder=3,
        )

    ax.set_xticks(np.arange(1, len(experiments) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Best latency per tile-size combo{units}")
    _apply_sci_notation(ax, axis="y")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def plot_best_per_tilesize_bar(experiments: list[Experiment], out_path_no_ext: str, units: str) -> None:
    """Grouped bar chart: per tile-size combination, one bar per experiment.

    Restricted to combinations that have a best latency in every experiment so
    the comparison is apples-to-apples.
    """
    if len(experiments) < 2:  # noqa: PLR2004
        return

    # Find combos common to all experiments (with finite best latency in each).
    common: list[tuple[str, str, str]] = []
    first = experiments[0]
    for r in first.rows:
        if not r.get("best_latency"):
            continue
        key = (r["seq_tile"], r["emb_tile"], r["hid_tile"])
        if all(e.best_for(key) is not None for e in experiments):
            common.append(key)
    if not common:
        return

    # Sort common combos by mean best latency across experiments (ascending).
    def mean_best(key: tuple[str, str, str]) -> float:
        return float(np.mean([e.best_for(key) for e in experiments]))

    common.sort(key=mean_best)
    combo_labels = [f"{s}x{e}x{h}" for s, e, h in common]
    n_exp = len(experiments)
    n_combo = len(common)

    matrix = np.array(
        [[experiments[j].best_for(k) for k in common] for j in range(n_exp)],
        dtype=float,
    )

    colors = _experiment_colors(n_exp)
    width = max(8.0, 0.7 * n_combo + 2.0)
    fig, ax = plt.subplots(figsize=(width, 5.5))
    group_x = np.arange(n_combo)
    bar_w = 0.8 / n_exp
    for j, e in enumerate(experiments):
        offset = (j - (n_exp - 1) / 2) * bar_w
        ax.bar(
            group_x + offset,
            matrix[j],
            width=bar_w,
            color=colors[j],
            edgecolor="black",
            linewidth=0.4,
            label=e.label,
        )
    ax.set_xticks(group_x)
    ax.set_xticklabels(combo_labels, rotation=45, ha="right")
    ax.set_xlabel("Tile sizes (seq × emb × hid)")
    ax.set_ylabel(f"Best latency{units}")
    _apply_sci_notation(ax, axis="y")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def plot_success_rate_bar(experiments: list[Experiment], out_path_no_ext: str) -> None:
    if not experiments:
        return
    labels = [e.label for e in experiments]
    succ = np.array([e.total_succeeded for e in experiments])
    fail = np.array([e.total_failed for e in experiments])

    width = max(5.5, 1.4 * len(experiments) + 2.0)
    fig, ax = plt.subplots(figsize=(width, 5))
    x = np.arange(len(experiments))
    ax.bar(x, succ, color="#3b6cb7", edgecolor="black", linewidth=0.5, label="succeeded")
    ax.bar(
        x,
        fail,
        bottom=succ,
        color="#c14b3a",
        edgecolor="black",
        linewidth=0.5,
        label="failed",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of mappings")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    for xi, s, f in zip(x, succ, fail, strict=True):
        total = s + f
        if total:
            ax.text(
                float(xi),
                float(total),
                f"{s / total:.0%}",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#333333",
            )
    ax.set_ylim(0, (succ + fail).max() * 1.12)
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_compare_summary(experiments: list[Experiment], out_path: str, units: str) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "path",
                "best_latency" + (f" ({units})" if units else ""),
                "best_combo",
                "median_best_per_tile",
                "n_combos_with_data",
                "total_mappings",
                "succeeded",
                "failed",
                "success_rate",
            ]
        )
        for e in experiments:
            arr = e.best_per_tilesize
            w.writerow(
                [
                    e.label,
                    e.path,
                    f"{e.best_overall:.6g}" if arr.size else "",
                    e.best_combo_label,
                    f"{np.median(arr):.6g}" if arr.size else "",
                    arr.size,
                    e.total_mappings,
                    e.total_succeeded,
                    e.total_failed,
                    f"{e.success_rate:.4f}",
                ]
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not EXPERIMENTS:
        raise SystemExit("Configure EXPERIMENTS at the top of this file.")

    os.makedirs(OUT_DIR, exist_ok=True)
    set_publication_style()

    experiments = [load_experiment(p, lbl) for p, lbl in EXPERIMENTS]
    units_label = f" ({UNITS})" if UNITS else ""

    plot_best_overall_bar(experiments, os.path.join(OUT_DIR, "best_overall_bar"), units_label)
    plot_best_per_tilesize_box(experiments, os.path.join(OUT_DIR, "best_per_tilesize_box"), units_label)
    plot_best_per_tilesize_bar(experiments, os.path.join(OUT_DIR, "best_per_tilesize_bar"), units_label)
    plot_success_rate_bar(experiments, os.path.join(OUT_DIR, "success_rate_bar"))
    write_compare_summary(experiments, os.path.join(OUT_DIR, "compare_summary.csv"), UNITS)

    print(f"Compared {len(experiments)} experiment(s):")
    for e in experiments:
        best = e.best_overall
        best_str = f"{best:.4g}" if not np.isnan(best) else "n/a"
        print(
            f"  {e.label:<28s} best={best_str} "
            f"({e.best_combo_label or '—'})  "
            f"success={e.total_succeeded}/{e.total_mappings}"
        )
    print(f"\nWrote comparison plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
