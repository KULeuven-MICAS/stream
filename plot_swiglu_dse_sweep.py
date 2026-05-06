#!/usr/bin/env python3
"""Visualize a SwiGLU DSE tile-size sweep produced by ``main_swiglu_dse.py``.

Expected layout::

    EXPERIMENT_DIR/
        tilesizes_{s}_{e}_{h}/
            0/latency.yaml
            0/swiglu_*_mapping.yaml
            1/latency.yaml
            ...
        tilesizes_{s2}_{e2}_{h2}/
            ...

For each tile-size combination it produces:
    - ``report.txt``: counts of evaluated/succeeded/failed mappings + summary stats
    - ``latency_hist.{pdf,png}``
    - ``latency_ecdf.{pdf,png}``
    - ``latency_violin_box.{pdf,png}``
    - ``latency_best_so_far.{pdf,png}``

Across combinations it produces:
    - ``summary.csv``, ``summary.txt``
    - ``best_latency_bar.{pdf,png}``
    - ``latency_box_compare.{pdf,png}``
    - ``ecdf_compare.{pdf,png}``
    - ``success_rate_bar.{pdf,png}``
    - ``heatmap_best_seq{s}.{pdf,png}`` (one per distinct seq_len tile size)

All plots use scientific notation and large fonts, suitable for academic papers.

Usage::

    python plot_swiglu_dse_sweep.py \\
        --experiment-dir outputs/dse-inf-fifo-depth-...-4_row_8_col \\
        --out-dir outputs/dse-inf-fifo-depth-...-4_row_8_col/_plots
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

_TILE_DIR_RE = re.compile(r"^tilesizes_(\d+)_(\d+)_(\d+)$")


# -----------------------------
# Data loading
# -----------------------------


@dataclass
class TileCombo:
    """Aggregated per-(seq, embedding, hidden) tile-size data."""

    seq: int
    embedding: int
    hidden: int
    path: str
    total_dirs: int = 0
    yaml_present: int = 0
    yaml_missing: int = 0
    yaml_invalid: int = 0
    succeeded: int = 0
    failed: int = 0  # missing + invalid + non-finite
    finite_latencies: list[float] = field(default_factory=list)
    idx_latency_pairs: list[tuple[int, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.seq}x{self.embedding}x{self.hidden}"

    @property
    def best_latency(self) -> float:
        return min(self.finite_latencies) if self.finite_latencies else float("inf")

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total_dirs if self.total_dirs else 0.0


def _safe_load_latency_yaml(path: str) -> tuple[float | None, str | None]:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return None, f"yaml_load_error: {e}"
    if not isinstance(data, dict) or "latency" not in data:
        return None, "missing_or_malformed"
    try:
        latency = float(data["latency"])
    except Exception:
        return None, "non_numeric"
    if math.isnan(latency) or math.isinf(latency):
        return None, "non_finite"
    return latency, None


def _list_index_dirs(base_dir: str) -> list[tuple[int, str]]:
    if not os.path.isdir(base_dir):
        return []
    out: list[tuple[int, str]] = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
            continue
        try:
            out.append((int(name), p))
        except ValueError:
            continue
    out.sort(key=lambda x: x[0])
    return out


def _load_tile_combo(tile_path: str, seq: int, embedding: int, hidden: int) -> TileCombo:
    combo = TileCombo(seq=seq, embedding=embedding, hidden=hidden, path=tile_path)
    index_dirs = _list_index_dirs(tile_path)
    combo.total_dirs = len(index_dirs)
    for idx, d in index_dirs:
        latency_path = os.path.join(d, "latency.yaml")
        if not os.path.exists(latency_path):
            combo.yaml_missing += 1
            combo.failed += 1
            continue
        combo.yaml_present += 1
        latency = _safe_load_latency_yaml(latency_path)[0]
        if latency is None:
            combo.yaml_invalid += 1
            combo.failed += 1
            continue
        combo.succeeded += 1
        combo.finite_latencies.append(latency)
        combo.idx_latency_pairs.append((idx, latency))
    return combo


def discover_tile_combos(experiment_dir: str) -> list[TileCombo]:
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Experiment dir not found: {experiment_dir}")
    combos: list[TileCombo] = []
    for name in sorted(os.listdir(experiment_dir)):
        m = _TILE_DIR_RE.match(name)
        if not m:
            continue
        seq, emb, hid = (int(x) for x in m.groups())
        combos.append(_load_tile_combo(os.path.join(experiment_dir, name), seq, emb, hid))
    combos.sort(key=lambda c: (c.seq, c.embedding, c.hidden))
    return combos


# -----------------------------
# Style
# -----------------------------


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


def _apply_sci_notation(ax: Axes, axis: str = "both") -> None:
    if axis in ("both", "x"):
        ax.xaxis.set_major_formatter(_sci_formatter())
    if axis in ("both", "y"):
        ax.yaxis.set_major_formatter(_sci_formatter())


def _save_fig(fig: Figure, out_path_no_ext: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path_no_ext}.{ext}")
    plt.close(fig)


# -----------------------------
# Per-combination plots
# -----------------------------


def _plot_histogram(latencies: np.ndarray, out_path_no_ext: str, units: str) -> None:
    if latencies.size == 0:
        return
    q25, q75 = np.percentile(latencies, [25, 75])
    iqr = q75 - q25
    if iqr > 0 and latencies.max() > latencies.min():
        bin_width = 2 * iqr * (latencies.size ** (-1 / 3))
        bins = max(10, min(int(np.ceil((latencies.max() - latencies.min()) / bin_width)), 60))
    else:
        bins = 20

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(latencies, bins=bins, color="#3b6cb7", edgecolor="white", linewidth=0.6)
    ax.set_xlabel(f"Latency{units}")
    ax.set_ylabel("Mapping count")
    _apply_sci_notation(ax, axis="x")
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _plot_ecdf(latencies: np.ndarray, out_path_no_ext: str, units: str) -> None:
    if latencies.size == 0:
        return
    x = np.sort(latencies)
    y = np.arange(1, x.size + 1) / x.size

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, y, color="#c14b3a")
    ax.set_xlabel(f"Latency{units}")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1.02)
    _apply_sci_notation(ax, axis="x")
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _plot_violin_box(latencies: np.ndarray, out_path_no_ext: str, units: str) -> None:
    if latencies.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5.5))
    parts = ax.violinplot(latencies, showmeans=False, showextrema=False, showmedians=False)
    bodies = parts["bodies"]
    for body in list(bodies):  # type: ignore[arg-type]
        body.set_facecolor("#3b6cb7")
        body.set_edgecolor("#1f3f70")
        body.set_alpha(0.5)
    ax.boxplot(
        latencies,
        vert=True,
        widths=0.18,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "white", "edgecolor": "black"},
        medianprops={"color": "#c14b3a", "linewidth": 2.0},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#3b6cb7", "alpha": 0.6},
    )
    ax.set_ylabel(f"Latency{units}")
    ax.set_xticks([1])
    ax.set_xticklabels(["all evaluated"])
    _apply_sci_notation(ax, axis="y")
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _load_optimization_trace(trace_path: str) -> list[float] | None:
    """Return ordered list of incumbent latencies from a tetra optimization_trace.yaml.

    Returns ``None`` if the file is missing or has no usable incumbent values
    (e.g. mapping did not meet constraints, so tetra/ only contains model.ilp).
    """
    if not os.path.exists(trace_path):
        return None
    try:
        with open(trace_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    trace = data.get("trace") or []
    incumbents: list[float] = []
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        v = entry.get("incumbent")
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        incumbents.append(f)
    return incumbents or None


def _plot_optimization_progress(  # noqa: PLR0912, PLR0915
    combo: TileCombo,
    out_path_no_ext: str,
    units: str,
    ymax: float | None = None,
    annotate_optimal: bool = False,
    optimal_rtol: float = 0.005,
) -> None:
    """Per-tile-size plot: optimization incumbent descent at each mapping index.

    Stylistically mirrors ``_plot_box_compare`` — same orange "This Work"
    palette, black-edged final-incumbent marker for emphasis, grey × markers
    for mapping indices whose tetra/ run did not produce an optimization
    trace (i.e. did not meet constraints).
    """
    index_dirs = _list_index_dirs(combo.path)
    if not index_dirs:
        return

    indices_ok: list[int] = []
    traces: list[list[float]] = []
    indices_fail: list[int] = []
    for idx, d in index_dirs:
        trace = _load_optimization_trace(os.path.join(d, "tetra", "optimization_trace.yaml"))
        if trace:
            indices_ok.append(idx)
            traces.append(trace)
        else:
            indices_fail.append(idx)

    if not indices_ok and not indices_fail:
        return

    this_color = "#DD8452"

    width = max(6.5, 0.13 * len(index_dirs) + 2.8)
    fig, ax = plt.subplots(figsize=(width, 5.0))

    stair_w = 0.7  # horizontal extent of each per-mapping staircase (in index units)
    for x_center, trace in zip(indices_ok, traces, strict=False):
        k = len(trace)
        if k == 1:
            xs = [float(x_center)]
        else:
            xs = [x_center - stair_w / 2.0 + (i / (k - 1)) * stair_w for i in range(k)]
        ax.plot(
            xs,
            trace,
            color=this_color,
            alpha=0.55,
            linewidth=1.4,
            drawstyle="steps-post",
            solid_capstyle="butt",
            zorder=2,
        )
        if k > 1:
            ax.scatter(
                xs[:-1],
                trace[:-1],
                color=this_color,
                alpha=0.5,
                s=14,
                edgecolors="none",
                zorder=3,
            )
        ax.scatter(
            [xs[-1]],
            [trace[-1]],
            color=this_color,
            alpha=0.95,
            s=28,
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
        )

    if annotate_optimal and traces:
        best_y = min(t[-1] for t in traces)
        threshold = best_y * (1.0 + optimal_rtol)
        candidates: list[tuple[int, float, float]] = []
        for x_center, trace in zip(indices_ok, traces, strict=False):
            if trace[-1] > threshold:
                continue
            k = len(trace)
            x_final = float(x_center) if k == 1 else x_center + stair_w / 2.0
            candidates.append((x_center, x_final, float(trace[-1])))
        candidates.sort(key=lambda c: c[0])

        # Cluster annotations that would otherwise overplot: consecutive indices
        # whose final-incumbent y values match within a relative tolerance get
        # merged into a single comma-separated label at the cluster centroid.
        groups: list[list[tuple[int, float, float]]] = []
        for cand in candidates:
            if groups:
                prev = groups[-1][-1]
                close_x = cand[1] - prev[1] <= 2.0  # noqa: PLR2004
                close_y = abs(cand[2] - prev[2]) <= 1e-3 * max(prev[2], 1.0)
                if close_x and close_y:
                    groups[-1].append(cand)
                    continue
            groups.append([cand])

        for group in groups:
            label = ",".join(str(g[0]) for g in group)
            x_pos = sum(g[1] for g in group) / len(group)
            y_pos = max(g[2] for g in group)
            ax.annotate(
                label,
                xy=(x_pos, y_pos),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                zorder=6,
            )

    if indices_fail:
        if traces:
            # Place X markers among the final-incumbent cluster so they stay
            # visible whether the y-axis is auto-fit or zoomed in.
            y_marker = float(np.median([t[-1] for t in traces]))
        else:
            y_marker = 0.5
        ax.scatter(
            indices_fail,
            [y_marker] * len(indices_fail),
            marker="x",
            s=80,
            color="#555555",
            linewidths=1.8,
            zorder=5,
            label="Doesn't meet constraints",
        )

    all_idxs = indices_ok + indices_fail
    ax.set_xlim(min(all_idxs) - 0.5, max(all_idxs) + 0.5)
    ax.set_xlabel("Mapping index (evaluation order)")
    ax.set_ylabel(f"Latency{units}")
    _apply_sci_notation(ax, axis="y")

    if ymax is not None:
        ax.set_ylim(0.0, ymax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Custom legend so it explains both elements without cluttering the plot.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=this_color,
            alpha=0.95,
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=7,
            linestyle="None",
            label="Optimal solution for allocation",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=this_color,
            alpha=0.5,
            markeredgecolor="none",
            markersize=7,
            linestyle="None",
            label="Constraint optimization progress",
        ),
    ]
    if indices_fail:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="#555555",
                markersize=8,
                markeredgewidth=1.8,
                linestyle="None",
                label="Doesn't meet constraints",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.4,
        borderpad=0.3,
    )
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _plot_best_so_far(idx_latency_pairs: list[tuple[int, float]], out_path_no_ext: str, units: str) -> None:
    if not idx_latency_pairs:
        return
    pairs = sorted(idx_latency_pairs, key=lambda p: p[0])
    idxs = np.array([p[0] for p in pairs], dtype=int)
    lats = np.array([p[1] for p in pairs], dtype=float)
    best_so_far = np.minimum.accumulate(lats)
    is_new_best = np.r_[True, best_so_far[1:] < best_so_far[:-1]]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(idxs, lats, s=14, alpha=0.4, color="#5a7fb3", label="evaluated mapping")
    ax.plot(idxs, best_so_far, linewidth=2.2, color="#c14b3a", label="best-so-far")
    ax.scatter(idxs[is_new_best], best_so_far[is_new_best], s=36, color="#c14b3a", zorder=4)
    ax.set_xlabel("Mapping index (evaluation order)")
    ax.set_ylabel(f"Latency{units}")
    _apply_sci_notation(ax, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _write_combo_report(combo: TileCombo, out_path: str, units: str) -> None:
    lats = np.asarray(combo.finite_latencies, dtype=float)
    lines = [
        f"Tile sizes: seq_len={combo.seq}, embedding={combo.embedding}, hidden={combo.hidden}",
        f"Source dir: {combo.path}",
        "",
        "Mapping evaluation counts:",
        f"  total mapping dirs:   {combo.total_dirs}",
        f"  latency.yaml present: {combo.yaml_present}",
        f"  latency.yaml missing: {combo.yaml_missing}",
        f"  invalid / non-finite: {combo.yaml_invalid}",
        f"  succeeded (finite):   {combo.succeeded}",
        f"  failed (any reason):  {combo.failed}",
        f"  success rate:         {combo.success_rate:.2%}",
    ]
    if lats.size:
        lines += [
            "",
            f"Latency summary{units}:",
            f"  best (min): {lats.min():.6g}",
            f"  p05:        {np.percentile(lats, 5):.6g}",
            f"  median:     {np.percentile(lats, 50):.6g}",
            f"  mean:       {lats.mean():.6g}",
            f"  p95:        {np.percentile(lats, 95):.6g}",
            f"  max:        {lats.max():.6g}",
            f"  std:        {lats.std(ddof=1) if lats.size > 1 else 0.0:.6g}",
        ]
    else:
        lines += ["", "No finite latencies — all mappings failed."]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def render_per_combo(combo: TileCombo, out_dir: str, units: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    units_label = f" ({units})" if units else ""

    _write_combo_report(combo, os.path.join(out_dir, "report.txt"), units_label)

    lats = np.asarray(combo.finite_latencies, dtype=float)
    _plot_histogram(lats, os.path.join(out_dir, "latency_hist"), units=units_label)
    _plot_ecdf(lats, os.path.join(out_dir, "latency_ecdf"), units=units_label)
    _plot_violin_box(lats, os.path.join(out_dir, "latency_violin_box"), units=units_label)
    _plot_best_so_far(
        combo.idx_latency_pairs,
        os.path.join(out_dir, "latency_best_so_far"),
        units=units_label,
    )
    _plot_optimization_progress(
        combo,
        os.path.join(out_dir, "optimization_progress"),
        units=units_label,
    )
    if combo.finite_latencies:
        ymax_zoom = float(max(combo.finite_latencies)) * (4.0 / 3.0)
        _plot_optimization_progress(
            combo,
            os.path.join(out_dir, "optimization_progress_zoom"),
            units=units_label,
            ymax=ymax_zoom,
        )


# -----------------------------
# Cross-combination plots
# -----------------------------


def _combos_with_data(combos: list[TileCombo]) -> list[TileCombo]:
    return [c for c in combos if c.finite_latencies]


def _plot_best_latency_bar(combos: list[TileCombo], out_path_no_ext: str, units: str) -> None:
    valid = _combos_with_data(combos)
    if not valid:
        return
    valid_sorted = sorted(valid, key=lambda c: c.best_latency)
    labels = [c.label for c in valid_sorted]
    bests = np.array([c.best_latency for c in valid_sorted])

    height = max(3.5, 0.35 * len(valid_sorted) + 1.5)
    fig, ax = plt.subplots(figsize=(9, height))
    y = np.arange(len(valid_sorted))
    ax.barh(y, bests, color="#3b6cb7", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Best latency{units}")
    ax.set_ylabel("Tile sizes (seq × emb × hid)")
    _apply_sci_notation(ax, axis="x")
    # Annotate best with idx of best mapping
    for c, yi, b in zip(valid_sorted, y, bests, strict=False):
        best_idx = min(c.idx_latency_pairs, key=lambda p: p[1])[0]
        ax.text(
            float(b),
            float(yi),
            f"  idx={best_idx}",
            va="center",
            ha="left",
            fontsize=11,
            color="#333333",
        )
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _plot_box_compare(combos: list[TileCombo], out_path_no_ext: str, units: str) -> None:
    if not combos:
        return
    valid = _combos_with_data(combos)
    invalid = [c for c in combos if not c.finite_latencies]
    if not valid and not invalid:
        return

    # Valid combos sorted by best latency; failed combos appended at the end.
    valid_sorted = sorted(valid, key=lambda c: c.best_latency)
    invalid_sorted = sorted(invalid, key=lambda c: (c.seq, c.embedding, c.hidden))
    all_combos = valid_sorted + invalid_sorted
    labels = [c.label for c in all_combos]
    positions = list(range(1, len(all_combos) + 1))

    # Match the "This Work" style from ~/iron/plot_gemm.py for paper consistency.
    this_color = "#DD8452"

    width = max(7.0, 0.45 * len(all_combos) + 3.5)
    fig, ax = plt.subplots(figsize=(width, 5.5))

    if valid_sorted:
        data = [np.asarray(c.finite_latencies) for c in valid_sorted]
        valid_positions = positions[: len(valid_sorted)]
        bp = ax.boxplot(
            data,
            positions=valid_positions,
            vert=True,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": this_color,
                "markeredgecolor": this_color,
                "alpha": 0.5,
            },
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(this_color)
            patch.set_edgecolor(this_color)
            patch.set_alpha(0.35)
        for key in ("whiskers", "caps"):
            for line in bp[key]:
                line.set_color(this_color)
                line.set_alpha(0.7)

    if invalid_sorted:
        invalid_positions = positions[len(valid_sorted) :]
        if valid_sorted:
            all_lats = np.concatenate([np.asarray(c.finite_latencies) for c in valid_sorted])
            y_marker = float(np.median(all_lats))
        else:
            y_marker = 0.5
        ax.scatter(
            invalid_positions,
            [y_marker] * len(invalid_positions),
            marker="x",
            s=140,
            color="#555555",
            linewidths=2.2,
            zorder=5,
            label="Doesn't meet constraints",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(0.5, len(all_combos) + 0.5)
    ax.set_xlabel("Tile sizes (seq × emb × hid)")
    ax.set_ylabel(f"Latency{units}")
    _apply_sci_notation(ax, axis="y")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    if invalid_sorted:
        ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _plot_ecdf_compare(combos: list[TileCombo], out_path_no_ext: str, units: str, top_k: int = 10) -> None:
    valid = _combos_with_data(combos)
    if len(valid) < 2:  # noqa: PLR2004
        return
    valid_sorted = sorted(valid, key=lambda c: c.best_latency)
    shown = valid_sorted[:top_k]
    cmap = plt.get_cmap("tab10" if len(shown) <= 10 else "tab20")  # noqa: PLR2004

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, c in enumerate(shown):
        x = np.sort(np.asarray(c.finite_latencies))
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, color=cmap(i % cmap.N), label=c.label, linewidth=2.0)
    ax.set_xlabel(f"Latency{units}")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1.02)
    _apply_sci_notation(ax, axis="x")
    ax.legend(title="seq × emb × hid", loc="lower right", ncol=1, fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _plot_success_rate_bar(combos: list[TileCombo], out_path_no_ext: str) -> None:
    if not combos:
        return
    combos_sorted = sorted(combos, key=lambda c: -c.success_rate)
    labels = [c.label for c in combos_sorted]
    succ = np.array([c.succeeded for c in combos_sorted])
    fail = np.array([c.failed for c in combos_sorted])

    height = max(4.0, 0.32 * len(combos_sorted) + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))
    y = np.arange(len(combos_sorted))
    ax.barh(y, succ, color="#3b6cb7", edgecolor="black", linewidth=0.4, label="succeeded")
    ax.barh(y, fail, left=succ, color="#c14b3a", edgecolor="black", linewidth=0.4, label="failed")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of mappings")
    ax.set_ylabel("Tile sizes (seq × emb × hid)")
    ax.legend(loc="lower right")
    for yi, (s, f) in enumerate(zip(succ, fail, strict=False)):
        total = s + f
        if total:
            ax.text(float(total), float(yi), f"  {s / total:.0%}", va="center", ha="left", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_path_no_ext)


def _plot_heatmaps_by_seq(combos: list[TileCombo], out_dir: str, units: str) -> None:
    """For each distinct seq tile size, plot a heatmap of best latency over (embedding, hidden)."""
    valid = _combos_with_data(combos)
    if not valid:
        return
    seqs = sorted({c.seq for c in valid})
    embs = sorted({c.embedding for c in valid})
    hids = sorted({c.hidden for c in valid})
    if len(embs) < 2 or len(hids) < 2:  # noqa: PLR2004
        return  # nothing meaningful to compare in 2D

    by_key = {(c.seq, c.embedding, c.hidden): c for c in valid}
    for s in seqs:
        grid = np.full((len(embs), len(hids)), np.nan)
        for i, e in enumerate(embs):
            for j, h in enumerate(hids):
                c = by_key.get((s, e, h))
                if c is not None:
                    grid[i, j] = c.best_latency

        if np.all(np.isnan(grid)):
            continue
        fig, ax = plt.subplots(figsize=(0.9 * len(hids) + 3.5, 0.7 * len(embs) + 3))
        im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(hids)))
        ax.set_xticklabels([str(h) for h in hids])
        ax.set_yticks(range(len(embs)))
        ax.set_yticklabels([str(e) for e in embs])
        ax.set_xlabel("Hidden tile size")
        ax.set_ylabel("Embedding tile size")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Best latency{units}")
        cbar.formatter = _sci_formatter()
        cbar.update_ticks()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2e}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="white" if v > np.nanmean(grid) else "black",
                    )
        fig.tight_layout()
        _save_fig(fig, os.path.join(out_dir, f"heatmap_best_seq{s}"))


def _write_summary(combos: list[TileCombo], out_dir: str, units: str) -> None:
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seq_tile",
                "emb_tile",
                "hid_tile",
                "total_mappings",
                "succeeded",
                "failed",
                "success_rate",
                "best_latency",
                "median_latency",
                "mean_latency",
                "best_idx",
            ]
        )
        for c in combos:
            best_idx = min(c.idx_latency_pairs, key=lambda p: p[1])[0] if c.idx_latency_pairs else ""
            lats = np.asarray(c.finite_latencies)
            w.writerow(
                [
                    c.seq,
                    c.embedding,
                    c.hidden,
                    c.total_dirs,
                    c.succeeded,
                    c.failed,
                    f"{c.success_rate:.4f}",
                    f"{c.best_latency:.6g}" if lats.size else "",
                    f"{np.median(lats):.6g}" if lats.size else "",
                    f"{lats.mean():.6g}" if lats.size else "",
                    best_idx,
                ]
            )

    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("SwiGLU DSE tile-size sweep summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Combinations evaluated: {len(combos)}\n")
        valid = _combos_with_data(combos)
        f.write(f"Combinations with at least one finite latency: {len(valid)}\n")
        if valid:
            best = min(valid, key=lambda c: c.best_latency)
            f.write(f"Global best latency{units}: {best.best_latency:.6g} at {best.label}\n")
        f.write("\n")
        for c in combos:
            f.write(f"--- {c.label} ---\n")
            f.write(f"  total: {c.total_dirs}, succeeded: {c.succeeded}, failed: {c.failed}\n")
            if c.finite_latencies:
                f.write(f"  best{units}: {c.best_latency:.6g}, median: {np.median(c.finite_latencies):.6g}\n")
            else:
                f.write("  no finite latencies\n")


def render_cross_combo(combos: list[TileCombo], out_dir: str, units: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    units_label = f" ({units})" if units else ""
    _plot_best_latency_bar(combos, os.path.join(out_dir, "best_latency_bar"), units_label)
    _plot_box_compare(combos, os.path.join(out_dir, "latency_box_compare"), units_label)
    _plot_ecdf_compare(combos, os.path.join(out_dir, "ecdf_compare"), units_label)
    _plot_success_rate_bar(combos, os.path.join(out_dir, "success_rate_bar"))
    _plot_heatmaps_by_seq(combos, out_dir, units_label)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize a SwiGLU DSE tile-size sweep.")
    p.add_argument(
        "--experiment-dir",
        required=True,
        help="Path to outputs/<experiment-id>/ directory containing tilesizes_*/ subfolders.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write plots. Defaults to <experiment-dir>/_plots.",
    )
    p.add_argument(
        "--units",
        default="cycles",
        help="Latency units shown on axis labels (default: 'cycles'). Use '' for none.",
    )
    p.add_argument(
        "--top-k-ecdf",
        type=int,
        default=10,
        help="Limit ECDF overlay to the top-K combinations by best latency.",
    )
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.experiment_dir, "_plots")
    os.makedirs(out_dir, exist_ok=True)
    set_publication_style()

    combos = discover_tile_combos(args.experiment_dir)
    if not combos:
        raise SystemExit(f"No 'tilesizes_*' folders found in {args.experiment_dir}")

    print(f"Found {len(combos)} tile-size combination(s) under {args.experiment_dir}")
    units_label = f" ({args.units})" if args.units else ""

    per_dir = os.path.join(out_dir, "per_tilesize")
    os.makedirs(per_dir, exist_ok=True)
    for c in combos:
        sub = os.path.join(per_dir, f"tilesizes_{c.seq}_{c.embedding}_{c.hidden}")
        render_per_combo(c, sub, args.units)
        finite = len(c.finite_latencies)
        print(
            f"  {c.label}: total={c.total_dirs}, succ={c.succeeded}, "
            f"fail={c.failed}, best={c.best_latency if finite else float('inf'):.4g}"
        )

    cross_dir = os.path.join(out_dir, "cross_tilesize")
    render_cross_combo(combos, cross_dir, args.units)
    _write_summary(combos, out_dir, units_label)

    print(f"\nWrote plots to: {out_dir}")
    print(f"Summary: {os.path.join(out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
