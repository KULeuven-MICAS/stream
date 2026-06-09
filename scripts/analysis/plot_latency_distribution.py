#!/usr/bin/env python3
"""
Plot latency distribution from per-run output directories.

Expected layout:
  BASE_DIR/
    0/latency.yaml
    1/latency.yaml
    2/latency.yaml
    ...

Each latency.yaml is expected to contain:
  latency: <float or int>

Some indices may be missing latency.yaml (e.g. failed runs). This script:
  - collects all latencies it can find
  - tracks missing/invalid files
  - prints summary stats
  - saves plots to OUT_DIR

Usage:
  python plot_latency_distribution.py --base-dir /path/to/base --out-dir /path/to/save/plots

Example:
  python plot_latency_distribution.py --base-dir ./results/mappings --out-dir ./results/plots
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import yaml

_LAT_DIR_RE = re.compile(r"(^|/)(\d+)($|/)")


@dataclass
class LoadReport:
    base_dir: str
    found_latency_files: int
    parsed_latencies: int
    missing_latency_files: int
    invalid_latency_files: int
    non_numeric_latency: int
    dirs_scanned: int
    indices_with_latency: list[int]
    indices_missing_latency: list[int]
    indices_invalid_latency: list[int]


def _list_index_dirs(base_dir: str) -> list[tuple[int, str]]:
    """
    Return [(idx, path), ...] for directories directly under base_dir whose
    folder name is an integer.
    """
    index_dirs: list[tuple[int, str]] = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory does not exist or is not a directory: {base_dir}")

    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            try:
                idx = int(name)
                index_dirs.append((idx, p))
            except ValueError:
                continue

    index_dirs.sort(key=lambda x: x[0])
    return index_dirs


def _safe_load_latency_yaml(path: str) -> tuple[float | None, str | None]:
    """
    Returns (latency, error_reason).
    latency is None if there was any error.
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return None, f"yaml_load_error: {e}"

    if not isinstance(data, dict):
        return None, "not_a_dict"
    if "latency" not in data:
        return None, "missing_key_latency"

    val = data["latency"]
    try:
        latency = float(val)
    except Exception:
        return None, "non_numeric_latency"

    if math.isnan(latency) or math.isinf(latency):
        return None, "nan_or_inf_latency"

    return latency, None


def load_latencies(base_dir: str) -> tuple[list[float], list[tuple[int, float]], LoadReport]:
    """
    Scan BASE_DIR/<idx>/latency.yaml and return (latencies, idx_latency_pairs, report).
    """
    index_dirs = _list_index_dirs(base_dir)
    latencies: list[float] = []
    idx_latency_pairs: list[tuple[int, float]] = []

    found_latency_files = 0
    parsed_latencies = 0
    missing_latency_files = 0
    invalid_latency_files = 0
    non_numeric_latency = 0

    indices_with_latency: list[int] = []
    indices_missing_latency: list[int] = []
    indices_invalid_latency: list[int] = []

    for idx, d in index_dirs:
        latency_path = os.path.join(d, "latency.yaml")
        if not os.path.exists(latency_path):
            missing_latency_files += 1
            indices_missing_latency.append(idx)
            continue

        found_latency_files += 1
        latency, err = _safe_load_latency_yaml(latency_path)
        if latency is None:
            invalid_latency_files += 1
            indices_invalid_latency.append(idx)
            if err == "non_numeric_latency":
                non_numeric_latency += 1
            continue
        latencies.append(latency)
        parsed_latencies += 1
        indices_with_latency.append(idx)
        idx_latency_pairs.append((idx, latency))

    report = LoadReport(
        base_dir=base_dir,
        found_latency_files=found_latency_files,
        parsed_latencies=parsed_latencies,
        missing_latency_files=missing_latency_files,
        invalid_latency_files=invalid_latency_files,
        non_numeric_latency=non_numeric_latency,
        dirs_scanned=len(index_dirs),
        indices_with_latency=indices_with_latency,
        indices_missing_latency=indices_missing_latency,
        indices_invalid_latency=indices_invalid_latency,
    )
    return latencies, idx_latency_pairs, report


def _pretty_histogram(latencies: np.ndarray, out_path: str, title: str) -> None:
    """
    Save a clean histogram with an automatically chosen number of bins.
    Uses Freedman–Diaconis rule when possible for a 'pretty' binning.
    """
    if latencies.size == 0:
        return

    # Freedman–Diaconis bin width
    q25, q75 = np.percentile(latencies, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        bin_width = 2 * iqr * (latencies.size ** (-1 / 3))
        if bin_width > 0:
            bins = int(np.ceil((latencies.max() - latencies.min()) / bin_width))
            bins = max(10, min(bins, 100))  # clamp to reasonable range
        else:
            bins = 30
    else:
        bins = 30

    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=bins)
    plt.title(title)
    plt.xlabel("Latency")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _pretty_ecdf(latencies: np.ndarray, out_path: str, title: str) -> None:
    """
    Save an ECDF plot (very good for distributions).
    """
    if latencies.size == 0:
        return

    x = np.sort(latencies)
    y = np.arange(1, x.size + 1) / x.size

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Latency")
    plt.ylabel("ECDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _pretty_box_and_violin(latencies: np.ndarray, out_path: str, title: str) -> None:
    """
    Combined violin + box plot (nice compact summary).
    """
    if latencies.size == 0:
        return

    plt.figure(figsize=(10, 4))
    plt.violinplot(latencies, showmeans=True, showextrema=True, showmedians=True)
    plt.boxplot(latencies, vert=True, widths=0.2)
    plt.title(title)
    plt.ylabel("Latency")
    plt.xticks([1], ["All runs"])
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _find_trace_yaml(run_dir: str) -> str | None:
    """Walk *run_dir* and return the first ``optimization_trace.yaml`` found, or None."""
    for root, _dirs, files in os.walk(run_dir):
        if "optimization_trace.yaml" in files:
            return os.path.join(root, "optimization_trace.yaml")
    return None


def _load_incumbent_series(trace_path: str) -> list[tuple[float, float]] | None:
    """
    Load an optimization trace YAML and return a list of (time_s, incumbent)
    pairs representing the incumbent progression.

    Between explicit incumbent events the value stays constant (carried forward),
    so every trace entry that has or inherits an incumbent produces a point.
    Returns None if the file cannot be read or has no incumbent data.
    """
    try:
        with open(trace_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    entries = data.get("trace") if isinstance(data, dict) else None
    if not entries:
        return None

    points: list[tuple[float, float]] = []
    current_incumbent: float | None = None

    for entry in entries:
        t = entry.get("time_s")
        if t is None:
            continue
        if "incumbent" in entry:
            current_incumbent = float(entry["incumbent"])
        if current_incumbent is not None:
            points.append((float(t), current_incumbent))

    return points if points else None


def _plot_best_so_far(  # noqa: PLR0915
    idx_latency_pairs,
    out_path: str,
    units: str = "",
    base_dir: str | None = None,
) -> None:
    """
    idx_latency_pairs: list of (idx:int, latency:float), for valid runs only.
    Saves a single plot showing:
      - scatter of all latencies
      - line of best-so-far (running minimum)
      - idx labels next to each "new best" point
      - per-run incumbent progress lines (from optimization_trace.yaml)
    """
    if not idx_latency_pairs:
        return

    idxs = np.array([p[0] for p in idx_latency_pairs], dtype=int)
    lats = np.array([p[1] for p in idx_latency_pairs], dtype=float)

    # Sort by idx so the evolution is correct
    order = np.argsort(idxs)
    idxs = idxs[order]
    lats = lats[order]

    best_so_far = np.minimum.accumulate(lats)

    # Points where we actually improved the running minimum ("best so far" dots)
    is_new_best = np.r_[True, best_so_far[1:] < best_so_far[:-1]]
    best_lats = best_so_far[is_new_best]

    plt.figure(figsize=(15, 6))

    # Per-run incumbent progress lines from optimization traces
    # Track the right-edge x of each run's trace line so the best-so-far
    # dots and line align with the endpoint of the trace, not the run center.
    run_end_x: dict[int, float] = {}  # idx → right-edge x position
    line_half_width = 0.0

    if base_dir is not None:
        # Determine a uniform line width: fraction of the gap between run points
        if len(idxs) >= 2:  # noqa: PLR2004
            avg_gap = (float(idxs[-1]) - float(idxs[0])) / (len(idxs) - 1)
        else:
            avg_gap = 1.0
        line_half_width = avg_gap * 0.35  # each line spans ~70% of the gap

        trace_plotted = False
        final_latency_of = {int(idx): lat for idx, lat in zip(idxs, lats, strict=False)}

        for idx_val in idxs:
            run_dir = os.path.join(base_dir, str(idx_val))
            trace_path = _find_trace_yaml(run_dir)
            if trace_path is None:
                continue
            series = _load_incumbent_series(trace_path)
            if series is None:
                continue

            # Ensure the final latency.yaml value is the last point in the series
            final_lat = final_latency_of.get(int(idx_val))
            if final_lat is not None:
                last_t, last_v = series[-1]
                if abs(last_v - final_lat) > 0.5:  # noqa: PLR2004
                    series.append((last_t + 1e-6, final_lat))

            # Normalise trace times to [idx - half_width, idx + half_width]
            t_min = series[0][0]
            t_max = series[-1][0]
            t_range = t_max - t_min if t_max > t_min else 1.0

            xs = [float(idx_val) - line_half_width + (t - t_min) / t_range * 2 * line_half_width for t, _ in series]
            ys = [v for _, v in series]

            run_end_x[int(idx_val)] = xs[-1]

            label = "incumbent progress" if not trace_plotted else None
            plt.step(xs, ys, where="post", linewidth=1.0, alpha=0.6, color="tab:blue", label=label)
            plt.plot(xs[-1], ys[-1], "o", markersize=3.5, alpha=0.6, color="tab:blue")
            trace_plotted = True

    # Best-so-far line and dots: use the trace right-edge x when available,
    # otherwise fall back to the run's idx.
    best_so_far_xs = np.array([run_end_x.get(int(i), float(i)) for i in idxs])
    plt.plot(best_so_far_xs, best_so_far, linewidth=1.0, color="tab:orange", label="best-so-far (running min)")

    # Highlight and label new-best points
    best_xs = best_so_far_xs[is_new_best]
    plt.scatter(best_xs, best_lats, s=22, alpha=0.9, zorder=3, color="tab:orange")
    for x, y in zip(best_xs, best_lats, strict=False):
        plt.annotate(
            str(int(x)),
            (x, y),
            textcoords="offset points",
            xytext=(6, -10),
            ha="left",
            va="bottom",
            fontsize=8,
        )

    u = f" ({units})" if units else ""
    plt.title(f"Best latency found vs evaluated mappings{u}")
    plt.xlabel("Mapping idx (evaluation order)")
    plt.ylabel(f"Latency{u}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_report(latencies: np.ndarray, report: LoadReport) -> None:
    print(f"Base dir: {report.base_dir}")
    print(f"Index dirs scanned: {report.dirs_scanned}")
    print(f"latency.yaml present: {report.found_latency_files}")
    print(f"latency.yaml missing: {report.missing_latency_files}")
    print(f"latency.yaml invalid: {report.invalid_latency_files} (non-numeric: {report.non_numeric_latency})")
    print(f"Parsed latencies: {report.parsed_latencies}")

    if latencies.size == 0:
        print("No valid latencies found. Nothing to plot.")
        return

    print("\nLatency summary:")
    print(f"  count: {latencies.size}")
    print(f"  min:   {latencies.min():.6g}")
    print(f"  p01:   {np.percentile(latencies, 1):.6g}")
    print(f"  p05:   {np.percentile(latencies, 5):.6g}")
    print(f"  p50:   {np.percentile(latencies, 50):.6g}")
    print(f"  p95:   {np.percentile(latencies, 95):.6g}")
    print(f"  p99:   {np.percentile(latencies, 99):.6g}")
    print(f"  max:   {latencies.max():.6g}")
    print(f"  mean:  {latencies.mean():.6g}")
    print(f"  std:   {latencies.std(ddof=1) if latencies.size > 1 else 0.0:.6g}")

    missing_rate = report.missing_latency_files / report.dirs_scanned if report.dirs_scanned else 0.0
    invalid_rate = report.invalid_latency_files / report.dirs_scanned if report.dirs_scanned else 0.0
    print("\nRun completeness:")
    print(f"  missing rate: {missing_rate:.2%}")
    print(f"  invalid rate: {invalid_rate:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latency distribution from per-index output dirs.")
    parser.add_argument("--base-dir", required=True, help="Base directory containing ./0 ./1 ./2 ...")
    parser.add_argument("--out-dir", required=True, help="Directory where plots will be saved.")
    parser.add_argument("--prefix", default="latency", help="Prefix for saved plot filenames.")
    parser.add_argument(
        "--units",
        default="",
        help="Optional units label appended to axis labels, e.g. 'cycles' or 'ms'.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    latencies_list, idx_latency_pairs, report = load_latencies(args.base_dir)
    latencies = np.array(latencies_list, dtype=float)

    print_report(latencies, report)

    if latencies.size == 0:
        return

    units = f" ({args.units})" if args.units else ""
    _pretty_histogram(
        latencies,
        out_path=os.path.join(args.out_dir, f"{args.prefix}_hist.png"),
        title=f"Latency distribution (histogram){units}",
    )
    _pretty_ecdf(
        latencies,
        out_path=os.path.join(args.out_dir, f"{args.prefix}_ecdf.png"),
        title=f"Latency distribution (ECDF){units}",
    )
    _pretty_box_and_violin(
        latencies,
        out_path=os.path.join(args.out_dir, f"{args.prefix}_violin_box.png"),
        title=f"Latency distribution (violin + box){units}",
    )
    _plot_best_so_far(
        idx_latency_pairs,
        out_path=os.path.join(args.out_dir, f"{args.prefix}_best_so_far.png"),
        units=args.units,
        base_dir=args.base_dir,
    )

    print("\nSaved plots:")
    print(f"  {os.path.join(args.out_dir, f'{args.prefix}_hist.png')}")
    print(f"  {os.path.join(args.out_dir, f'{args.prefix}_ecdf.png')}")
    print(f"  {os.path.join(args.out_dir, f'{args.prefix}_violin_box.png')}")
    print(f"  {os.path.join(args.out_dir, f'{args.prefix}_best_so_far.png')}")


if __name__ == "__main__":
    main()
