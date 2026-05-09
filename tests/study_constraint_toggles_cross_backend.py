"""Cross-backend comparison of constraint toggle study results.

Loads YAML results from per-backend study runs and produces:
1. A terminal comparison table showing objective and solve time per backend per combination
2. A grouped bar chart comparing objectives across backends
3. A grouped bar chart comparing solve times across backends
4. A heatmap of objective deltas (%) per backend relative to each backend's own all-enabled baseline

Usage:
    python tests/study_constraint_toggles_cross_backend.py \
        --results outputs/constraint_study/gurobi/results.yaml \
                  outputs/constraint_study/ortools_gscip/results.yaml \
                  outputs/constraint_study/ortools_highs/results.yaml \
        --output-dir outputs/constraint_study/cross_backend
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BACKEND_DISPLAY_NAMES = {
    "gurobi": "Gurobi",
    "ortools_gscip": "OR-Tools GSCIP",
    "ortools_highs": "OR-Tools HiGHS",
}

BACKEND_COLORS = {
    "gurobi": "#2196F3",
    "ortools_gscip": "#4CAF50",
    "ortools_highs": "#FF9800",
}


def _load_results(yaml_paths: list[str]) -> dict[str, list[dict]]:
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    backend_results: dict[str, list[dict]] = {}
    for path in yaml_paths:
        with open(path) as f:
            data = yaml.safe_load(f)

        backend_name = os.path.basename(os.path.dirname(path))
        backend_results[backend_name] = data

    return backend_results


def _short_label(label: str) -> str:
    if "+" not in label and "None" not in label:
        return label
    parts = label.split(" + ")
    abbrev = {
        "Memory Capacity": "Mem",
        "Object FIFO Depth": "FIFO",
        "Buffer Descriptors": "BD",
        "DMA Channels": "DMA",
    }
    short_parts = [abbrev.get(p.strip(), p.strip()) for p in parts]
    result = " + ".join(short_parts)
    if label == "None (all disabled)":
        return "None"
    return result


def _print_cross_backend_table(backend_results: dict[str, list[dict]]) -> None:
    backends = list(backend_results.keys())
    first_backend = backends[0]
    labels = [r["label"] for r in backend_results[first_backend]]

    header_parts = [f"{'#':>3}  {'Constraints':<45}"]
    for b in backends:
        display = BACKEND_DISPLAY_NAMES.get(b, b)
        header_parts.append(f"{'Obj (' + display + ')':>20}  {'Time':>8}")
    header = "  ".join(header_parts)
    sep = "-" * len(header)

    print()
    print("=== Cross-Backend Constraint Toggle Comparison ===")
    print()
    print(header)
    print(sep)

    for i, label in enumerate(labels):
        short = _short_label(label)
        if len(short) > 45:
            short = short[:42] + "..."
        parts = [f"{i + 1:>3}  {short:<45}"]
        for b in backends:
            r = backend_results[b][i]
            obj = r.get("objective")
            time_s = r.get("solve_time_s", 0)
            if obj is not None:
                parts.append(f"{obj:>20,.0f}  {time_s:>7.1f}s")
            else:
                parts.append(f"{'FAILED':>20}  {time_s:>7.1f}s")
        print("  ".join(parts))

    print(sep)

    print()
    baselines = {}
    for b in backends:
        obj = backend_results[b][0].get("objective")
        display = BACKEND_DISPLAY_NAMES.get(b, b)
        if obj is not None:
            baselines[b] = obj
            print(f"Baseline (all enabled) {display}: {obj:,.0f}")

    if len(baselines) >= 2:
        vals = list(baselines.values())
        keys = list(baselines.keys())
        for i in range(1, len(vals)):
            diff = abs(vals[i] - vals[0]) / max(abs(vals[0]), 1e-10) * 100
            d0 = BACKEND_DISPLAY_NAMES.get(keys[0], keys[0])
            di = BACKEND_DISPLAY_NAMES.get(keys[i], keys[i])
            print(f"Cross-backend delta ({d0} vs {di}): {diff:.4f}%")
    print()


def _plot_grouped_bars(
    backend_results: dict[str, list[dict]],
    value_key: str,
    title: str,
    xlabel: str,
    filename: str,
    output_dir: str,
) -> None:
    backends = list(backend_results.keys())
    first_backend = backends[0]
    labels = [_short_label(r["label"]) for r in backend_results[first_backend]]
    n = len(labels)
    n_backends = len(backends)

    bar_height = 0.8 / n_backends
    fig, ax = plt.subplots(figsize=(14, max(8, n * 0.5 + 2)))

    for j, b in enumerate(backends):
        values = []
        for r in backend_results[b]:
            v = r.get(value_key)
            values.append(v if v is not None else 0)

        y_positions = np.arange(n) + j * bar_height - (n_backends - 1) * bar_height / 2
        display = BACKEND_DISPLAY_NAMES.get(b, b)
        color = BACKEND_COLORS.get(b, f"C{j}")
        ax.barh(y_positions, values, height=bar_height, label=display, color=color, alpha=0.85)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_delta_heatmap(
    backend_results: dict[str, list[dict]],
    output_dir: str,
) -> None:
    backends = list(backend_results.keys())
    first_backend = backends[0]
    labels = [_short_label(r["label"]) for r in backend_results[first_backend]]
    n = len(labels)

    deltas = {}
    for b in backends:
        baseline_obj = backend_results[b][0].get("objective")
        if baseline_obj is None or baseline_obj == 0:
            deltas[b] = [None] * n
            continue
        d = []
        for r in backend_results[b]:
            obj = r.get("objective")
            if obj is not None:
                d.append((obj - baseline_obj) / baseline_obj * 100)
            else:
                d.append(None)
        deltas[b] = d

    fig, ax = plt.subplots(figsize=(max(6, len(backends) * 2.5 + 2), max(8, n * 0.45 + 2)))

    data = np.zeros((n, len(backends)))
    for j, b in enumerate(backends):
        for i in range(n):
            val = deltas[b][i]
            data[i, j] = val if val is not None else 0

    vmax = max(abs(data.min()), abs(data.max())) if data.size > 0 else 1
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(backends)))
    ax.set_xticklabels(
        [BACKEND_DISPLAY_NAMES.get(b, b) for b in backends],
        fontsize=10,
    )
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(len(backends)):
            val = data[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=7, color=text_color)

    ax.set_title("Objective Delta (%) by Constraint Combination × Backend")
    fig.colorbar(im, ax=ax, label="Delta %", shrink=0.8)
    plt.tight_layout()

    path = os.path.join(output_dir, "cross_backend_delta_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_solve_time_comparison(
    backend_results: dict[str, list[dict]],
    output_dir: str,
) -> None:
    backends = list(backend_results.keys())
    first_backend = backends[0]
    labels = [_short_label(r["label"]) for r in backend_results[first_backend]]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, len(backends) * 2.5 + 2), max(8, n * 0.45 + 2)))

    data = np.zeros((n, len(backends)))
    for j, b in enumerate(backends):
        for i in range(n):
            data[i, j] = backend_results[b][i].get("solve_time_s", 0)

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(backends)))
    ax.set_xticklabels(
        [BACKEND_DISPLAY_NAMES.get(b, b) for b in backends],
        fontsize=10,
    )
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(len(backends)):
            val = data[i, j]
            text_color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}s", ha="center", va="center", fontsize=7, color=text_color)

    ax.set_title("Solve Time (s) by Constraint Combination × Backend")
    fig.colorbar(im, ax=ax, label="Time (s)", shrink=0.8)
    plt.tight_layout()

    path = os.path.join(output_dir, "cross_backend_solve_time_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-backend comparison of constraint toggle study results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="Paths to per-backend results.yaml files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/constraint_study/cross_backend",
        help="Directory for cross-backend comparison plots",
    )
    args = parser.parse_args()

    backend_results = _load_results(args.results)

    lengths = {b: len(v) for b, v in backend_results.items()}
    if len(set(lengths.values())) > 1:
        print(f"WARNING: backends have different result counts: {lengths}", file=sys.stderr)

    _print_cross_backend_table(backend_results)

    os.makedirs(args.output_dir, exist_ok=True)

    _plot_grouped_bars(
        backend_results,
        value_key="objective",
        title="Objective Value by Constraint Combination (Cross-Backend)",
        xlabel="Objective Value",
        filename="cross_backend_objective.png",
        output_dir=args.output_dir,
    )

    _plot_grouped_bars(
        backend_results,
        value_key="solve_time_s",
        title="Solve Time by Constraint Combination (Cross-Backend)",
        xlabel="Solve Time (s)",
        filename="cross_backend_solve_time.png",
        output_dir=args.output_dir,
    )

    _plot_delta_heatmap(backend_results, args.output_dir)
    _plot_solve_time_comparison(backend_results, args.output_dir)

    print(f"\nAll cross-backend plots saved to: {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
