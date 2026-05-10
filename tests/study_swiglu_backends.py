"""Cross-backend solver study for SwiGLU workloads at multiple sizes.

Compares Gurobi, GSCIP, and HiGHS across SwiGLU configurations with
varying sequence lengths and tile sizes. Produces a results YAML and
comparison plots.

Usage:
    PYTHONPATH=. python tests/study_swiglu_backends.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import yaml

from stream.api import configure_logging, optimize_allocation_co
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload

ACCELERATOR = os.path.join(
    os.path.dirname(__file__),
    "../stream/inputs/aie/hardware/whole_array_strix.yaml",
)

BACKENDS = ["gurobi", "ortools_gscip", "ortools_highs"]
BACKEND_LABELS = {"gurobi": "Gurobi", "ortools_gscip": "GSCIP", "ortools_highs": "HiGHS"}
COLORS = {"gurobi": "#E24A33", "ortools_gscip": "#348ABD", "ortools_highs": "#988ED5"}

CONFIGS = [
    {"seq_len": 256, "embedding_dim": 2048, "hidden_dim": 8192, "rows": 4, "cols": 8},
    {"seq_len": 512, "embedding_dim": 2048, "hidden_dim": 8192, "rows": 4, "cols": 8},
    {"seq_len": 2048, "embedding_dim": 2048, "hidden_dim": 8192, "rows": 4, "cols": 8},
]

TILE_CONFIGS = [
    {"seq_len_tile": 16, "embedding_tile": 128, "hidden_tile": 32},
    {"seq_len_tile": 16, "embedding_tile": 128, "hidden_tile": 64},
    {"seq_len_tile": 32, "embedding_tile": 128, "hidden_tile": 32},
]


def _run_single(cfg, tiles, backend, output_root):  # noqa: PLR0913
    """Run a single SwiGLU configuration and return result dict."""
    seq, emb, hid = cfg["seq_len"], cfg["embedding_dim"], cfg["hidden_dim"]
    st, et, ht = tiles["seq_len_tile"], tiles["embedding_tile"], tiles["hidden_tile"]

    label = f"seq{seq}_emb{emb}_hid{hid}_tile{st}x{et}x{ht}_{backend}"
    print(f"  Running {label} ...", end=" ", flush=True)

    result = {
        "seq_len": seq,
        "embedding_dim": emb,
        "hidden_dim": hid,
        "tile_seq": st,
        "tile_emb": et,
        "tile_hid": ht,
        "backend": backend,
        "solver": BACKEND_LABELS.get(backend, backend),
        "status": "FAILED",
        "objective": None,
        "latency_total": None,
        "solve_time_s": None,
    }

    t0 = time.perf_counter()
    try:
        workload_path = make_swiglu_workload(seq, emb, hid, "bf16", "bf16")
        mapping_path = make_swiglu_mapping(seq, emb, hid, True, st, et, ht)

        hw_name = ACCELERATOR.split("/")[-1].split(".")[0]
        wl_name = re.split(r"/|\.", workload_path)[-1]
        if wl_name == "onnx":
            wl_name = re.split(r"/|\.", workload_path)[-2]
        experiment_id = f"{hw_name}-{wl_name}-{cfg['rows']}_row_{cfg['cols']}_col"

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(output_root or tmpdir, label)
            ctx = optimize_allocation_co(
                hardware=ACCELERATOR,
                workload=workload_path,
                mapping=mapping_path,
                experiment_id=experiment_id,
                output_path=out_path,
                skip_if_exists=False,
                trace_size=1048576,
                nb_cols_to_use=cfg["cols"],
                npu="npu2",
                backend=backend,
            )

        scheduler = ctx.get("scheduler")
        result["objective"] = float(scheduler.latency_total)
        result["latency_total"] = int(scheduler.latency_total)
        result["status"] = "OPTIMAL"

    except Exception as exc:
        result["status"] = f"ERROR: {str(exc)[:100]}"
        traceback.print_exc(file=sys.stderr)

    result["solve_time_s"] = round(time.perf_counter() - t0, 2)
    print(f"{result['status']} obj={result.get('latency_total', 'N/A')} t={result['solve_time_s']}s")
    return result


def run_study(output_root=None):
    """Run the full study and return list of result dicts."""
    results = []
    total = len(CONFIGS) * len(TILE_CONFIGS) * len(BACKENDS)
    i = 0

    for cfg in CONFIGS:
        for tiles in TILE_CONFIGS:
            tile_label = f"{tiles['seq_len_tile']}x{tiles['embedding_tile']}x{tiles['hidden_tile']}"
            print(f"\n=== SwiGLU {cfg['seq_len']}x{cfg['embedding_dim']}x{cfg['hidden_dim']} tiles={tile_label} ===")
            for backend in BACKENDS:
                i += 1
                print(f"[{i}/{total}]", end=" ")
                r = _run_single(cfg, tiles, backend, output_root)
                results.append(r)

    return results


def save_results(results, path):
    """Save results to YAML."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"\nResults saved to: {path}")


def plot_results(results, output_dir):  # noqa: PLR0912, PLR0915
    """Generate comparison plots from study results."""
    os.makedirs(output_dir, exist_ok=True)

    # Group results by (seq_len, tile_config)
    groups = {}
    for r in results:
        key = (r["seq_len"], f"{r['tile_seq']}x{r['tile_emb']}x{r['tile_hid']}")
        groups.setdefault(key, []).append(r)

    seq_lens = sorted(set(r["seq_len"] for r in results))
    tile_cfgs = sorted(set(f"{r['tile_seq']}x{r['tile_emb']}x{r['tile_hid']}" for r in results))
    n_tiles = len(tile_cfgs)
    n_seqs = len(seq_lens)

    # --- Plot 1: Solve time comparison (main result) ---
    fig, axes = plt.subplots(1, n_seqs, figsize=(5 * n_seqs, 6), sharey=False)
    if n_seqs == 1:
        axes = [axes]
    fig.suptitle("SwiGLU Solver Backend Comparison — Solve Time", fontsize=14, fontweight="bold", y=1.02)

    for ax_idx, seq in enumerate(seq_lens):
        ax = axes[ax_idx]
        x = np.arange(n_tiles)
        width = 0.25

        for b_idx, backend in enumerate(BACKENDS):
            times = []
            for tc in tile_cfgs:
                key = (seq, tc)
                group = groups.get(key, [])
                match = [r for r in group if r["backend"] == backend]
                t = match[0]["solve_time_s"] if match and match[0]["solve_time_s"] is not None else 0
                times.append(t)

            bars = ax.bar(
                x + b_idx * width,
                times,
                width,
                label=BACKEND_LABELS[backend],
                color=COLORS[backend],
                edgecolor="black",
                linewidth=0.5,
            )
            for bar, t in zip(bars, times, strict=False):
                if t > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{t:.0f}s",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.set_title(f"seq_len={seq}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Tile Configuration (seq×emb×hid)")
        if ax_idx == 0:
            ax.set_ylabel("Solve Time (seconds)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(tile_cfgs, fontsize=9)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path1 = os.path.join(output_dir, "swiglu_study_solve_time.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {path1}")

    # --- Plot 2: Objective comparison ---
    fig2, axes2 = plt.subplots(1, n_seqs, figsize=(5 * n_seqs, 6), sharey=False)
    if n_seqs == 1:
        axes2 = [axes2]
    fig2.suptitle("SwiGLU Solver Backend Comparison — Objective (Latency)", fontsize=14, fontweight="bold", y=1.02)

    for ax_idx, seq in enumerate(seq_lens):
        ax = axes2[ax_idx]
        x = np.arange(n_tiles)
        width = 0.25

        for b_idx, backend in enumerate(BACKENDS):
            objs = []
            for tc in tile_cfgs:
                key = (seq, tc)
                group = groups.get(key, [])
                match = [r for r in group if r["backend"] == backend]
                o = match[0]["objective"] if match and match[0]["objective"] is not None else 0
                objs.append(o)

            ax.bar(
                x + b_idx * width,
                objs,
                width,
                label=BACKEND_LABELS[backend],
                color=COLORS[backend],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_title(f"seq_len={seq}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Tile Configuration (seq×emb×hid)")
        if ax_idx == 0:
            ax.set_ylabel("Latency Total (cycles)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(tile_cfgs, fontsize=9)
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    path2 = os.path.join(output_dir, "swiglu_study_objective.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")

    # --- Plot 3: Summary table ---
    fig3, ax3 = plt.subplots(figsize=(16, max(4, len(results) * 0.35 + 1.5)))
    ax3.axis("off")
    ax3.set_title("SwiGLU Cross-Backend Study — Full Results", fontsize=14, fontweight="bold", pad=20)

    table_data = []
    for r in results:
        tile_label = f"{r['tile_seq']}x{r['tile_emb']}x{r['tile_hid']}"
        obj_str = f"{r['objective']:,.0f}" if r["objective"] is not None else "FAILED"

        gurobi_match = [
            x
            for x in results
            if x["seq_len"] == r["seq_len"]
            and x["tile_seq"] == r["tile_seq"]
            and x["tile_emb"] == r["tile_emb"]
            and x["tile_hid"] == r["tile_hid"]
            and x["backend"] == "gurobi"
            and x["objective"] is not None
        ]
        if gurobi_match and r["objective"] is not None:
            diff = abs(r["objective"] - gurobi_match[0]["objective"])
            vs = "identical" if diff <= 1 else f"{diff:,.0f} diff"
        else:
            vs = "N/A"

        gurobi_time = gurobi_match[0]["solve_time_s"] if gurobi_match and gurobi_match[0]["solve_time_s"] else None
        if gurobi_time and r["solve_time_s"] and r["backend"] != "gurobi":
            rel = f"{r['solve_time_s'] / gurobi_time:.1f}x"
        elif r["backend"] == "gurobi":
            rel = "baseline"
        else:
            rel = "N/A"

        table_data.append(
            [
                f"{r['seq_len']}",
                tile_label,
                r["solver"],
                obj_str,
                vs,
                f"{r['solve_time_s']}s" if r["solve_time_s"] else "N/A",
                rel,
            ]
        )

    table = ax3.table(
        cellText=table_data,
        colLabels=["seq_len", "Tiles", "Solver", "Objective", "vs Gurobi", "Time", "Relative"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for j in range(7):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(table_data, start=1):
        solver = row[2]
        for backend_key, label in BACKEND_LABELS.items():
            if solver == label and backend_key in COLORS:
                table[i, 2].set_facecolor(COLORS[backend_key] + "33")
        if row[4] == "identical":
            table[i, 4].set_facecolor("#C6EFCE")

    path3 = os.path.join(output_dir, "swiglu_study_table.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"Saved: {path3}")


if __name__ == "__main__":
    configure_logging()
    output_dir = "outputs/backend_study"
    os.makedirs(output_dir, exist_ok=True)

    results = run_study(output_root=output_dir)
    save_results(results, os.path.join(output_dir, "swiglu_study_results.yaml"))
    plot_results(results, output_dir)
