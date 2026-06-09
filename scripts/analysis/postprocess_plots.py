import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid", context="talk")


def make_dims(sweep_dim: str, sweep_val: int, const_val: int) -> tuple[int, int, int]:
    sweep_dim = sweep_dim.upper()
    if sweep_dim == "M":
        return sweep_val, const_val, const_val
    elif sweep_dim == "K":
        return const_val, sweep_val, const_val
    elif sweep_dim == "N":
        return const_val, const_val, sweep_val
    else:
        raise ValueError(f"Unknown sweep_dim: {sweep_dim}")


def read_efficiency_data(
    hw_id: str,
    sweep_dim: str,
    sweep_values: list[int],
    const_val: int,
    nb_rows: int,
    nb_cols: int,
    tile: str,
) -> tuple[list[int], list[float | None], list[float | None]]:
    x_vals, kernel_eff, system_eff = [], [], []

    for sv in sweep_values:
        M, K, N = make_dims(sweep_dim, sv, const_val)
        folder = f"outputs/{hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces"
        file_path = os.path.join(folder, f"{tile}_report.json")

        if not os.path.exists(folder):
            print(f"Note: folder missing (skipping): {folder}")
            x_vals.append(sv)
            kernel_eff.append(None)
            system_eff.append(None)
            continue

        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                x_vals.append(sv)
                kernel_eff.append(data.get("theoretical_peak_efficiency_kernel_percent"))
                system_eff.append(data.get("theoretical_peak_efficiency_system_percent"))
            except Exception as e:
                print(f"Warning: failed reading {file_path}: {e}")
                x_vals.append(sv)
                kernel_eff.append(None)
                system_eff.append(None)
        else:
            print(f"Note: report missing (skipping): {file_path}")
            x_vals.append(sv)
            kernel_eff.append(None)
            system_eff.append(None)

    return x_vals, kernel_eff, system_eff


def _find_known_indices(y: list[float | None]) -> list[int]:
    return [i for i, v in enumerate(y) if v is not None and not (isinstance(v, float) and np.isnan(v))]


def _gap_segments(y: list[float | None]) -> list[tuple[int, int]]:
    """
    Return list of (left_known_idx, right_known_idx) pairs that bound
    one or more consecutive missing values in between.
    Only gaps with known endpoints on BOTH sides are returned.
    """
    known = _find_known_indices(y)
    gaps = []
    max_gap = 2
    if len(known) < max_gap:
        return gaps
    for a, b in zip(known[:-1], known[1:], strict=False):
        if b - a > 1:
            gaps.append((a, b))
    return gaps


def _linear_values(x0, y0, x1, y1, xs: list[int]) -> list[float]:
    """Linear interpolation y(x) between (x0,y0) and (x1,y1) for integer xs."""
    xs_arr = np.array(xs, dtype=float)
    return list(y0 + (y1 - y0) * (xs_arr - x0) / (x1 - x0))


def _plot_with_missing(
    ax,
    positions: list[int],
    categories: list[str],
    y: list[float | None],
    label: str,
    color=None,
    marker_line="o",
    marker_miss="x",
):
    """
    Plot solid line for known points, dashed lines across gaps with red 'x' markers at missing positions
    using interpolated y between the gap endpoints. Does nothing for edge-missing values.
    """
    # Convert to numpy with NaNs for missing, so Matplotlib breaks the solid line at gaps
    y_np = np.array([np.nan if (v is None) else v for v in y], dtype=float)

    # Base solid line for known values
    ax.plot(positions, y_np, marker=marker_line, linewidth=2, label=label, color=color)

    # For each internal gap bounded by known points, draw dashed interpolation and red crosses
    for left, right in _gap_segments(y):
        x_sub = list(range(left, right + 1))
        y_left = y[left]
        y_right = y[right]
        # safety: both endpoints must be real numbers
        if y_left is None or y_right is None:
            continue
        y_sub = _linear_values(left, float(y_left), right, float(y_right), x_sub)

        # Dashed segment bridging the gap (including endpoints to make the segment continuous)
        ax.plot(x_sub, y_sub, linestyle="--", linewidth=1.5, color=color)

        # Mark only the missing internal points with a red cross
        miss_indices = x_sub[1:-1]  # exclude endpoints (they are known)
        miss_y = [y_sub[i - left] for i in miss_indices]
        if miss_indices:
            ax.scatter(miss_indices, miss_y, marker=marker_miss, color="red", zorder=5, label=None)


def plot_efficiency(
    x_axis_vals: list[int],
    kernel_eff: list[float | None],
    system_eff: list[float | None],
    sweep_dim: str,
    const_val: int,
    hw_id: str,
    tile: str,
    output_folder: str,
):
    if not x_axis_vals:
        print("No data found to plot.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Equal spacing using categorical labels: plot against integer positions, label with categories
    categories = [str(v) for v in x_axis_vals]
    positions = list(range(len(categories)))

    fig, ax = plt.subplots(figsize=(9, 6))

    _plot_with_missing(ax, positions, categories, kernel_eff, label="Kernel Efficiency (%)")
    _plot_with_missing(ax, positions, categories, system_eff, label="System Efficiency (%)", marker_line="s")

    ax.set_xlabel(f"{sweep_dim.upper()} Dimension")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title(f"GEMM Efficiency vs {sweep_dim.upper()} (other dims = {const_val}) on {hw_id}")
    # Filter out None values from kernel_eff for ylim calculation
    kernel_eff_valid = [v for v in kernel_eff if v is not None]
    if kernel_eff_valid:
        ax.set_ylim(bottom=0, top=1.05 * max(kernel_eff_valid))

    # Categorical ticks at equal spacing
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)

    ax.legend()
    fig.tight_layout()

    out_name = f"{hw_id}_gemm_efficiency_sweep-{sweep_dim.upper()}_const-{const_val}_{tile}.png"
    output_path = os.path.join(output_folder, out_name)
    fig.savefig(output_path)
    print(f"Figure saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GEMM efficiency while sweeping a single dimension (M, K, or N). "
        "Other two dimensions are set to a single constant value. "
        "Missing internal points are interpolated and shown with red crosses and dashed gaps."
    )
    parser.add_argument("--hw_id", type=str, default="single_core", help="Hardware system ID")
    parser.add_argument(
        "--sweep_dim", type=str, choices=["M", "K", "N", "m", "k", "n"], required=True, help="Which dimension to sweep"
    )
    parser.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        required=True,
        help="List of values for the swept dimension (e.g., --sweep 64 128 256)",
    )
    parser.add_argument("--const", type=int, required=True, help="Constant value used for the two non-swept dimensions")
    parser.add_argument("--row", type=int, default=1, help="Number of rows in the PE grid")
    parser.add_argument("--col", type=int, default=1, help="Number of columns in the PE grid")
    parser.add_argument("--tile", type=str, default="tile3,1", help="Tile identifier (e.g., tile3,1)")
    return parser.parse_args()


def main():
    args = parse_args()

    x_axis, kernel_eff, system_eff = read_efficiency_data(
        hw_id=args.hw_id,
        sweep_dim=args.sweep_dim,
        sweep_values=args.sweep,
        const_val=args.const,
        nb_rows=args.row,
        nb_cols=args.col,
        tile=args.tile,
    )

    output_folder = "outputs/plots/"

    # Informative notes for non-interpolatable edge-missing points
    known_idxs_k = _find_known_indices(kernel_eff)
    known_idxs_s = _find_known_indices(system_eff)
    if not known_idxs_k:
        print("Warning: kernel efficiency has no valid points; nothing to plot for that series.")
    if not known_idxs_s:
        print("Warning: system efficiency has no valid points; nothing to plot for that series.")

    plot_efficiency(
        x_axis_vals=x_axis,
        kernel_eff=kernel_eff,
        system_eff=system_eff,
        sweep_dim=args.sweep_dim,
        const_val=args.const,
        hw_id=args.hw_id,
        tile=args.tile,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    main()
