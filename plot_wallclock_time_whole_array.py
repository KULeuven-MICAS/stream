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


def read_wallclock_data(
    sweep_dim: str,
    sweep_values: list[int],
    const_val: int,
    nb_rows: int,
    nb_cols: int,
) -> tuple[list[int], list[float | None]]:
    x_vals, min_wallclock = [], []

    for sv in sweep_values:
        M, K, N = make_dims(sweep_dim, sv, const_val)
        folder = f"outputs/whole_array-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces"
        file_path = os.path.join(folder, "wall_clock_time.json")

        if not os.path.exists(folder):
            print(f"Note: folder missing (skipping): {folder}")
            x_vals.append(sv)
            min_wallclock.append(None)
            continue

        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                x_vals.append(sv)
                min_wallclock.append(data.get("min"))
            except Exception as e:
                print(f"Warning: failed reading {file_path}: {e}")
                x_vals.append(sv)
                min_wallclock.append(None)
        else:
            print(f"Note: report missing (skipping): {file_path}")
            x_vals.append(sv)
            min_wallclock.append(None)

    return x_vals, min_wallclock


def read_amd_wallclock_data(
    sweep_dim: str,
    sweep_values: list[int],
    const_val: int,
) -> list[float | None]:
    amd_min_wallclock = []
    for sv in sweep_values:
        # For AMD, K is the swept dimension
        if sweep_dim.upper() == "K":
            K = sv
        else:
            K = const_val
        amd_file = f"outputs/plots/amd/{K}/wall_clock_time.json"
        if os.path.exists(amd_file):
            try:
                with open(amd_file) as f:
                    data = json.load(f)
                amd_min_wallclock.append(data.get("min"))
            except Exception as e:
                print(f"Warning: failed reading {amd_file}: {e}")
                amd_min_wallclock.append(None)
        else:
            print(f"Note: AMD report missing (skipping): {amd_file}")
            amd_min_wallclock.append(None)
    return amd_min_wallclock


def _find_known_indices(y: list[float | None]) -> list[int]:
    return [i for i, v in enumerate(y) if v is not None and not (isinstance(v, float) and np.isnan(v))]


def _gap_segments(y: list[float | None]) -> list[tuple[int, int]]:
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
    y_np = np.array([np.nan if (v is None) else v for v in y], dtype=float)
    ax.plot(positions, y_np, marker=marker_line, linewidth=2, label=label, color=color)
    for left, right in _gap_segments(y):
        x_sub = list(range(left, right + 1))
        y_left = y[left]
        y_right = y[right]
        if y_left is None or y_right is None:
            continue
        y_sub = _linear_values(left, float(y_left), right, float(y_right), x_sub)
        ax.plot(x_sub, y_sub, linestyle="--", linewidth=1.5, color=color)
        miss_indices = x_sub[1:-1]
        miss_y = [y_sub[i - left] for i in miss_indices]
        if miss_indices:
            ax.scatter(miss_indices, miss_y, marker=marker_miss, color="red", zorder=5, label=None)


def plot_wallclock(
    x_axis_vals: list[int],
    min_wallclock: list[float | None],
    amd_wallclock: list[float | None],
    sweep_dim: str,
    const_val: int,
    nb_rows: int,
    nb_cols: int,
    output_folder: str,
):
    if not x_axis_vals:
        print("No data found to plot.")
        return

    os.makedirs(output_folder, exist_ok=True)

    categories = [str(v) for v in x_axis_vals]
    positions = list(range(len(categories)))

    fig, ax = plt.subplots(figsize=(9, 6))

    _plot_with_missing(ax, positions, categories, min_wallclock, label="STREAM-AIE (us)", color="blue")
    # AMD line (red, no interpolation)
    y_amd_np = np.array([np.nan if (v is None) else v for v in amd_wallclock], dtype=float)
    ax.plot(positions, y_amd_np, marker="s", linewidth=2, label="MLIR-AIE (us)", color="red")

    ax.set_xlabel(f"{sweep_dim.upper()} Dimension")
    ax.set_ylabel("STREAM-AIE (us)")
    ax.set_title(f"Min Wall Clock Time vs {sweep_dim.upper()} (other dims = {const_val}) on whole_array")
    min_wallclock_valid = [v for v in min_wallclock if v is not None]
    amd_wallclock_valid = [v for v in amd_wallclock if v is not None]
    all_valid = min_wallclock_valid + amd_wallclock_valid
    if all_valid:
        ax.set_ylim(bottom=0, top=1.05 * max(all_valid))

    ax.set_xticks(positions)
    ax.set_xticklabels(categories)

    ax.legend()
    fig.tight_layout()

    out_name = f"whole_array_gemm_wallclock_sweep-{sweep_dim.upper()}_const-{const_val}_{nb_rows}row_{nb_cols}col.png"
    output_path = os.path.join(output_folder, out_name)
    fig.savefig(output_path)
    print(f"Figure saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot min wall clock time while sweeping a single dimension (M, K, or N). "
        "Other two dimensions are set to a single constant value. "
        "Missing internal points are interpolated and shown with red crosses and dashed gaps."
    )
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
    return parser.parse_args()


def main():
    args = parse_args()

    x_axis, min_wallclock = read_wallclock_data(
        sweep_dim=args.sweep_dim,
        sweep_values=args.sweep,
        const_val=args.const,
        nb_rows=args.row,
        nb_cols=args.col,
    )

    amd_wallclock = read_amd_wallclock_data(
        sweep_dim=args.sweep_dim,
        sweep_values=args.sweep,
        const_val=args.const,
    )

    output_folder = "outputs/plots/"

    known_idxs = _find_known_indices(min_wallclock)
    if not known_idxs:
        print("Warning: min wall clock time has no valid points; nothing to plot.")

    plot_wallclock(
        x_axis_vals=x_axis,
        min_wallclock=min_wallclock,
        amd_wallclock=amd_wallclock,
        sweep_dim=args.sweep_dim,
        const_val=args.const,
        nb_rows=args.row,
        nb_cols=args.col,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    main()
