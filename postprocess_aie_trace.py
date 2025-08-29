import argparse
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable
from statistics import median
from typing import TypeAlias

import matplotlib.pyplot as plt

CLOCK_FREQUENCY_GHZ = 1.25  # 1.25 GHz (might be slightly off, but close enough for efficiency calc)
PEAK_MACS_PER_CYCLE_PER_CORE = 64  # if data type changes this value may need to be updated


def fill_missing_equally_spaced(xs, required_len=None, tol_ratio=0.25):  # noqa: PLR0911, PLR0912
    """
    Fill missing integers in a roughly equally-spaced *sorted* list `xs`.
    Assumes no consecutive misses within the observed list.

    required_len: if provided, after filling inferred gaps, ensure the final
                  length equals this. If we're exactly one short, assume the
                  last item was missing and append it using the last difference.
    tol_ratio: tolerance as a fraction of the nominal step when deciding if a
               gap matches an integer multiple of the step.
    """
    if len(xs) == required_len:
        return xs
    if not xs:
        # If we must hit a required_len with no data, we can't infer values robustly.
        return []
    if len(xs) < 2:  # noqa: PLR2004
        out = sorted(xs)
        # If one-off short and we have only one number, we can't infer a last diff;
        # leave as-is (or the caller can handle).
        if required_len is not None and len(out) + 1 == required_len:
            # Best effort: duplicate spacing using step=1
            out.append(out[-1] + 1)
        return out

    xs = sorted(xs)
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] > xs[i]]
    if not diffs:
        return sorted(dict.fromkeys(xs))

    step_est = median(diffs)
    step = max(1, int(round(step_est)))  # keep integer spacing
    tol = max(1, int(round(tol_ratio * step)))

    out = [xs[0]]
    for a, b in zip(xs, xs[1:], strict=False):
        if b <= a:
            if b != out[-1]:
                out.append(b)
            continue
        d = b - a
        k = int(round(d / step))  # expected number of steps across the gap
        if k >= 2 and abs(d - k * step) <= tol:  # noqa: PLR2004
            # Insert k-1 interior points
            for j in range(1, k):
                cand = a + j * step
                if cand > out[-1] and cand < b:
                    out.append(int(cand))
            out.append(b)
        else:
            out.append(b)

    # De-dup and sort defensively
    out = sorted(dict.fromkeys(out))

    # Post-check against required_len
    if required_len is not None:
        if len(out) == required_len:
            return out
        if len(out) + 1 == required_len:
            # Assume the *last* item is missing; use the last observed difference
            if len(out) >= 2:  # noqa: PLR2004
                last_diff = out[-1] - out[-2]
                # Fallback to nominal step if last_diff is zero (shouldn't happen, but safe)
                if last_diff == 0:
                    last_diff = step
                out.append(out[-1] + last_diff)
                return out
            else:
                # If we somehow have <2 points, fall back to step
                out.append(out[-1] + step)
                return out
        # If the mismatch is larger than one, leave as-is (robust: don't over-infer)
        # You could raise or log here depending on your needs.

    return out


def parse_perfetto_trace(file_path, expected_nb_kernels_per_core):
    with open(file_path) as file:
        data = json.load(file)

    traces = defaultdict(lambda: {"starts": [], "ends": [], "name": None})

    for event in data:
        pid = event.get("pid")
        if event.get("name") == "process_name" and event.get("ph") == "M":
            traces[pid]["name"] = event.get("args", {}).get("name")
        elif event.get("name") == "INSTR_EVENT_0" and event.get("ph") == "B":
            traces[pid]["starts"].append(event.get("ts"))
        elif event.get("name") == "INSTR_EVENT_1" and event.get("ph") == "E":
            traces[pid]["ends"].append(event.get("ts"))

    for _, events in traces.items():
        if events["starts"] and events["ends"]:
            updated_starts = fill_missing_equally_spaced(events["starts"], required_len=expected_nb_kernels_per_core)
            updated_ends = fill_missing_equally_spaced(events["ends"], required_len=expected_nb_kernels_per_core)
            events["INSTR_EVENT_0"] = updated_starts
            events["INSTR_EVENT_1"] = updated_ends

    return traces


def parse_wall_clock_time_us(output_base, stats, M, K, N, nb_rows, nb_cols) -> dict[str, int]:  # noqa: N803
    """
    Extracts the wall clock time (us) for each requested stat from run_trace.log.
    stats: list containing "avg", "min", and/or "max" to select which values to extract.
    Returns a dict mapping stat name to time in us.
    """
    # Remove trailing 'traces/' if present in output_base
    base_dir = output_base
    if base_dir.endswith("traces") or base_dir.endswith("traces/"):
        base_dir = os.path.dirname(base_dir.rstrip("/"))
    log_path = os.path.join(base_dir, "run_trace.log")
    if not os.path.exists(log_path):
        raise ValueError(f"run_trace.log not found at {log_path}. Cannot determine wall clock time.")

    stat_map = {
        "avg": "Avg NPU matmul time:",
        "min": "Min NPU matmul time:",
        "max": "Max NPU matmul time:",
    }

    results = defaultdict(lambda: -1)
    with open(log_path) as log_file:
        for line in log_file:
            for stat in stats:
                search_str = stat_map.get(stat.lower())
                if search_str and search_str in line:
                    match = re.search(r"(\d+)us", line)
                    if match:
                        results[stat] = int(match.group(1))
    # Check for missing stats
    if len(results) != len(stats):
        missing = [stat for stat in stats if results[stat] == -1]
        raise ValueError(f"Wall clock time(s) not found for: {', '.join(missing)} in {log_path}.")

    # Compute the gMACs/s for all stats
    gmacs_per_sec = defaultdict(lambda: -1.0)
    for stat in stats:
        time_us = results[stat]
        if time_us > 0:
            gmacs_per_sec[stat] = (M * K * N) / time_us / 1_000.0  # assume time_us in microseconds

    # Compute the wall clock efficiency assuming 64 MACs/cycle and 1.25 GHz clock
    efficiency = defaultdict(lambda: -1.0)
    peak_macs_per_cycle = PEAK_MACS_PER_CYCLE_PER_CORE * nb_rows * nb_cols  # 64 MACs/cycle/core * rows * cols

    peak_gmacs_per_sec = peak_macs_per_cycle * CLOCK_FREQUENCY_GHZ
    for stat in stats:
        if gmacs_per_sec[stat] > 0:
            efficiency[stat] = (gmacs_per_sec[stat] / peak_gmacs_per_sec) * 100.0

    # Print the results
    for stat in stats:
        print(
            f"{stat.capitalize()} Wall clock time = {results[stat]} us, "
            f"{gmacs_per_sec[stat]:.2f} gMACs/s, "
            f"Efficiency = {efficiency[stat]:.1f} % (assuming 64 MACs/cycle at 1.25 GHz)."
        )

    # Save it to a json (similar to tile reports)
    wall_clock_json_path = os.path.join(output_base, "wall_clock_time.json")
    with open(wall_clock_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    return results


def calculate_time_differences(traces):
    results = {}
    for pid, events in traces.items():
        if len(events["INSTR_EVENT_0"]) != len(events["INSTR_EVENT_1"]):
            raise ValueError(
                f"Mismatched INSTR_EVENT_0 ({len(events['INSTR_EVENT_0'])}) and "
                f"INSTR_EVENT_1 ({len(events['INSTR_EVENT_1'])}) for PID {pid}"
            )

        time_differences = [
            end - start for start, end in zip(events["INSTR_EVENT_0"], events["INSTR_EVENT_1"], strict=False)
        ]
        total_difference = events["INSTR_EVENT_1"][-1] - events["INSTR_EVENT_0"][0]
        results[pid] = {
            "time_differences": time_differences,
            "total_difference": total_difference,
            "num_kernels": len(time_differences),
        }
    return results


def plot_time_differences(time_differences, fig_path, pid, num_kernels):
    plt.grid()
    plt.plot(list(range(len(time_differences))), time_differences, marker="o", linestyle="-", color="b")
    plt.xticks(ticks=list(range(len(time_differences))), labels=[str(i) for i in range(len(time_differences))])
    plt.xlabel("Event number")
    plt.ylabel("Time difference (cycles)")
    plt.title(f"Time Differences for PID {pid} (Kernels: {num_kernels})")
    plt.gcf().set_size_inches(6, 4)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def save_report(report_path, report_data):
    with open(report_path, "w") as report_file:
        json.dump(report_data, report_file, indent=4)


def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)


def fmt_float(x, digits=2):
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return str(x)


def get_valid_report_data(  # noqa: PLR0911, PLR0912, N803
    pid: int,
    trace: dict,
    m: int,
    n: int,
    k: int,
    nb_kernels_per_core: int,
) -> dict:
    # Total runtime = very first start to very last end
    total_difference = trace["INSTR_EVENT_1"][-1] - trace["INSTR_EVENT_0"][0]

    print(f"Processing core_trace for PID: {pid}")
    print(f"Total difference (first start → last end): {total_difference} cycles")

    # Define a uniform "average per-kernel time" based on the total span
    avg_diff = (total_difference / nb_kernels_per_core) if nb_kernels_per_core else 0.0
    print(f"Average difference (uniform over kernels) = {avg_diff}")

    # MACs:
    # - per-kernel MACs (for reporting only)
    macs_per_kernel = m * n * k
    # - total MACs attributable to this core during the whole run
    macs_total_this_core = macs_per_kernel * nb_kernels_per_core

    # Since we ignore per-kernel variability, both kernel/system MACs-per-cycle
    # are based on the same total runtime window.
    macs_per_cycle_system = (macs_total_this_core / total_difference) if total_difference else 0.0

    print(f"MACs per cycle (system) = {macs_per_cycle_system:.2f}")

    # Efficiencies relative to peak
    efficiency = (macs_per_cycle_system / PEAK_MACS_PER_CYCLE_PER_CORE * 100) if PEAK_MACS_PER_CYCLE_PER_CORE else 0.0

    print(f"Efficiency = {efficiency:.1f} % (assuming {PEAK_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core).")
    print("=" * 80)

    # Extract tile name
    tile_name = trace["name"].split(" for ")[-1]
    print(f"Valid data for PID {pid}, tile {tile_name}. Processing...")
    # Save per-tile JSON report
    report_data = {
        "pid": pid,
        "tile_name": tile_name,
        "num_kernels": nb_kernels_per_core,
        "total_kernel_time_cycles": total_difference,
        "average_kernel_time_cycles": avg_diff,
        "macs_per_cycle": macs_per_cycle_system,
        "efficiency_percent": efficiency,
    }
    return report_data


def get_invalid_report_data(pid, tile_name):
    print(f"WARNING: No valid data for PID {pid}, tile {tile_name}. Saving invalid data (-1).")
    return {
        "pid": pid,
        "tile_name": tile_name,
        "num_kernels": -1,
        "total_kernel_time_cycles": -1,
        "average_kernel_time_cycles": -1,
        "macs_per_cycle": -1,
        "efficiency_percent": -1,
    }


def process_core_trace(pid, trace, M, N, K, m, n, k, nb_kernels_per_core, output_base):  # noqa: PLR0913, N803
    tile_name = trace["name"].split(" for ")[-1]
    if not trace["starts"] or not trace["ends"]:
        report_data = get_invalid_report_data(pid, tile_name)
    else:
        report_data = get_valid_report_data(pid, trace, m, n, k, nb_kernels_per_core)

    os.makedirs(output_base, exist_ok=True)

    report_path = os.path.join(output_base, f"{tile_name}_report.json")
    save_report(report_path, report_data)

    # Return a tuple for the run-level details.md generation
    return report_data


# Local safe formatters (fall back if global helpers not present)
def _fmt_int(x):
    try:
        return fmt_int(x)  # type: ignore[name-defined]
    except Exception:
        return "-" if x is None or x == -1 else f"{int(x):,}"


def _fmt_float(x, digits: int = 3):
    try:
        return fmt_float(x)  # type: ignore[name-defined]
    except Exception:
        if x is None or x == -1:
            return "-"
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return "-"


# Column specification: (Header, key_in_report_data, formatter)
# This provides a single source of truth translating dict keys to table columns.
Column: TypeAlias = tuple[str, str, Callable[[object], str]]
COLUMNS: list[Column] = [
    ("Tile", "tile_name", lambda v: str(v) if v not in (None, -1) else "-"),
    ("Kernels", "num_kernels", _fmt_int),
    ("Total cycles", "total_kernel_time_cycles", _fmt_int),
    ("Avg cycles per kernel", "average_kernel_time_cycles", lambda v: _fmt_float(v, 3)),
    ("MACs/cycle", "macs_per_cycle", lambda v: _fmt_float(v, 3)),
    ("Efficiency %", "efficiency_percent", lambda v: _fmt_float(v, 2)),
]


def write_details_markdown(
    output_base: str,
    hwid: str,
    M: int,  # noqa: N803
    N: int,  # noqa: N803
    K: int,  # noqa: N803
    report_rows: list[dict],
    wall_clock_time_us: dict[str, float],
) -> None:
    """
    Write {output_base}/details.md containing a single <details> block with a tile table.
    Rows are sorted by 'macs_per_cycle_system' descending for quick glance.
    Expects report_rows as a list of dicts like:
      {
        "pid": ...,
        "tile_name": ...,
        "num_kernels": ...,
        "average_kernel_time_cycles": ...,
        "macs_per_cycle": ...,
        "efficiency_percent": ...,
      }
    """

    if not report_rows:
        return

    # Sort rows by system MACs/cycle high to low; treat None/-1 as very small
    def _sort_key(d: dict) -> float:
        v = d.get("macs_per_cycle_system", None)
        if v is None or v == -1:
            return -1e9
        try:
            return float(v)
        except Exception:
            return -1e9

    report_rows = sorted(report_rows, key=_sort_key, reverse=True)

    details_path = os.path.join(output_base, "details.md")
    with open(details_path, "w") as f:
        f.write(f"<details><summary><strong>[{hwid}] M={M} K={K} N={N}</strong></summary>\n\n")

        # Print wall clock times in a stable, readable order if known keys exist
        preferred_order = ["min", "avg", "max", "p50", "p90", "p95", "p99", "total"]
        printed = set()
        for stat in preferred_order:
            if stat in wall_clock_time_us:
                f.write(f"- {stat.capitalize()} Wall clock time = {wall_clock_time_us[stat]} us\n")
                printed.add(stat)
        # Any remaining keys (deterministic order)
        for stat in sorted(k for k in wall_clock_time_us.keys() if k not in printed):
            f.write(f"- {stat.capitalize()} Wall clock time = {wall_clock_time_us[stat]} us\n")
        f.write("\n")

        # Header
        headers = " | ".join(h for h, _, _ in COLUMNS)
        f.write(f"| {headers} |\n")
        f.write(f"|{'|'.join('-' * len(h) for h, _, _ in COLUMNS)}|\n")

        # Rows
        for row in report_rows:
            cells = []
            for _, key, formatter in COLUMNS:
                value = row.get(key, None)
                cells.append(formatter(value))
            f.write(f"| {' | '.join(cells)} |\n")

        f.write("\n</details>\n")


def main():
    parser = argparse.ArgumentParser(description="Postprocess AIE trace data.")
    parser.add_argument("--input", required=True, help="Path to the input .json trace file.")
    parser.add_argument("--output", required=True, help="Base path for the output files (e.g., report and plots).")
    parser.add_argument("--M", type=int, required=True, help="Total layer size M.")
    parser.add_argument("--N", type=int, required=True, help="Total layer size N.")
    parser.add_argument("--K", type=int, required=True, help="Total layer size K.")
    parser.add_argument("--m", type=int, required=True, help="Tile size m.")
    parser.add_argument("--n", type=int, required=True, help="Tile size n.")
    parser.add_argument("--k", type=int, required=True, help="Tile size k.")
    parser.add_argument("--row", type=int, required=True, help="Number of rows used.")
    parser.add_argument("--col", type=int, required=True, help="Number of columns used.")
    parser.add_argument("--hwid", type=str, required=True, help="Hardware ID label for the details section (optional).")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    m, n, k = args.m, args.n, args.k
    hwid = args.hwid
    nb_rows, nb_cols = args.row, args.col

    input_file = args.input
    output_base = args.output

    # Wall clock time (from run_trace.log)
    wall_clock_time_stats = ["avg", "min", "max"]
    wall_clock_time_us = parse_wall_clock_time_us(output_base, wall_clock_time_stats, M, K, N, nb_rows, nb_cols)

    nb_kernels_per_core = (M * N * K) // (m * n * k) // (nb_rows * nb_cols)

    traces = parse_perfetto_trace(input_file, nb_kernels_per_core)
    print("=" * 80)
    # Collect rows for the run-level details.md
    tile_rows = []
    for pid, trace in traces.items():
        if trace["name"] is None:
            print(f"Skipping PID {pid} with no name.")
            continue
        if "core_trace" in trace["name"]:
            row = process_core_trace(pid, trace, M, N, K, m, n, k, nb_kernels_per_core, output_base)
            tile_rows.append(row)
        elif "shim_trace" in trace["name"] or "memtile_trace" in trace["name"]:
            print(f"Skipping {trace['name']} for PID {pid}.")
        else:
            print(f"Unknown trace type for PID {pid}: {trace['name']}")

    # Write the run-level Markdown details block
    write_details_markdown(output_base, hwid, M, N, K, tile_rows, wall_clock_time_us)


if __name__ == "__main__":
    main()
