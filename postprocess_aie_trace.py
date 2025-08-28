import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

RUN_DIR_PATTERN = re.compile(r"^(.+)-gemm_(\d+)_(\d+)_(\d+)-fused-constraint-optimization$")
CLOCK_FREQUENCY_GHZ = 1.25  # 1.25 GHz (might be slightly off, but close enough for efficiency calc)
PEAK_MACS_PER_CYCLE_PER_CORE = 64  # if data type changes this value may need to be updated


def parse_perfetto_trace(file_path):
    with open(file_path) as file:
        data = json.load(file)

    traces = defaultdict(lambda: {"INSTR_EVENT_0": [], "INSTR_EVENT_1": [], "name": None})

    for event in data:
        pid = event.get("pid")
        if event.get("name") == "process_name" and event.get("ph") == "M":
            traces[pid]["name"] = event.get("args", {}).get("name")
        elif event.get("name") == "INSTR_EVENT_0" and event.get("ph") == "B":
            traces[pid]["INSTR_EVENT_0"].append(event.get("ts"))
        elif event.get("name") == "INSTR_EVENT_1" and event.get("ph") == "E":
            traces[pid]["INSTR_EVENT_1"].append(event.get("ts"))

    for pid, events in traces.items():
        if events["INSTR_EVENT_0"] and events["INSTR_EVENT_1"]:
            if len(events["INSTR_EVENT_0"]) != len(events["INSTR_EVENT_1"]):
                if len(events["INSTR_EVENT_0"]) == len(events["INSTR_EVENT_1"]) + 1:
                    last_difference = events["INSTR_EVENT_1"][-1] - events["INSTR_EVENT_0"][-2]
                    events["INSTR_EVENT_1"].append(events["INSTR_EVENT_0"][-1] + last_difference)
                else:
                    raise ValueError(
                        f"Mismatched INSTR_EVENT_0 ({len(events['INSTR_EVENT_0'])}) and "
                        f"INSTR_EVENT_1 ({len(events['INSTR_EVENT_1'])}) for PID {pid}"
                    )

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


def process_core_trace(pid, trace, M, N, K, m, n, k, output_base):  # noqa: PLR0913, N803
    time_differences = [end - start for start, end in zip(trace["INSTR_EVENT_0"], trace["INSTR_EVENT_1"], strict=False)]
    total_difference = trace["INSTR_EVENT_1"][-1] - trace["INSTR_EVENT_0"][0]
    num_kernels_this_core = len(time_differences)
    num_kernels_total = M * N * K // (m * n * k)
    core_parallelism = max(1, num_kernels_total // max(1, num_kernels_this_core))

    print(f"Processing core_trace for PID: {pid}")
    print(f"Total difference: {total_difference} cycles")
    avg_diff = sum(time_differences) / len(time_differences)
    print(f"Average difference = {avg_diff}")

    # Calculate the macs/cycle
    macs_total_this_core = M * N * K // core_parallelism
    macs_kernel = m * n * k
    macs_per_cycle_system = macs_total_this_core / total_difference if total_difference else 0.0
    macs_per_cycle_kernel = macs_kernel / avg_diff if avg_diff else 0.0
    print(f"MACs per cycle (kernel) = {macs_per_cycle_kernel:.2f}")

    # Calculate efficiencies
    kernel_efficiency = (
        (macs_per_cycle_kernel / PEAK_MACS_PER_CYCLE_PER_CORE * 100) if PEAK_MACS_PER_CYCLE_PER_CORE else 0.0
    )
    system_efficiency = (
        (macs_per_cycle_system / PEAK_MACS_PER_CYCLE_PER_CORE * 100) if PEAK_MACS_PER_CYCLE_PER_CORE else 0.0
    )
    print(
        f"Theoretical peak efficiency (kernel) = "
        f"{kernel_efficiency:.1f} % "
        f"(assuming {PEAK_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core)."
    )
    print(f"MACs/cycle (system) = {macs_per_cycle_system:.2f}")
    print(
        f"Theoretical peak efficiency (system) = "
        f"{system_efficiency:.1f} % "
        f"(assuming {PEAK_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core)."
    )
    print("=" * 80)

    # Extract tile name
    tile_name = trace["name"].split(" for ")[-1]

    # Save per-tile JSON report
    report_data = {
        "pid": pid,
        "tile_name": tile_name,
        "num_kernels": num_kernels_this_core,
        "total_kernel_time_cycles": total_difference,
        "average_kernel_time_cycles": avg_diff,
        "macs_per_cycle_kernel": macs_per_cycle_kernel,
        "theoretical_peak_efficiency_kernel_percent": kernel_efficiency,
        "macs_per_cycle_system": macs_per_cycle_system,
        "theoretical_peak_efficiency_system_percent": system_efficiency,
    }
    os.makedirs(output_base, exist_ok=True)

    report_path = os.path.join(output_base, f"{tile_name}_report.json")
    save_report(report_path, report_data)

    fig_path = os.path.join(output_base, f"{tile_name}_plot.png")
    plot_time_differences(time_differences, fig_path, tile_name, num_kernels_this_core)

    # Return a tuple for the run-level details.md generation
    return (
        tile_name,
        num_kernels_this_core,
        total_difference,
        avg_diff,
        macs_per_cycle_kernel,
        kernel_efficiency,
        macs_per_cycle_system,
        system_efficiency,
    )


def infer_hwid(output_base):
    base = os.path.basename(os.path.normpath(output_base))
    m = RUN_DIR_PATTERN.match(base)
    if m:
        hwid = m.group(1)
        return hwid
    return None


def write_details_markdown(output_base, hwid, M, N, K, tile_rows, wall_clock_time_us):  # noqa: N803
    """
    Write {output_base}/details.md containing a single <details> block with a tile table.
    Rows are sorted by macs_per_cycle_system descending for quick glance.
    """
    if not tile_rows:
        return

    # Sort rows by system MACs/cycle (index 6) high to low
    tile_rows = sorted(tile_rows, key=lambda r: (r[6] if r[6] is not None else -1e9), reverse=True)

    details_path = os.path.join(output_base, "details.md")
    with open(details_path, "w") as f:
        title_hwid = hwid or "?"
        f.write(f"<details><summary><strong>[{title_hwid}] M={M} K={K} N={N}</strong></summary>\n\n")
        for stat in wall_clock_time_us:
            f.write(f"- {stat.capitalize()} Wall clock time = {wall_clock_time_us[stat]} us\n")
        f.write(
            "| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) |"
            " Eff. kernel % | MACs/cycle (system) | Eff. system % |\n"
        )
        f.write(
            "|------|---------|--------------|-----------------------|---------------------|"
            "--------------------|---------------------|--------------------|\n"
        )
        for tile_name, nk, tcy, acy, mpck, peffk, mpcs, peffs in tile_rows:
            f.write(
                f"| {tile_name} | {nk} | {fmt_int(tcy)} | {fmt_float(acy)} | "
                f"{fmt_float(mpck)} | {fmt_float(peffk)} | {fmt_float(mpcs)} | {fmt_float(peffs)} |\n"
            )
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
    parser.add_argument("--hwid", type=str, default=None, help="Hardware ID label for the details section (optional).")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    m, n, k = args.m, args.n, args.k
    hwid = args.hwid
    nb_rows, nb_cols = args.row, args.col

    input_file = args.input
    output_base = args.output

    if hwid is None:
        hwid = infer_hwid(output_base)

    # Wall clock time (from run_trace.log)
    wall_clock_time_stats = ["avg", "min", "max"]
    wall_clock_time_us = parse_wall_clock_time_us(output_base, wall_clock_time_stats, M, K, N, nb_rows, nb_cols)
    for wall_clock_time_stat in wall_clock_time_stats:
        print(f"{wall_clock_time_stat.capitalize()} Wall clock time = {wall_clock_time_us[wall_clock_time_stat]} us")

    print("=" * 80)
    traces = parse_perfetto_trace(input_file)
    # Collect rows for the run-level details.md
    tile_rows = []
    for pid, trace in traces.items():
        if trace["name"] is None:
            print(f"Skipping PID {pid} with no name.")
            continue
        if "core_trace" in trace["name"]:
            row = process_core_trace(pid, trace, M, N, K, m, n, k, output_base)
            tile_rows.append(row)
        elif "shim_trace" in trace["name"] or "memtile_trace" in trace["name"]:
            print(f"Skipping {trace['name']} for PID {pid}.")
        else:
            print(f"Unknown trace type for PID {pid}: {trace['name']}")

    # Write the run-level Markdown details block
    write_details_markdown(output_base, hwid, M, N, K, tile_rows, wall_clock_time_us)


if __name__ == "__main__":
    main()
