import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

RUN_DIR_PATTERN = re.compile(r"^(.+)-gemm_(\d+)_(\d+)_(\d+)-fused-constraint-optimization$")


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
    peak_macs_per_cycle = 64  # Assuming 64 MACs/cycle/core

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
    kernel_efficiency = (macs_per_cycle_kernel / peak_macs_per_cycle * 100) if peak_macs_per_cycle else 0.0
    system_efficiency = (macs_per_cycle_system / peak_macs_per_cycle * 100) if peak_macs_per_cycle else 0.0
    print(
        f"Theoretical peak efficiency (kernel) = "
        f"{kernel_efficiency:.1f} % "
        f"(assuming {peak_macs_per_cycle} MACs/cycle/core)."
    )
    print(f"MACs/cycle (system) = {macs_per_cycle_system:.2f}")
    print(
        f"Theoretical peak efficiency (system) = "
        f"{system_efficiency:.1f} % "
        f"(assuming {peak_macs_per_cycle} MACs/cycle/core)."
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


def write_details_markdown(output_base, hwid, M, N, K, tile_rows):  # noqa: N803
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
        f.write("### Details for Successful Runs\n\n")
        f.write(f"<details><summary><strong>[{title_hwid}] M={M} K={K} N={N}</strong></summary>\n\n")
        f.write(
            "| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) |"
            " Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |\n"
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
    parser.add_argument("--hwid", type=str, default=None, help="Hardware ID label for the details section (optional).")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    m, n, k = args.m, args.n, args.k
    hwid = args.hwid

    input_file = args.input
    output_base = args.output

    if hwid is None:
        hwid = infer_hwid(output_base)

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
    write_details_markdown(output_base, hwid, M, N, K, tile_rows)


if __name__ == "__main__":
    main()
