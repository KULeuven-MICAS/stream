import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


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


def process_core_trace(pid, trace, M, N, K, m, n, k, output_base):  # noqa: PLR0913, N803
    time_differences = [end - start for start, end in zip(trace["INSTR_EVENT_0"], trace["INSTR_EVENT_1"], strict=False)]
    total_difference = trace["INSTR_EVENT_1"][-1] - trace["INSTR_EVENT_0"][0]
    num_kernels_this_core = len(time_differences)
    num_kernels_total = M * N * K // (m * n * k)
    core_parallelism = num_kernels_total // num_kernels_this_core
    peak_macs_per_cycle = 64  # Assuming 64 MACs/cycle/core

    print(f"Processing core_trace for PID: {pid}")
    print(f"Total difference: {total_difference} cycles")
    avg_diff = sum(time_differences) / len(time_differences)
    print(f"Average difference = {avg_diff}")

    # Calculate the macs/cycle
    macs_total_this_core = M * N * K // core_parallelism
    macs_kernel = m * n * k
    macs_per_cycle_system = macs_total_this_core / total_difference
    macs_per_cycle_kernel = macs_kernel / avg_diff
    print(f"MACs per cycle (kernel) = {macs_per_cycle_kernel:.2f}")

    # Calculate efficiencies
    kernel_efficiency = macs_per_cycle_kernel / peak_macs_per_cycle * 100
    system_efficiency = macs_per_cycle_system / peak_macs_per_cycle * 100
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

    # Save report
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
    # Ensure output_base exists as a directory
    os.makedirs(output_base, exist_ok=True)

    report_path = os.path.join(output_base, f"{tile_name}_report.json")
    save_report(report_path, report_data)

    fig_path = os.path.join(output_base, f"{tile_name}_plot.png")
    plot_time_differences(time_differences, fig_path, tile_name, num_kernels_this_core)


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
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    m, n, k = args.m, args.n, args.k

    input_file = args.input
    output_base = args.output

    print("=" * 80)
    traces = parse_perfetto_trace(input_file)

    for pid, trace in traces.items():
        if trace["name"] is None:
            print(f"Skipping PID {pid} with no name.")
            continue

        if "core_trace" in trace["name"]:
            process_core_trace(pid, trace, M, N, K, m, n, k, output_base)
        elif "shim_trace" in trace["name"] or "memtile_trace" in trace["name"]:
            print(f"Skipping {trace['name']} for PID {pid}.")
        else:
            print(f"Unknown trace type for PID {pid}: {trace['name']}")


if __name__ == "__main__":
    main()
