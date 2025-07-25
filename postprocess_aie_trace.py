import argparse
import json

import matplotlib.pyplot as plt


def parse_perfetto_trace(file_path):
    with open(file_path) as file:
        data = json.load(file)

    instr_event_0_times = []
    instr_event_1_times = []

    for event in data:
        if event.get("name") == "INSTR_EVENT_0" and event.get("ph") == "B":
            instr_event_0_times.append(event.get("ts"))
        elif event.get("name") == "INSTR_EVENT_1" and event.get("ph") == "E":
            instr_event_1_times.append(event.get("ts"))

    if len(instr_event_0_times) != len(instr_event_1_times):
        if len(instr_event_0_times) == len(instr_event_1_times) + 1:
            instr_event_0_times = instr_event_0_times[:-1]
        else:
            raise ValueError(
                f"Mismatched INSTR_EVENT_0 ({len(instr_event_0_times)}) and INSTR_EVENT_1 ({len(instr_event_1_times)})"
            )

    time_differences = [end - start for start, end in zip(instr_event_0_times, instr_event_1_times, strict=False)]
    total_difference = instr_event_1_times[-1] - instr_event_0_times[0]

    return time_differences, total_difference


def plot_time_differences(time_differences, fig_path):
    plt.grid()
    plt.plot(list(range(len(time_differences))), time_differences, marker="o", linestyle="-", color="b")
    plt.xticks(ticks=list(range(len(time_differences))), labels=list(range(len(time_differences))))
    plt.xlabel("Event number")
    plt.ylabel("Time difference (cycles)")
    # Set figure size in pixels
    plt.gcf().set_size_inches(6, 4)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Postprocess AIE trace data.")
    parser.add_argument("--input", required=True, help="Path to the input .json trace file.")
    parser.add_argument("--output", required=True, help="Path to the output .png file for saving the plot.")
    parser.add_argument("--fig", required=True, help="Path to the output .json file for saving the report.")
    parser.add_argument("--M", type=int, required=True, help="Total layer size M.")
    parser.add_argument("--N", type=int, required=True, help="Total layer size N.")
    parser.add_argument("--K", type=int, required=True, help="Total layer size K.")
    parser.add_argument("--m", type=int, required=True, help="Tile size m.")
    parser.add_argument("--n", type=int, required=True, help="Tile size n.")
    parser.add_argument("--k", type=int, required=True, help="Tile size k.")
    args = parser.parse_args()

    def save_report(report_path, report_data):
        with open(report_path, "w") as report_file:
            json.dump(report_data, report_file, indent=4)

    M, N, K = args.M, args.N, args.K
    m, n, k = args.m, args.n, args.k
    nb_kernels = (M // m) * (N // n) * (K // k)  # number of kernels
    MAX_MACS_PER_CYCLE_PER_CORE = 64  # for int16 x int16

    input_file = args.input
    report_path = args.output
    fig_path = args.fig

    print("=" * 80)
    time_differences, total_difference = parse_perfetto_trace(input_file)
    if len(time_differences) != nb_kernels:
        raise ValueError(f"Expected {nb_kernels} time differences, but got {len(time_differences)} in {input_file}")
    print(f"File: {input_file}")
    print(f"Total difference: {total_difference} cycles")
    avg_diff = sum(time_differences) / len(time_differences)
    print(f"Average difference = {avg_diff}")

    # Calculate the macs/cycle
    macs_total = M * N * K
    macs_kernel = m * n * k
    macs_per_cycle_system = macs_total / total_difference
    macs_per_cycle_kernel = macs_kernel / avg_diff
    print(f"MACs per cycle (kernel) = {macs_per_cycle_kernel:.2f}")
    print(
        f"Theoretical peak efficiency (kernel) = "
        f"{macs_per_cycle_kernel / MAX_MACS_PER_CYCLE_PER_CORE * 100:.1f} % "
        f"(assuming {MAX_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core)."
    )
    print(f"MACs/cycle (system) = {macs_per_cycle_system:.2f}")
    print(
        f"Theoretical peak efficiency (system) = "
        f"{macs_per_cycle_system / MAX_MACS_PER_CYCLE_PER_CORE * 100:.1f} % "
        f"(assuming {MAX_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core)."
    )
    print("=" * 80)

    # Save report
    report_data = {
        "file": input_file,
        "total_difference_cycles": total_difference,
        "average_difference_cycles": avg_diff,
        "macs_per_cycle_kernel": macs_per_cycle_kernel,
        "theoretical_peak_efficiency_kernel_percent": macs_per_cycle_kernel / MAX_MACS_PER_CYCLE_PER_CORE * 100,
        "macs_per_cycle_system": macs_per_cycle_system,
        "theoretical_peak_efficiency_system_percent": macs_per_cycle_system / MAX_MACS_PER_CYCLE_PER_CORE * 100,
    }
    save_report(report_path, report_data)

    # Plot and save the figure
    plot_time_differences(time_differences, fig_path)


if __name__ == "__main__":
    main()
