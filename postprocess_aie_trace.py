import json
import os
import os


def parse_perfetto_trace(file_path):
    with open(file_path, "r") as file:
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
        if len(instr_event_0_times) == len(instr_event_1_times) + 1:
            instr_event_0_times = instr_event_0_times[:-1]
        else:
            raise ValueError(
                f"Mismatched INSTR_EVENT_0 ({len(instr_event_0_times)}) and INSTR_EVENT_1 ({len(instr_event_1_times)}) events"
            )

    time_differences = [end - start for start, end in zip(instr_event_0_times, instr_event_1_times)]
    total_difference = instr_event_1_times[-1] - instr_event_0_times[0]

    return time_differences, total_difference


def plot_time_differences(time_differences, fig_path):
    import matplotlib.pyplot as plt

    plt.grid()
    plt.plot(list(range(len(time_differences))), time_differences, marker='o', linestyle='-', color='b')
    plt.xticks(ticks=list(range(len(time_differences))), labels=list(range(len(time_differences))))
    plt.xlabel("Event number")
    plt.ylabel("Time difference (cycles)")
    # Set figure size in pixels
    plt.gcf().set_size_inches(6, 4)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    M, N, K = 128, 128, 32  # total layer size
    m, n, k = 32, 32, 32  # tile size
    nb_kernels = (M // m) * (N // n) * (K // k)  # number of kernels
    MAX_MACS_PER_CYCLE_PER_CORE = 64  # for int16 x int16
    input_folder = "/home/asymons/Documents/traces/gemm/zero_events/"
    output_folder = "/home/asymons/Documents/traces/gemm/zero_events/plots/"
    # Automatically generate the output folder based on the input folder
    output_folder = os.path.join(input_folder, "plots")

    os.makedirs(output_folder, exist_ok=True)

    # Get all .json files from the input folder
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".json")]

    print("=" * 80)
    for file_path in file_paths:
        time_differences, total_difference = parse_perfetto_trace(file_path)
        if len(time_differences) != nb_kernels:
            raise ValueError(
                f"Expected {nb_kernels} time differences, but got {len(time_differences)} in {file_path}"
            )
        print(f"File: {file_path}")
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
            f"Theoretical peak efficiency (kernel) = {macs_per_cycle_kernel / MAX_MACS_PER_CYCLE_PER_CORE * 100:.1f} % (assuming {MAX_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core)"
        )
        print(f"MACs/cycle (system) = {macs_per_cycle_system:.2f}")
        print(
            f"Theoretical peak efficiency (system) = {macs_per_cycle_system / MAX_MACS_PER_CYCLE_PER_CORE * 100:.1f} % (assuming {MAX_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core)"
        )
        print("=" * 80)
        # Generate output PNG file path
        base_name = os.path.basename(file_path)
        png_name = os.path.splitext(base_name)[0] + ".png"
        fig_path = os.path.join(output_folder, png_name)

        # Plot and save the figure
        plot_time_differences(time_differences, fig_path)
