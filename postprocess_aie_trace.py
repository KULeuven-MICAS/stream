import json
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
    plt.plot(list(range(len(time_differences))), time_differences)
    plt.xlabel("Event number")
    plt.ylabel("Time difference (cycles)")
    # Set figure size in pixels
    plt.gcf().set_size_inches(4, 3)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    M, N, K = 128, 128, 32
    MAX_MACS_PER_CYCLE_PER_CORE = 64  # for int16 x int16
    input_folder = "/home/asymons/Documents/traces/gemm/stream_squashed_dma_copies"
    output_folder = "/home/asymons/Documents/traces/gemm/stream_squashed_dma_copies/png_outputs"

    os.makedirs(output_folder, exist_ok=True)

    # Get all .json files from the input folder
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".json")]

    for file_path in file_paths:
        time_differences, total_difference = parse_perfetto_trace(file_path)
        # for i, diff in enumerate(time_differences):
        #     print(f"File: {file_path}, Event {i}: {diff} cycles")
        print(f"File: {file_path}, Total difference: {total_difference} cycles")
        avg_diff = sum(time_differences) / len(time_differences)
        print(f"File: {file_path}, Average difference = {avg_diff}")

        # Calculate the macs/cycle
        macs = M * N * K
        macs_per_cycle = macs / total_difference
        print(f"File: {file_path}, MACs/cycle = {macs_per_cycle}")
        print(
            f"Theoretical peak efficiency = {macs_per_cycle / MAX_MACS_PER_CYCLE_PER_CORE * 100:.1f} % (assuming {MAX_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core)"
        )
        # Generate output PNG file path
        base_name = os.path.basename(file_path)
        png_name = os.path.splitext(base_name)[0] + ".png"
        fig_path = os.path.join(output_folder, png_name)

        # Plot and save the figure
        plot_time_differences(time_differences, fig_path)
