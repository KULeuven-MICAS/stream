import json

def parse_perfetto_trace(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    instr_event_0_times = []
    instr_event_1_times = []

    for event in data:
        if event.get('name') == 'INSTR_EVENT_0' and event.get('ph') == 'B':
            instr_event_0_times.append(event.get('ts'))
        elif event.get('name') == 'INSTR_EVENT_1' and event.get('ph') == 'E':
            instr_event_1_times.append(event.get('ts'))

    if len(instr_event_0_times) != len(instr_event_1_times):
        if len(instr_event_1_times) == 31 and len(instr_event_0_times) == 32:
            instr_event_0_times = instr_event_0_times[:-1]
        else:
            raise ValueError("Mismatched INSTR_EVENT_0 and INSTR_EVENT_1 events")

    time_differences = [end - start for start, end in zip(instr_event_0_times, instr_event_1_times)]
    total_difference = instr_event_1_times[-1] - instr_event_0_times[0]

    return time_differences, total_difference

def plot_time_differences(time_differences, fig_path):
    import matplotlib.pyplot as plt
    plt.grid()
    plt.plot(list(range(len(time_differences))), time_differences)
    plt.xlabel('Event number')
    plt.ylabel('Time difference (cycles)')
    # Set figure size in pixels
    plt.gcf().set_size_inches(4, 3)
    plt.savefig(fig_path, bbox_inches="tight")

if __name__ == "__main__":
    file_path = 'trace_conv2d.json'
    fig_path = 'time_differences.png'
    time_differences, total_difference = parse_perfetto_trace(file_path)
    for i, diff in enumerate(time_differences):
        print(f"Event {i}: {diff} cycles")
    print(f"Total difference: {total_difference} cycles")
    plot_time_differences(time_differences, fig_path)
    # Average difference
    avg_diff = sum(time_differences) / len(time_differences)
    print(f"Average difference = ", avg_diff)