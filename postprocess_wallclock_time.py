import argparse
import json
import os
import re
from collections import defaultdict

CLOCK_FREQUENCY_GHZ = 1.25
PEAK_MACS_PER_CYCLE_PER_CORE = 64


def parse_wall_clock_time_us(output_base, stats, M, K, N, nb_rows, nb_cols, is_amd=False) -> dict[str, int]:  # noqa: PLR0912, N803
    """
    Extracts the wall clock time (us) for each requested stat from run_trace.log.
    stats: list containing "avg", "min", and/or "max" to select which values to extract.
    Returns a dict mapping stat name to time in us.
    """
    if is_amd:
        log_path = os.path.join(output_base, f"{K}.log")
    else:
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
                    match = re.search(r"([0-9]*\.?[0-9]+)us", line)
                    if match:
                        # Parse as float and keep as float
                        results[stat] = float(match.group(1))
    if len(results) != len(stats):
        missing = [stat for stat in stats if results[stat] == -1]
        raise ValueError(f"Wall clock time(s) not found for: {', '.join(missing)} in {log_path}.")

    gmacs_per_sec = defaultdict(lambda: -1.0)
    for stat in stats:
        time_us = results[stat]
        if time_us > 0:
            gmacs_per_sec[stat] = (M * K * N) / time_us / 1_000.0

    peak_macs_per_cycle = PEAK_MACS_PER_CYCLE_PER_CORE * nb_rows * nb_cols
    peak_gmacs_per_sec = peak_macs_per_cycle * CLOCK_FREQUENCY_GHZ
    efficiency = defaultdict(lambda: -1.0)
    for stat in stats:
        if gmacs_per_sec[stat] > 0:
            efficiency[stat] = (gmacs_per_sec[stat] / peak_gmacs_per_sec) * 100.0

    for stat in stats:
        print(
            f"{stat.capitalize()} Wall clock time = {results[stat]} us, "
            f"{gmacs_per_sec[stat]:.2f} gMACs/s, "
            f"Efficiency = {efficiency[stat]:.1f} % (assuming 64 MACs/cycle at 1.25 GHz)."
        )

    os.makedirs(output_base, exist_ok=True)
    wall_clock_json_path = os.path.join(output_base, "wall_clock_time.json")
    with open(wall_clock_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Saved wall clock time stats to {wall_clock_json_path}.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Parse and save wall clock time from run_trace.log.")
    parser.add_argument("--M", type=int, required=True, help="Total layer size M.")
    parser.add_argument("--N", type=int, required=True, help="Total layer size N.")
    parser.add_argument("--K", type=int, required=True, help="Total layer size K.")
    parser.add_argument("--row", type=int, required=True, help="Number of rows used.")
    parser.add_argument("--col", type=int, required=True, help="Number of columns used.")
    parser.add_argument("--hwid", type=str, help="Hardware configuration identifier.")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    nb_rows, nb_cols = args.row, args.col
    hw_id = args.hwid
    output_base = f"outputs/{hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces/"
    # output_base_amd = f"outputs/plots/amd/{K}/"

    wall_clock_time_stats = ["avg", "min", "max"]
    parse_wall_clock_time_us(output_base, wall_clock_time_stats, M, K, N, nb_rows, nb_cols, is_amd=False)


if __name__ == "__main__":
    main()
