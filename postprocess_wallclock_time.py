#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict

CLOCK_FREQUENCY_GHZ = 1.25
PEAK_MACS_PER_CYCLE_PER_CORE = 64


def parse_wall_clock_time_us(
    output_base: str,
    stats: list[str],
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    N: int,  # noqa: N803
    nb_rows: int,
    nb_cols: int,
    is_amd: bool = False,
) -> dict:
    """
    Extracts wall clock time (us) for each requested stat from run_trace.log (or {K}.log if AMD).
    Returns a dict with one entry per stat, each containing time_us, gmacs_per_second, efficiency_percent.
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

    # 1) Parse times (us)
    times_us = defaultdict(lambda: -1.0)
    with open(log_path) as log_file:
        for line in log_file:
            for stat in stats:
                search_str = stat_map.get(stat.lower())
                if search_str and search_str in line:
                    match = re.search(r"([0-9]*\.?[0-9]+)\s*us", line)
                    if match:
                        times_us[stat] = float(match.group(1))

    missing = [s for s in stats if times_us[s] < 0]
    if missing:
        raise ValueError(f"Wall clock time(s) not found for: {', '.join(missing)} in {log_path}.")

    # 2) Derived metrics
    total_macs = float(M) * float(K) * float(N)  # system MACs per iteration/layer
    gmacs_per_sec = {}
    for stat in stats:
        t_us = times_us[stat]
        gmacs_per_sec[stat] = (total_macs / (t_us * 1_000.0)) if t_us > 0 else 0.0

    peak_macs_per_cycle_system = PEAK_MACS_PER_CYCLE_PER_CORE * nb_rows * nb_cols
    peak_gmacs_per_sec = peak_macs_per_cycle_system * CLOCK_FREQUENCY_GHZ
    efficiency = {}
    for stat in stats:
        eff = (gmacs_per_sec[stat] / peak_gmacs_per_sec) * 100.0 if peak_gmacs_per_sec > 0 else 0.0
        efficiency[stat] = eff

    # 3) Build single JSON payload
    payload = {
        "meta": {
            "M": M,
            "K": K,
            "N": N,
            "rows": nb_rows,
            "cols": nb_cols,
            "clock_GHz": CLOCK_FREQUENCY_GHZ,
            "peak_macs_per_cycle_per_core": PEAK_MACS_PER_CYCLE_PER_CORE,
            "peak_system_gmacs_per_second": peak_gmacs_per_sec,
            "log_path": log_path,
        },
        "stats": {
            stat: {
                "time_us": times_us[stat],
                "gmacs_per_second": gmacs_per_sec[stat],
                "efficiency_percent": efficiency[stat],
            }
            for stat in stats
        },
    }

    # 4) Persist
    os.makedirs(output_base, exist_ok=True)
    out_path = os.path.join(output_base, "wall_clock_summary.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[write] {out_path}")

    # 5) Optional: keep your console prints
    for stat in stats:
        print(
            f"{stat.capitalize()} Wall clock time = {times_us[stat]} us, "
            f"{gmacs_per_sec[stat]:.2f} GMAC/s, "
            f"Efficiency = {efficiency[stat]:.1f} % "
            f"(assuming {PEAK_MACS_PER_CYCLE_PER_CORE} MACs/cycle/core "
            f"at {CLOCK_FREQUENCY_GHZ} GHz across {nb_rows * nb_cols} cores)."
        )

    return payload


def main():
    parser = argparse.ArgumentParser(description="Parse and save wall clock time and throughput/efficiency.")
    # Use M K N ordering (consistent with other scripts)
    parser.add_argument("--M", type=int, required=True, help="Total layer size M.")
    parser.add_argument("--K", type=int, required=True, help="Total layer size K.")
    parser.add_argument("--N", type=int, required=True, help="Total layer size N.")
    parser.add_argument("--row", type=int, required=True, help="Number of rows used.")
    parser.add_argument("--col", type=int, required=True, help="Number of columns used.")
    parser.add_argument("--hwid", type=str, required=True, help="Hardware configuration identifier.")
    parser.add_argument("--is-amd", action="store_true", help="Use AMD log naming ({K}.log).")
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    nb_rows, nb_cols = args.row, args.col
    hw_id = args.hwid

    # Same path pattern you’ve been using
    output_base = f"outputs/{hw_id}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces/"

    wall_clock_time_stats = ["avg", "min", "max"]
    parse_wall_clock_time_us(
        output_base=output_base,
        stats=wall_clock_time_stats,
        M=M,
        K=K,
        N=N,
        nb_rows=nb_rows,
        nb_cols=nb_cols,
        is_amd=args.is_amd,
    )


if __name__ == "__main__":
    main()
