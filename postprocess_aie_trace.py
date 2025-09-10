#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any

# ---------- Tunables ----------
CLOCK_FREQUENCY_GHZ = 1.25  # cycles at 1.25 GHz
PEAK_MACS_PER_CYCLE_PER_CORE = 64
LOCK_STALL_FACTOR = 20.0  # boundary if LOCK_STALL duration ≥ this × median kernel duration
DUP_GAP_FRACTION = 0.25  # duplicate filter for EVENT_0/1 based on median kernel dur
OUTLIER_LO = 1.0 / 3.0  # duration outlier filter window [median/3, 3×median]
OUTLIER_HI = 3.0
# ------------------------------


@dataclass
class KernelPair:
    start: int
    end: int
    duration: int


@dataclass
class StallSpan:
    start: int
    end: int
    duration: int


def parse_perfetto_trace_events(file_path: str) -> dict[int, dict[str, Any]]:
    with open(file_path) as f:
        data = json.load(f)

    traces = defaultdict(
        lambda: {
            "name": None,
            "starts": [],
            "ends": [],
            "lock_stall_B": [],
            "lock_stall_E": [],
        }
    )

    for ev in data:
        pid = ev.get("pid")
        nm = ev.get("name")
        ph = ev.get("ph")
        ts = ev.get("ts")

        if nm == "process_name" and ph == "M":
            traces[pid]["name"] = ev.get("args", {}).get("name")

        if nm == "INSTR_EVENT_0" and ph == "B":
            traces[pid]["starts"].append(int(ts))
        elif nm == "INSTR_EVENT_1" and ph == "E":
            traces[pid]["ends"].append(int(ts))
        elif nm == "LOCK_STALL" and ph == "B":
            traces[pid]["lock_stall_B"].append(int(ts))
        elif nm == "LOCK_STALL" and ph == "E":
            traces[pid]["lock_stall_E"].append(int(ts))

    for _, t in traces.items():
        t["starts"].sort()
        t["ends"].sort()
        t["lock_stall_B"].sort()
        t["lock_stall_E"].sort()

        bs, es = t["lock_stall_B"], t["lock_stall_E"]
        i = j = 0
        spans: list[StallSpan] = []
        while i < len(bs) and j < len(es):
            if es[j] <= bs[i]:
                j += 1
                continue
            spans.append(StallSpan(bs[i], es[j], es[j] - bs[i]))
            i += 1
            j += 1
        t["lock_stalls"] = spans
        del t["lock_stall_B"]
        del t["lock_stall_E"]

    return traces


def _dedup_by_gap(ts_list: list[int], min_gap: int) -> list[int]:
    if not ts_list:
        return []
    out = [ts_list[0]]
    for t in ts_list[1:]:
        if t > out[-1] and (t - out[-1]) >= min_gap:
            out.append(t)
    return out


def _greedy_pairs(starts: list[int], ends: list[int]) -> list[KernelPair]:
    i = j = 0
    pairs: list[KernelPair] = []
    while i < len(starts) and j < len(ends):
        if ends[j] <= starts[i]:
            j += 1
            continue
        pairs.append(KernelPair(starts[i], ends[j], ends[j] - starts[i]))
        i += 1
        j += 1
    return pairs


def robust_pair_kernels(starts: list[int], ends: list[int]) -> list[KernelPair]:
    prelim = _greedy_pairs(starts, ends)
    med_dur = max(1, int(median([p.duration for p in prelim]))) if prelim else 1

    min_gap = max(1, int(DUP_GAP_FRACTION * med_dur))
    s_clean = _dedup_by_gap(sorted(starts), min_gap)
    e_clean = _dedup_by_gap(sorted(ends), min_gap)

    i = j = 0
    pairs: list[KernelPair] = []
    last_end = -1
    while i < len(s_clean) and j < len(e_clean):
        if s_clean[i] <= last_end:
            i += 1
            continue
        while j < len(e_clean) and e_clean[j] <= s_clean[i]:
            j += 1
        if j >= len(e_clean):
            break
        start_t = s_clean[i]
        end_t = e_clean[j]
        if end_t > start_t:
            pairs.append(KernelPair(start_t, end_t, end_t - start_t))
            last_end = end_t
        i += 1
        j += 1

    if not pairs:
        return []

    med = median([p.duration for p in pairs])
    lo = max(1, int(OUTLIER_LO * med))
    hi = max(lo + 1, int(OUTLIER_HI * med))
    filtered = [p for p in pairs if lo <= p.duration <= hi]
    return filtered if filtered else pairs


def derive_iteration_boundaries_from_lockstall(
    lock_stalls: list[StallSpan],
    median_kernel_dur: float | None,
    factor: float,
) -> list[int]:
    if not lock_stalls or not median_kernel_dur or median_kernel_dur <= 0:
        return []
    thr = factor * float(median_kernel_dur)
    return sorted({s.end for s in lock_stalls if s.duration >= thr})


def split_pairs_by_boundaries(pairs: list[KernelPair], boundary_times: list[int]) -> list[list[KernelPair]]:
    if not pairs:
        return []
    if not boundary_times:
        return [pairs]
    iters: list[list[KernelPair]] = []
    bidx = 0
    cur: list[KernelPair] = []
    for p in pairs:
        while bidx < len(boundary_times) and p.start > boundary_times[bidx]:
            iters.append(cur)
            cur = []
            bidx += 1
        cur.append(p)
    iters.append(cur)
    return iters


def summarize_iteration(iter_pairs: list[KernelPair]) -> dict[str, Any]:
    if not iter_pairs:
        return {
            "num_kernels": 0,
            "avg_duration_cycles": 0.0,
            "min_duration_cycles": 0,
            "max_duration_cycles": 0,
            "total_span_cycles": 0,
            "first_start": None,
            "last_end": None,
        }
    durs = [p.duration for p in iter_pairs]
    first_start = iter_pairs[0].start
    last_end = iter_pairs[-1].end
    return {
        "num_kernels": len(durs),
        "avg_duration_cycles": sum(durs) / len(durs),
        "min_duration_cycles": min(durs),
        "max_duration_cycles": max(durs),
        "total_span_cycles": last_end - first_start,
        "first_start": first_start,
        "last_end": last_end,
    }


def process_core(  # noqa: PLR0913
    pid: int,
    name: str,
    starts: list[int],
    ends: list[int],
    lock_stalls: list[StallSpan],
    warmup: int,
    iterations: int,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    N: int,  # noqa: N803
    rows: int,
    cols: int,
) -> dict[str, Any]:
    if "core_trace" not in name:
        return {}

    pairs = robust_pair_kernels(starts, ends)
    if not pairs:
        return {
            "pid": pid,
            "tile_name": name.split(" for ")[-1] if " for " in name else name,
            "detected_total_iterations": 0,
            "expected_total_iterations": warmup + iterations,
            "mismatch_note": "No valid kernel pairs detected after robust filtering.",
            "iterations": [],
            "performance_window": [],
        }

    med_kernel = median([p.duration for p in pairs])
    boundaries = derive_iteration_boundaries_from_lockstall(lock_stalls, med_kernel, LOCK_STALL_FACTOR)
    iters = split_pairs_by_boundaries(pairs, boundaries)

    # Per-core work per iteration:
    macs_per_iter_per_core = (float(M) * float(K) * float(N)) / max(1, rows * cols)

    iter_summaries = []
    for it in iters:
        summ = summarize_iteration(it)
        span = summ["total_span_cycles"]
        gmacs_per_s = (macs_per_iter_per_core * CLOCK_FREQUENCY_GHZ / span) if span and span > 0 else 0.0
        summ["gmacs_per_second"] = gmacs_per_s
        iter_summaries.append(summ)

    detected_total = len(iter_summaries)
    expected_total = warmup + iterations
    mismatch = None
    if detected_total != expected_total:
        if detected_total < expected_total:
            mismatch = (
                f"Detected {detected_total} iterations but expected {expected_total}. "
                f"Proceeding with available iterations."
            )
        else:
            mismatch = (
                f"Detected {detected_total} iterations but expected {expected_total}. "
                f"Truncating to first {expected_total} iterations for phase labeling."
            )

    perf_start_idx = min(warmup, detected_total)
    perf_end_idx = min(perf_start_idx + iterations, detected_total)
    perf_window = list(range(perf_start_idx, perf_end_idx))

    phase_labels = []
    for idx in range(detected_total):
        if perf_start_idx <= idx < perf_end_idx:
            phase_labels.append("perf")
        else:
            phase_labels.append("warmup" if idx < perf_start_idx else "extra")

    per_iter_with_phase = []
    for idx, summ in enumerate(iter_summaries):
        row = dict(summ)
        row["iteration_index"] = idx
        row["phase"] = phase_labels[idx]
        per_iter_with_phase.append(row)

    return {
        "pid": pid,
        "tile_name": name.split(" for ")[-1] if " for " in name else name,
        "median_kernel_duration_cycles": med_kernel,
        "lock_stall_boundary_count": len(boundaries),
        "detected_total_iterations": detected_total,
        "expected_total_iterations": expected_total,
        "mismatch_note": mismatch,
        "iterations": per_iter_with_phase,
        "performance_window": perf_window,
    }


def write_per_core_reports(out_dir: str, cores: list[dict[str, Any]]) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for core in cores:
        if not core:
            continue
        tile = core["tile_name"]
        path = os.path.join(out_dir, f"{tile}.json")
        with open(path, "w") as f:
            json.dump(core, f, indent=2)
        print(f"[write] {path}")
        paths.append(path)
    return paths


def write_details_markdown(
    out_dir: str,
    hwid: str,
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    N: int,  # noqa: N803
    cores: list[dict[str, Any]],
) -> str:
    rows: list[tuple[str, int, str, int, float, int, int, int, float]] = []
    for core in cores:
        if not core:
            continue
        tile = core["tile_name"]
        for it in core["iterations"]:
            rows.append(
                (
                    tile,
                    it["iteration_index"],
                    it.get("phase", ""),
                    it["num_kernels"],
                    it["avg_duration_cycles"],
                    it["min_duration_cycles"],
                    it["max_duration_cycles"],
                    it["total_span_cycles"],
                    it.get("gmacs_per_second", 0.0),
                )
            )

    if not rows:
        return ""

    details_path = os.path.join(out_dir, "details.md")
    with open(details_path, "w") as f:
        f.write(f"<details><summary><strong>[{hwid}] M={M} K={K} N={N}</strong></summary>\n\n")
        f.write("| Tile | Iter | Phase | Kernels | Avg cycles | Min | Max | Total span | GMAC/s |\n")
        f.write("|------|------|-------|---------|------------|-----|-----|------------|--------|\n")
        for tile, idx, phase, nk, avgc, minc, maxc, span, gmacs in rows:
            f.write(f"| {tile} | {idx} | {phase} | {nk} | {avgc:.3f} | {minc} | {maxc} | {span} | {gmacs:.3f} |\n")
        f.write("\n</details>\n")
    print(f"[write] {details_path}")
    return details_path


def compute_and_write_aggregate(
    out_dir: str,
    core_json_paths: list[str],
    M: int,  # noqa: N803
    K: int,  # noqa: N803
    N: int,  # noqa: N803
) -> str:
    """
    Build a system-level aggregate over *common* perf iterations across all cores:
      - per-iteration runtime_cycles = max across cores' total_span_cycles
      - per-iteration GMAC/s = (M*K*N * CLOCK_FREQUENCY_GHZ) / runtime_cycles
      - report averages and min/max summaries
    Omit any perf iteration index not present on all cores.
    """
    cores = []
    for p in core_json_paths:
        with open(p) as f:
            cores.append(json.load(f))

    # Collect the set of perf iteration indices for each core
    perf_sets = []
    for c in cores:
        perf_idxs = set(c.get("performance_window", []))
        # also ensure those indices exist in iterations
        valid = {i for i in perf_idxs if 0 <= i < len(c.get("iterations", []))}
        perf_sets.append(valid)

    if not perf_sets:
        agg_path = os.path.join(out_dir, "aggregate_perf.json")
        with open(agg_path, "w") as f:
            json.dump({"note": "No cores to aggregate."}, f, indent=2)
        print(f"[write] {agg_path}")
        return agg_path

    common = set.intersection(*perf_sets) if perf_sets else set()
    # Truncate to ordered list (ascending)
    common_ordered = sorted(common)
    if not common_ordered:
        agg_path = os.path.join(out_dir, "aggregate_perf.json")
        with open(agg_path, "w") as f:
            json.dump({"note": "No common perf iterations across all cores."}, f, indent=2)
        print(f"[write] {agg_path}")
        return agg_path

    per_iter = []
    total_macs_system = float(M) * float(K) * float(N)

    for i in common_ordered:
        # max runtime across cores
        runtimes = []
        for c in cores:
            it = c["iterations"][i]
            runtimes.append(int(it["total_span_cycles"]))
        runtime_cycles = max(runtimes) if runtimes else 0
        gmacs_per_s = (total_macs_system * CLOCK_FREQUENCY_GHZ / runtime_cycles) if runtime_cycles > 0 else 0.0
        per_iter.append(
            {
                "iteration_index": i,
                "runtime_cycles": runtime_cycles,
                "gmacs_per_second": gmacs_per_s,
            }
        )

    # Averages / min / max
    runtimes = [x["runtime_cycles"] for x in per_iter]
    gmacses = [x["gmacs_per_second"] for x in per_iter]

    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0
    avg_gmacs = sum(gmacses) / len(gmacses) if gmacses else 0.0

    # best = min runtime / max GMAC/s
    best_idx = max(range(len(gmacses)), key=lambda k: gmacses[k]) if gmacses else None
    worst_idx = min(range(len(gmacses)), key=lambda k: gmacses[k]) if gmacses else None

    summary = {
        "iterations_considered": common_ordered,
        "count": len(common_ordered),
        "average": {
            "runtime_cycles": avg_runtime,
            "gmacs_per_second": avg_gmacs,
        },
        "best": per_iter[best_idx] if best_idx is not None else None,
        "worst": per_iter[worst_idx] if worst_idx is not None else None,
        "per_iteration": per_iter,
        "note": "Only iterations present on all cores are considered.",
    }

    agg_path = os.path.join(out_dir, "aggregate_perf.json")
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {agg_path}")
    return agg_path


def main():
    ap = argparse.ArgumentParser(
        description="Postprocess AIE trace data with robust kernel pairing and LOCK_STALL boundaries."
    )
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--row", type=int, required=True, help="Number of rows used.")
    ap.add_argument("--col", type=int, required=True, help="Number of cols used.")
    ap.add_argument("--hwid", type=str, required=True)
    ap.add_argument("--warmup", type=int, required=True, help="Number of warmup iterations.")
    ap.add_argument("--iterations", type=int, required=True, help="Number of performance iterations after warmup.")

    args = ap.parse_args()

    # Ordering and auto paths
    M, K, N = args.M, args.K, args.N
    m, k, n = args.m, args.k, args.n
    nb_rows, nb_cols = args.row, args.col
    hwid = args.hwid

    # Input path pattern with "{nb_rows}_row_{nb_cols}_col.json"
    input_path = f"outputs/{hwid}-gemm_{M}_{K}_{N}-{nb_rows}_row_{nb_cols}_col/traces/trace_mm_{M}_{K}_{N}.json"
    output_dir = os.path.dirname(input_path)

    warmup = max(0, int(args.warmup))
    perf_iters = max(0, int(args.iterations))

    traces = parse_perfetto_trace_events(input_path)

    cores: list[dict[str, Any]] = []
    for pid, t in traces.items():
        name = t.get("name") or ""
        if "core_trace" not in name:
            continue
        core_summary = process_core(
            pid=pid,
            name=name,
            starts=t.get("starts", []),
            ends=t.get("ends", []),
            lock_stalls=t.get("lock_stalls", []),
            warmup=warmup,
            iterations=perf_iters,
            M=M,
            K=K,
            N=N,
            rows=nb_rows,
            cols=nb_cols,
        )
        if core_summary:
            cores.append(core_summary)

    os.makedirs(output_dir, exist_ok=True)
    core_paths = write_per_core_reports(output_dir, cores)
    details_path = write_details_markdown(output_dir, hwid, M, K, N, cores)

    # Run-level index
    run_index = {
        "hwid": hwid,
        "M": M,
        "K": K,
        "N": N,
        "m": m,
        "k": k,
        "n": n,
        "rows": nb_rows,
        "cols": nb_cols,
        "input_path": input_path,
        "output_dir": output_dir,
        "warmup": warmup,
        "iterations": perf_iters,
        "clock_GHz": CLOCK_FREQUENCY_GHZ,
        "notes": [c["mismatch_note"] for c in cores if c.get("mismatch_note")],
        "details_md": details_path or None,
    }
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(run_index, f, indent=2)
    print(f"[write] {index_path}")

    # System-level aggregate over common perf iterations
    _ = compute_and_write_aggregate(output_dir, core_paths, M, K, N)


if __name__ == "__main__":
    main()
