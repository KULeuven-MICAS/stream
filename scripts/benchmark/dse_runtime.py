"""Where a solve spends its time: run a fixed suite of cases under the ``timing`` observer.

Writes one JSON with, per case, its wall time, its estimate (so a restructuring can show it changes no result)
and every span's calls, inclusive and exclusive seconds. Prints the share of the suite's time per span name.

    python scripts/benchmark/dse_runtime.py out.json [--cases 'swiglu*'] [--repeat 3] [--no-timing] [--no-artifacts]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from functools import cache
from typing import Any

from stream.api import MappingEstimate, SolveOptions, evaluate_mapping, generate_code
from stream.inputs.aie.mapping.make_swiglu_mapping import make_swiglu_mapping
from stream.inputs.aie.workload.make_onnx_swiglu import make_swiglu_workload
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.inputs.testing.workload.make_swiglu import make_small_swiglu_workload

HARDWARE = {
    "eyeriss_like_single_core": "eyeriss_like_single_core.yaml",
    "eyeriss_like_dual_core": "eyeriss_like_dual_core.yaml",
    "eyeriss_like_quad_core": "eyeriss_like_quad_core.yaml",
    "tpu_like_quad_core": "tpu_like_quad_core.yaml",
    "simba_small": "simba_small.yaml",
    "simba": "simba.yaml",
    "fusemax": "fusemax.yaml",
    "meta_prototype": "meta_prototype_dual_core_simd_offchip.yaml",
}
TWO_CONV = TwoConvWorkloadConfig(
    batch_size=1,
    height=32,
    width=32,
    in_channels=8,
    out_channels_1=16,
    out_channels_2=32,
    kernel_size=3,
    in_dtype="bf16",
    weight_dtype="bf16",
)
SWIGLU_TILING = [
    {"dim": "Gemm_Left.D1", "tile": 128},
    {"dim": "Gemm_Down.D2", "tile": 128},
    {"dim": "Gemm_Left.D2", "tile": 32},
    {"dim": "Gemm_Left.D0", "tile": 16},
]
AIE = "stream/inputs/aie/hardware/whole_array_strix.yaml"

Case = Callable[[str, dict[str, Any]], MappingEstimate]
SUITE: dict[str, Any] = {"backend": "ortools_gscip", "artifacts": True}


@cache
def _workload(name: str) -> str:
    """Each workload is built once per suite, as a sweep over mappings or hardware builds it once."""
    if name == "two_conv":
        return make_2_conv_workload(TWO_CONV)
    return make_small_swiglu_workload(seq_len=256, embedding_dim=2048, hidden_dim=8192)


def _two_conv(hardware: str) -> Case:
    def run(path: str, instrumentation: dict[str, Any]) -> MappingEstimate:
        options = SolveOptions(backend=SUITE["backend"], artifacts=SUITE["artifacts"], instrumentation=instrumentation)
        return evaluate_mapping(hardware, _workload("two_conv"), path, options=options)

    return run


def _swiglu(hardware: str) -> Case:
    def run(path: str, instrumentation: dict[str, Any]) -> MappingEstimate:
        options = SolveOptions(
            backend=SUITE["backend"],
            artifacts=SUITE["artifacts"],
            stage_options={"intra_core_tiling": SWIGLU_TILING},
            instrumentation=instrumentation,
        )
        return evaluate_mapping(hardware, _workload("swiglu"), path, options=options)

    return run


def _aie_swiglu(path: str, instrumentation: dict[str, Any]) -> MappingEstimate:
    workload = make_swiglu_workload(256, 512, 2048, "bf16", "bf16", last_gemm_down=True)
    mapping = make_swiglu_mapping(256, 512, 2048, True, 32, 32, 64)
    options = SolveOptions(
        backend=SUITE["backend"],
        artifacts=SUITE["artifacts"],
        nb_cols_to_use=8,
        stage_options={"npu": "npu2"},
        instrumentation=instrumentation,
    )
    return generate_code(AIE, workload, path, mapping, options)


def cases() -> dict[str, Case]:
    found: dict[str, Case] = {}
    for name, file in HARDWARE.items():
        hardware = f"stream/inputs/examples/hardware/{file}"
        found[f"two_conv[{name}]"] = _two_conv(hardware)
        found[f"swiglu[{name}]"] = _swiglu(hardware)
    found["aie_swiglu_codegen"] = _aie_swiglu
    return found


def run_case(case: Case, timing: bool) -> dict[str, Any]:
    _workload("two_conv"), _workload("swiglu")
    with tempfile.TemporaryDirectory() as tmp:
        report = f"{tmp}/timing.json"
        start = time.perf_counter()
        estimate = case(tmp, {"timing": {"path": report}} if timing else {})
        wall = time.perf_counter() - start
        spans = json.load(open(report))["spans"] if timing else []
    if timing:
        traced = sum(row["inclusive_s"] for row in spans if len(row["path"]) == 1)
        spans.append({"path": ["untraced"], "calls": 1, "inclusive_s": wall - traced, "exclusive_s": wall - traced})
    return {"wall_s": wall, "cycles": estimate.cycles, "group_cycles": list(estimate.group_cycles), "spans": spans}


def summarize(results: dict[str, list[dict[str, Any]]]) -> list[tuple[str, float, float]]:
    """Per span name: median-of-repeats exclusive seconds summed over cases, and its share of the suite."""
    by_name: dict[str, float] = defaultdict(float)
    total = 0.0
    for runs in results.values():
        total += statistics.median(run["wall_s"] for run in runs)
        per_run = []
        for run in runs:
            seconds: dict[str, float] = defaultdict(float)
            for row in run["spans"]:
                seconds[row["path"][-1]] += row["exclusive_s"]
            per_run.append(seconds)
        for name in {name for seconds in per_run for name in seconds}:
            by_name[name] += statistics.median(seconds.get(name, 0.0) for seconds in per_run)
    rows = sorted(by_name.items(), key=lambda item: -item[1])
    return [(name, seconds, seconds / total if total else 0.0) for name, seconds in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--cases", default="*")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--backend", default="ortools_gscip")
    args = parser.parse_args()
    SUITE.update(backend=args.backend, artifacts=not args.no_artifacts)
    selected = {name: case for name, case in cases().items() if fnmatch.fnmatch(name, args.cases)}
    results: dict[str, list[dict[str, Any]]] = {}
    for name, case in selected.items():
        results[name] = [run_case(case, not args.no_timing) for _ in range(args.repeat)]
        print(f"{name}: {statistics.median(r['wall_s'] for r in results[name]):.2f} s", flush=True)
    json.dump(results, open(args.output, "w"), indent=1)
    total = sum(statistics.median(r["wall_s"] for r in runs) for runs in results.values())
    print(f"\nsuite: {total:.1f} s over {len(results)} cases")
    for name, seconds, share in summarize(results)[:30]:
        print(f"{name:40s} {seconds:8.2f} s {share:6.1%}")


if __name__ == "__main__":
    main()
