"""Generic CO pipeline entry point for arbitrary workloads.

Supports two modes:
- Manual mapping: provide --mapping for optimize_allocation_co_with_mapping
- Auto mapping: omit --mapping for optimize_allocation_co_generic (auto-infers from workload+hardware)

Produces a YAML summary with total latency, per-group latencies, and group percentages.
"""

import argparse
import logging as _logging
import os
import re

import yaml

from stream.api import configure_logging, optimize_allocation_co_generic, optimize_allocation_co_with_mapping

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
configure_logging(level=_logging_level, fmt=_logging_format)

logger = _logging.getLogger(__name__)


def _derive_experiment_id(accelerator: str, workload_path: str) -> str:
    """Derive an experiment ID from accelerator and workload file names."""
    hw_name = accelerator.rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]
    wl_name = re.split(r"/|\.", workload_path)[-1]
    if wl_name == "onnx":
        wl_name = re.split(r"/|\.", workload_path)[-2]
    return f"{hw_name}-{wl_name}-constraint_optimization"


def _write_yaml_summary(ctx, output_path: str) -> None:
    """Write a YAML summary with total latency, per-group latencies, and percentages."""
    total_latency = ctx.get("total_latency")
    group_latencies = ctx.get("group_latencies", {})
    group_wall_times = ctx.get("group_wall_times", {})

    summary: dict = {"total_latency": total_latency}
    if group_wall_times:
        summary["total_wall_time_s"] = round(sum(group_wall_times.values()), 2)

    if group_latencies:
        groups = {}
        for group_id, latency in sorted(group_latencies.items()):
            pct = (latency / total_latency * 100) if total_latency > 0 else 0.0
            entry = {
                "latency": latency,
                "percentage": round(pct, 2),
            }
            if group_id in group_wall_times:
                entry["wall_time_s"] = round(group_wall_times[group_id], 2)
            groups[f"group_{group_id}"] = entry
        summary["groups"] = groups

    summary_path = os.path.join(output_path, "summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, sort_keys=False, default_flow_style=False)
    logger.info(f"Summary written to {summary_path}")
    logger.info(f"Total latency: {total_latency}")
    if group_latencies:
        for group_id, latency in sorted(group_latencies.items()):
            pct = (latency / total_latency * 100) if total_latency > 0 else 0.0
            wt = f", wall={group_wall_times[group_id]:.1f}s" if group_id in group_wall_times else ""
            logger.info(f"  Group {group_id}: {latency} ({pct:.1f}%{wt})")


def main():
    parser = argparse.ArgumentParser(description="Stream CO pipeline — generic entry point")
    parser.add_argument("--hardware", required=True, help="Path to hardware YAML")
    parser.add_argument("--workload", required=True, help="Path to workload ONNX")
    parser.add_argument("--mapping", default=None, help="Path to mapping YAML (omit for auto-generated mapping)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--experiment-id", default=None, help="Experiment ID (auto-derived if omitted)")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip if output exists")
    args = parser.parse_args()

    experiment_id = args.experiment_id or _derive_experiment_id(args.hardware, args.workload)
    output_path = os.path.join(args.output, experiment_id)
    os.makedirs(output_path, exist_ok=True)

    if args.mapping:
        logger.info(f"Running with manual mapping: {args.mapping}")
        ctx = optimize_allocation_co_with_mapping(
            hardware=args.hardware,
            workload=args.workload,
            mapping=args.mapping,
            experiment_id=experiment_id,
            output_path=args.output,
            skip_if_exists=args.skip_if_exists,
        )
    else:
        logger.info("Running with auto-generated mapping (generic pipeline)")
        ctx = optimize_allocation_co_generic(
            hardware=args.hardware,
            workload=args.workload,
            experiment_id=experiment_id,
            output_path=args.output,
            skip_if_exists=args.skip_if_exists,
        )

    _write_yaml_summary(ctx, output_path)


if __name__ == "__main__":
    main()
