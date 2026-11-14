"""Costs the solve reads from the hardware description rather than from code."""

from __future__ import annotations

import json
import tempfile
from itertools import pairwise
from pathlib import Path
from statistics import mean

from stream.cost_model.bandwidth import BandwidthModel
from stream.opt.allocation.constraint_optimization.hardware import AIE2Namespace
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.stage import LeafStage, MainStage

HARDWARE = Path("stream/inputs/aie/hardware")
MHZ = 1800  # the array's clock, cycles per microsecond


def _parse(path):
    ctx = StageContext.from_kwargs(accelerator=path, output_path=tempfile.mkdtemp())
    return MainStage([AcceleratorParserStage, LeafStage], ctx).run()[0].get("accelerator")


def test_the_strix_array_declares_its_off_chip_bandwidth_and_reconfiguration():
    accelerator = _parse("stream/inputs/aie/hardware/whole_array_strix.yaml")
    assert set(accelerator.bandwidth) == {accelerator.offchip_memory}
    assert isinstance(accelerator.bandwidth[accelerator.offchip_memory], BandwidthModel)
    assert set(accelerator.reconfiguration) == {"cycles_per_core", "cycles_per_group"}


def test_hardware_without_either_declares_neither():
    accelerator = _parse("stream/inputs/examples/hardware/tpu_like_quad_core.yaml")
    assert accelerator.bandwidth == {}
    assert accelerator.reconfiguration == {}


def test_aie2_configures_each_design_per_core_once_a_dispatch_holds_several():
    constraints = AIE2Namespace(reconfiguration={"cycles_per_core": 20000, "cycles_per_group": 50000})
    assert constraints.dispatch_overhead_cycles([32]) == 50000
    assert constraints.dispatch_overhead_cycles([32, 8]) == 2 * 50000 + 40 * 20000
    assert AIE2Namespace().dispatch_overhead_cycles([32, 8]) == 0.0


def test_the_strix_reconfiguration_is_fit_to_the_measured_group_runs():
    """At 512/512/2048, where every group's price is a few hundred microseconds: a group run alone takes a fixed time
    more than its price (the mean over the groups), and each design added to a dispatch adds its run alone and its
    configuration, proportional to its cores (least squares through the origin)."""
    measured = json.loads((HARDWARE / "measurements" / "strix_group_runs.json").read_text())
    point = (512, 512, 2048)
    groups = {r["group"]: r for r in measured["groups"] if (r["seq"], r["emb"], r["hidden"]) == point}
    fixed = mean(mean(r["alone_us"]) - r["priced_cycles"] / MHZ for r in groups.values())
    prefixes = sorted(
        (r["groups"], mean(r["npu_us"])) for r in measured["prefixes"] if (r["seq"], r["emb"], r["hidden"]) == point
    )
    runs = [0.0] + [us for _, us in prefixes]
    order = [groups[name] for name in ("gate", "gate", "silu", "mul", "down")]  # the up projection is the gate's design
    steps = zip(order, pairwise(runs), strict=True)
    added = [(r["cores"], after - before - mean(r["alone_us"])) for r, (before, after) in steps]
    per_core = sum(c * us for c, us in added) / sum(c * c for c, _ in added)
    declared = _parse(str(HARDWARE / "whole_array_strix.yaml")).reconfiguration
    assert declared == {"cycles_per_core": round(per_core * MHZ), "cycles_per_group": round(fixed * MHZ)}


def test_the_retired_reconfiguration_keys_are_rejected():
    import pytest
    from zigzag.utils import open_yaml

    from stream.parser.accelerator_validator import AcceleratorValidator

    with pytest.raises(ValueError, match="cycles_per_core"):
        AIE2Namespace(reconfiguration={"cycles_per_column": 40000, "reset_cycles": 63000})
    path = str(HARDWARE / "whole_array_strix.yaml")
    data = open_yaml(path) | {"reconfiguration": {"cycles_per_column": 40000, "reset_cycles": 63000}}
    assert not AcceleratorValidator(data, path).validate()


def test_an_estimate_charges_its_design_by_the_cores_it_occupies(tmp_path):
    """One GEMM on one core: one design, so its dispatch is its first fetch and last write, and no configuration."""
    from stream.api import SolveOptions, evaluate_mapping
    from stream.inputs.aie.mapping.make_gemm_mapping import make_gemm_mapping
    from stream.inputs.aie.workload.make_onnx_gemm import make_gemm_workload

    workload = make_gemm_workload(64, 64, 64, "bf16", "bf16", str(tmp_path))
    mapping = make_gemm_mapping(64, 64, 64, 64, 64, 64, nb_rows_to_use=1, nb_cols_to_use=1, output_dir=str(tmp_path))
    path = str(HARDWARE / "whole_array_strix.yaml")
    estimate = evaluate_mapping(path, workload, str(tmp_path), mapping, SolveOptions(nb_cols_to_use=1, artifacts=False))
    assert estimate.context.get("group_cores") == {0: 1}
    assert estimate.dispatch_cycles == _parse(path).reconfiguration["cycles_per_group"]
