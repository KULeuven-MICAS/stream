"""Costs the solve reads from the hardware description rather than from code."""

from __future__ import annotations

import tempfile

from stream.cost_model.bandwidth import BandwidthModel
from stream.opt.allocation.constraint_optimization.hardware import AIE2Namespace
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.stage import LeafStage, MainStage


def _parse(path):
    ctx = StageContext.from_kwargs(accelerator=path, output_path=tempfile.mkdtemp())
    return MainStage([AcceleratorParserStage, LeafStage], ctx).run()[0].get("accelerator")


def test_the_strix_array_declares_its_off_chip_bandwidth_and_reconfiguration():
    accelerator = _parse("stream/inputs/aie/hardware/whole_array_strix.yaml")
    assert set(accelerator.bandwidth) == {accelerator.offchip_core_id}
    assert isinstance(accelerator.bandwidth[accelerator.offchip_core_id], BandwidthModel)
    assert accelerator.reconfiguration == {"cycles_per_column": 40000, "reset_cycles": 63000}


def test_hardware_without_either_declares_neither():
    accelerator = _parse("stream/inputs/examples/hardware/tpu_like_quad_core.yaml")
    assert accelerator.bandwidth == {}
    assert accelerator.reconfiguration == {}


def test_aie2_charges_the_declared_reconfiguration_once_a_dispatch_holds_several_designs():
    constraints = AIE2Namespace(reconfiguration={"cycles_per_column": 69000, "reset_cycles": 63000})
    assert constraints.dispatch_overhead_cycles([8]) == 0.0
    assert constraints.dispatch_overhead_cycles([2, 3]) == 5 * 69000 + 63000
    assert AIE2Namespace().dispatch_overhead_cycles([2, 3]) == 0.0
