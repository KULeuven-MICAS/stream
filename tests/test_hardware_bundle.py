"""De-aliased hardware bundles (C1) and the built-in hardware cost model / budget guard (C4)."""

import json
import math
import tempfile

import pytest
import yaml

from stream.api import evaluate_mapping
from stream.hardware.bundle import HardwareBundle
from stream.hardware.cost import (
    BudgetVerdict,
    HardwareBudget,
    HardwareBudgetExceededError,
    assert_within_budget,
    check_budget,
    evaluate_bundle_cost,
)
from stream.hardware.ports import OUTPUT
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.stage import LeafStage, MainStage

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
# An Ironwood variant whose VMEM is seen as `vmem` on a memory core AND as `operand_buffer` on each of that
# TensorCore's two compute cores: three views of one memory.
TPU_V7_VIEWS = "tests/fixtures/hardware/tpu_v7_views.yaml"
QUAD_CORE = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
AIE2_STRIX = "stream/inputs/aie/hardware/whole_array_strix.yaml"

# 128 MiB VMEM, the views fixture's per-TensorCore scratchpad size in bits.
_VIEWS_VMEM_BITS = 1024 * 1024 * 1024

# One 64 MiB TensorCore VMEM, from the ZigZag declaration in cores/tpu_v7_vmem.yaml.
_VMEM_BITS = 512 * 1024 * 1024
# Each MXU output lane's result buffer (cores/tpu_v7_mxu.yaml).
_RESULT_BUFFER_BITS = 16 * 1024
# TensorCore 0's VMEM core, then the MXUs and VPU sharing its VMEM.
_TC0 = (3, 0, 1, 2)


def _grow_vmem(bundle: HardwareBundle, cores: tuple[int, ...] = _TC0) -> None:
    """Double one TensorCore's VMEM, as seen by every core sharing it."""
    for core_id in cores:
        bundle.cores[core_id]["memories"]["vmem"]["size"] *= 2


def _parse(accelerator_path: str):
    ctx = StageContext.from_kwargs(accelerator=accelerator_path, output_path=tempfile.mkdtemp())
    ctxs = MainStage([AcceleratorParserStage, LeafStage], ctx).run()
    assert len(ctxs) == 1
    return ctxs[0].get("accelerator")


# ── C1: de-aliasing ─────────────────────────────────────────────────────────────────────────────


def test_bundle_de_aliases_shared_core_files():
    """All eight MXUs are authored from one file; the bundle must give each its own description."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    assert bundle.cores[0] is not bundle.cores[4]
    assert bundle.cores[0] == bundle.cores[4]  # identical content, independent objects

    bundle.cores[0]["memories"]["result_buffer"]["size"] *= 2
    assert bundle.cores[4]["memories"]["result_buffer"]["size"] == _RESULT_BUFFER_BITS


def test_materialized_bundle_carries_a_per_core_vmem_and_parses():
    """The C1 done-condition: a bundle in which TensorCore 0's VMEM differs from TensorCore 1's, on
    disk, parsed by Stream."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    _grow_vmem(bundle)

    with tempfile.TemporaryDirectory() as tmpdir:
        accelerator_path = bundle.materialize(tmpdir)
        written = yaml.safe_load(accelerator_path.read_text())
        # Every core id resolves to its own file — that is what makes the mutation targetable.
        assert len(set(written["cores"].values())) == len(written["cores"])

        accelerator = _parse(str(accelerator_path))

    capacities = {core.id: accelerator.memory_of(core, OUTPUT).capacity_bits for core in accelerator.core_list}
    assert capacities[3] == capacities[0] == 2 * _VMEM_BITS
    assert capacities[7] == capacities[4] == _VMEM_BITS


def test_bundle_to_accelerator_needs_no_files():
    """Inline core descriptions reach the same validators as file references."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    accelerator = bundle.to_accelerator()
    assert len(accelerator.core_list) == len(bundle.cores)


def test_inline_core_rejected_when_invalid():
    """A mutated bundle that is no longer a legal accelerator must not silently reach a solve."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    del bundle.cores[0]["memories"]["result_buffer"]["size"]
    with pytest.raises(ValueError, match="not a valid accelerator"):
        bundle.validated_data()


def test_a_view_naming_no_memory_of_its_core_is_rejected():
    """A mistyped view would split one memory and quietly inflate the modelled area."""
    data = HardwareBundle.from_yaml(TPU_V7).to_data()
    data["memories"]["vmem_0"]["views"] = ["0.not_a_memory", "1.vmem", "2.vmem", "3.vmem"]
    validator = AcceleratorValidator(data, TPU_V7)
    assert not validator.validate()
    assert any("not_a_memory" in e for e in validator.errors)


@pytest.mark.timeout(300)
def test_asymmetric_bundle_runs_end_to_end():
    """Stream runs a materialized bundle whose cores were authored from one shared file: here core 0
    alone gets a bigger top-level memory, inexpressible without de-aliasing."""
    bundle = HardwareBundle.from_yaml(QUAD_CORE)
    top_memory = list(bundle.cores[0]["memories"])[-1]
    bundle.cores[0]["memories"][top_memory]["size"] *= 2

    workload_path = make_2_conv_workload(
        TwoConvWorkloadConfig(
            batch_size=1,
            in_channels=8,
            height=32,
            width=32,
            out_channels_1=16,
            out_channels_2=32,
            kernel_size=3,
            in_dtype="bf16",
            weight_dtype="bf16",
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        accelerator_path = bundle.materialize(tmpdir)
        ctx = evaluate_mapping(str(accelerator_path), workload_path, tmpdir).context
    assert ctx.get("total_latency") > 0


# ── C4: the built-in cost model ───────────────────────────────────────────────────────────────────


def test_the_cost_report_prices_only_what_a_solve_accepts():
    """A top level that is a view of no memory fails the solve; the report must not quietly price it as a private
    level beside the shared memory it was meant to be a view of."""
    from pathlib import Path

    path = Path("stream/inputs/examples/hardware/meta_prototype_dual_core_simd_offchip.yaml")
    data = yaml.safe_load(path.read_text())
    del data["memories"]["meta_prototype_0_weights"]  # core 0's sram_1MB_W is then a view of nothing
    for connection in data["connections"]:
        connection["ends"].pop("meta_prototype_0_weights", None)
    data["connections"] = [c for c in data["connections"] if len(c["ends"]) >= 2]
    bundle = HardwareBundle.from_data(data, str(path.parent))
    with pytest.raises(ValueError, match="core 0's top-level memory `sram_1MB_W` is a view of no memory"):
        evaluate_bundle_cost(bundle)


def test_a_core_viewing_the_off_chip_memory_is_its_controller_off_the_die():
    """A DRAM controller core computes nothing and its own levels sit off the die with the memory it views."""
    from pathlib import Path

    from stream.hardware.cost import _evaluate_builtin

    data = yaml.safe_load(Path(TPU_V7_VIEWS).read_text())
    hbm = next(name for name, memory in data["memories"].items() if memory["offchip"])
    data["memories"][hbm]["views"] = ["8.vmem"]
    del data["memories"][hbm]["capacity_bits"]
    data["memories"][hbm]["ports"] = {
        "rw_port_1": {"from": "r_port_1", "cores": [8]},
        "w": {"from": "w_port_1", "cores": [8]},
    }
    data["memories"]["vmem_0"]["views"] = ["0.operand_buffer", "1.operand_buffer"]
    data["memories"]["vmem_0"]["ports"] = {name: {"from": name, "cores": [0, 1]} for name in ("r_port_1", "w_port_1")}
    bundle = HardwareBundle.from_data(data, str(Path(TPU_V7_VIEWS).parent))
    report = _evaluate_builtin(bundle, None)
    controller = next(core for core in report.cores if core.core_id == 8)
    assert controller.compute is None and all(m.style == "off_die" for m in controller.memories)
    assert next(core for core in report.cores if core.core_id == 0).compute is not None


def test_views_disagreeing_on_their_instances_are_priced_with_a_warning():
    """Two cores whose arrays leave a shared level a different number of instances describe one memory badly; the
    report prices it at the most instances and says so rather than failing the whole cost report."""
    from stream.hardware.cost import _DENSITIES, DEFAULT_TECH_NODE, _memory_cost

    def core(served: list[str]) -> dict:
        level = {"size": 1024, "served_dimensions": served, "ports": [{"name": "r", "bandwidth_max": 64}]}
        return {"operational_array": {"dimensions": ["D1"], "sizes": [4]}, "memories": {"lvl": level}}

    warnings: list[str] = []
    cost = _memory_cost(
        "buf", [(0, core(["D1"]), "lvl"), (1, core([]), "lvl")], True, _DENSITIES[DEFAULT_TECH_NODE], warnings
    )
    assert cost.instances == 4 and cost.bits_per_instance == 1024
    assert warnings == [
        "the views of memory buf disagree on its (instances, bits) [(1, 1024), (4, 1024)]; priced at the most instances"
    ]


def test_simple_model_reports_positive_cost_and_prices_each_memory_once():
    """The built-in model prices a bundle with positive area and energy, and four memories seen through
    twelve views exactly once each."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7_VIEWS))
    assert report.total_area_mm2 > 0
    assert report.peak_access_energy_pj_per_cycle > 0

    vmems = [m for m in report.memories if m.bits_per_instance == _VIEWS_VMEM_BITS]
    assert [(m.name, m.core_ids) for m in vmems] == [(f"vmem_{k}", (8 + k, 2 * k, 2 * k + 1)) for k in range(4)]
    # Each VMEM declares two ports its three views all reach: two physical ports, not six.
    assert [m.nb_ports for m in vmems] == [2] * 4
    assert not any(m.bits_per_instance == _VIEWS_VMEM_BITS for c in report.cores for m in c.memories)

    # The built-in model advertises none of the detailed model's caveat fields.
    assert report.accuracy_claim is None
    assert report.authored_disagreements == []


def test_growing_a_memory_through_all_its_views_grows_its_area_once():
    bundle = HardwareBundle.from_yaml(TPU_V7_VIEWS)
    baseline = evaluate_bundle_cost(bundle)

    grown = bundle.copy()
    grown.cores[8]["memories"]["vmem"]["size"] *= 2
    for core_id in (0, 1):
        grown.cores[core_id]["memories"]["operand_buffer"]["size"] *= 2
    assert evaluate_bundle_cost(grown).on_die_memory_bits == baseline.on_die_memory_bits + _VIEWS_VMEM_BITS


def test_the_views_of_a_memory_hold_its_capacity_once():
    """The MXU, VPU and VMEM views of one scratchpad are one memory, of one scratchpad's capacity."""
    accelerator = HardwareBundle.from_yaml(TPU_V7_VIEWS).to_accelerator()
    vmem = accelerator.memories["vmem_0"]
    assert [(core.id, instance) for core, instance in vmem.views] == [
        (8, "vmem"),
        (0, "operand_buffer"),
        (1, "operand_buffer"),
    ]
    assert vmem.capacity_bits == _VIEWS_VMEM_BITS
    assert all(accelerator.memory_of(accelerator.get_core(i), OUTPUT) is vmem for i in (0, 1, 8))


def test_views_of_one_memory_that_disagree_on_its_size_are_rejected():
    bundle = HardwareBundle.from_yaml(TPU_V7_VIEWS)
    bundle.cores[0]["memories"]["operand_buffer"]["size"] *= 2
    with pytest.raises(ValueError, match="declare different sizes"):
        bundle.validated_data()
    with pytest.raises(ValueError, match="declare different sizes"):
        evaluate_bundle_cost(bundle)


def test_a_shared_vmem_is_priced_once_per_tensorcore():
    """Each TensorCore's 64 MiB VMEM is the top memory of its VMEM core, both MXUs and its VPU
    (one memory): four views of one macro, priced once."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))
    vmems = [m for m in report.memories if m.total_bits == _VMEM_BITS]
    assert [m.core_ids for m in vmems] == [tuple(range(4 * k, 4 * k + 4)) for k in range(4)]

    grown = HardwareBundle.from_yaml(TPU_V7)
    _grow_vmem(grown)
    growth = evaluate_bundle_cost(grown).on_die_memory_bits - report.on_die_memory_bits
    assert growth == _VMEM_BITS


def test_offchip_memory_carries_no_die_area():
    """An HBM stack is external memory; billing its capacity as SRAM would swamp everything else."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))
    (hbm,) = [m for m in report.memories if m.style == "off_die"]
    assert hbm.area_mm2 == 0.0
    assert report.off_die_access_energy_pj_per_cycle > 0


def test_aie_bundle_is_priceable_and_reports_what_it_cannot_model():
    """The AIE2 array must be priceable too -- and its unmodelled compute must read as unknown,
    not as zero."""
    bundle = HardwareBundle.from_yaml(AIE2_STRIX)
    report = evaluate_bundle_cost(bundle)

    assert report.memory_area_mm2 > 0
    assert not report.technology_declared
    assert report.technology_node == "n16"
    compute_tiles = [c for c in report.cores if c.compute]
    assert compute_tiles and all(not c.compute.modelled for c in compute_tiles)


def test_cost_report_is_plain_json():
    report = json.loads(json.dumps(evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7)).to_dict()))
    assert report["total_area_mm2"] > 0
    assert report["cores"][0]["memories"][0]["name"]
    assert report["memories"][0]["core_ids"]


# ── C4: the budget guard ────────────────────────────────────────────────────────────────────────


def test_baseline_is_its_own_budget():
    bundle = HardwareBundle.from_yaml(TPU_V7)
    budget = HardwareBudget.from_bundle(bundle)
    verdict = check_budget(bundle, budget)
    assert isinstance(verdict, BudgetVerdict)
    assert verdict.ok, verdict.violations


def test_over_budget_variant_is_rejected():
    bundle = HardwareBundle.from_yaml(TPU_V7)
    budget = HardwareBudget.from_bundle(bundle)

    variant = bundle.copy()
    _grow_vmem(variant)
    verdict = check_budget(variant, budget)
    assert not verdict.ok
    assert "area" in verdict.violations[0]
    with pytest.raises(HardwareBudgetExceededError):
        assert_within_budget(variant, budget)


def test_budget_headroom_admits_a_bounded_increase():
    bundle = HardwareBundle.from_yaml(TPU_V7)
    variant = bundle.copy()
    _grow_vmem(variant)
    growth = evaluate_bundle_cost(variant).total_area_mm2 / evaluate_bundle_cost(bundle).total_area_mm2

    assert not check_budget(variant, HardwareBudget.from_bundle(bundle, headroom=0.05)).ok
    generous = HardwareBudget.from_bundle(bundle, headroom=math.ceil(growth * 100) / 100)
    assert check_budget(variant, generous).ok
    assert assert_within_budget(variant, generous).total_area_mm2 > 0


# ── C4: the model's own caveats ──────────────────────────────────────────────────────────────────


def test_a_bundle_with_no_declared_array_says_its_compute_was_not_modelled():
    """An aie2 tile declares no operational_array, so its compute area is UNKNOWN, not an estimate."""
    aie2 = evaluate_bundle_cost(HardwareBundle.from_yaml(AIE2_STRIX))
    assert aie2.compute_modelled is False
    tpu = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))
    assert tpu.compute_modelled is True


def test_an_undeclared_technology_node_is_reported_as_the_substitution_it_is():
    bundle = HardwareBundle.from_yaml(TPU_V7)
    assert evaluate_bundle_cost(bundle).technology_declared

    undeclared = bundle.copy()
    undeclared.accelerator.pop("technology_node")
    report = evaluate_bundle_cost(undeclared)
    assert not report.technology_declared
    assert any("declares no `technology_node`" in w for w in report.warnings)
