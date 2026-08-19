"""De-aliased hardware bundles (C1) and the hardware cost model / budget guard (C4): what does a
hardware mutation cost, and can the answer be produced before a solve runs?"""

import math
import tempfile

import pytest
import yaml

from stream.api import hardware_cost_report, optimize_allocation_co_generic
from stream.hardware.bundle import HardwareBundle
from stream.hardware.cost import (
    ACCURACY_CLAIM,
    TECH_NODES,
    BudgetVerdict,
    ComputeCost,
    HardwareBudget,
    HardwareBudgetExceededError,
    HardwareCostReport,
    MemoryCost,
    assert_within_budget,
    check_budget,
    evaluate_bundle_cost,
)
from stream.inputs.testing.workload.make_2_conv import TwoConvWorkloadConfig, make_2_conv_workload
from stream.parser.accelerator_validator import AcceleratorValidator
from stream.stages.context import StageContext
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.stage import LeafStage, MainStage

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"
# The aliased Ironwood variant: VMEM is seen as `vmem` on a memory core AND as `operand_buffer` on
# each of that TensorCore's two compute cores. `memory_aliases` names the three as one physical macro.
TPU_V7_ALIASED = "tests/fixtures/hardware/tpu_v7_aliased.yaml"
QUAD_CORE = "stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
AIE2_STRIX = "stream/inputs/aie/hardware/whole_array_strix.yaml"

# 128 MiB VMEM, the aliased fixture's per-TensorCore scratchpad size in bits.
_ALIASED_VMEM_BITS = 1024 * 1024 * 1024

# One 64 MiB TensorCore VMEM, from the ZigZag declaration in cores/tpu_v7_vmem.yaml.
_VMEM_BITS = 512 * 1024 * 1024
# Array-local operand staging on each MXU/VPU compute core (cores/tpu_v7_mxu.yaml).
_OPERAND_BUFFER_BITS = 16 * 1024 * 1024


def _parse(accelerator_path: str):
    ctx = StageContext.from_kwargs(accelerator=accelerator_path, output_path=tempfile.mkdtemp())
    ctxs = MainStage([AcceleratorParserStage, LeafStage], ctx).run()
    assert len(ctxs) == 1
    return ctxs[0].get("accelerator")


def _vmem(report: HardwareCostReport) -> MemoryCost:
    """Core 9's `vmem` MemoryCost -- the TensorCore scratchpad most cost tests probe."""
    return next(m for c in report.cores if c.core_id == 9 for m in c.memories if m.memory_name == "vmem")


def _mxu(report: HardwareCostReport) -> ComputeCost:
    """Core 0's MXU ComputeCost."""
    return next(c.compute for c in report.cores if c.core_id == 0 and c.compute)


# ── C1: de-aliasing ─────────────────────────────────────────────────────────────────────────────


def test_bundle_de_aliases_shared_core_files():
    """Cores 0/2/4/6 are authored from one file; the bundle must give each its own description."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    assert bundle.cores[0] is not bundle.cores[2]
    assert bundle.cores[0] == bundle.cores[2]  # identical content, independent objects

    bundle.cores[0]["memories"]["operand_buffer"]["size"] *= 2
    assert bundle.cores[2]["memories"]["operand_buffer"]["size"] == _OPERAND_BUFFER_BITS


def test_materialized_bundle_carries_a_per_core_vmem_and_parses():
    """The C1 done-condition: a bundle in which core 9's VMEM differs from core 19's, on disk,
    parsed by Stream."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    bundle.cores[9]["memories"]["vmem"]["size"] = 2 * _VMEM_BITS

    with tempfile.TemporaryDirectory() as tmpdir:
        accelerator_path = bundle.materialize(tmpdir)
        written = yaml.safe_load(accelerator_path.read_text())
        # Every core id resolves to its own file — that is what makes the mutation targetable.
        assert len(set(written["cores"].values())) == len(written["cores"])

        accelerator = _parse(str(accelerator_path))

    capacities = {core.id: core.get_memory_capacity() for core in accelerator.core_list}
    assert capacities[9] == 2 * _VMEM_BITS
    assert capacities[19] == _VMEM_BITS


def test_bundle_to_accelerator_needs_no_files():
    """Inline core descriptions reach the same validators as file references."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    accelerator = bundle.to_accelerator()
    assert len(accelerator.core_list) == len(bundle.cores)


def test_inline_core_rejected_when_invalid():
    """A mutated bundle that is no longer a legal accelerator must not silently reach a solve."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    del bundle.cores[0]["memories"]["operand_buffer"]["size"]
    with pytest.raises(ValueError, match="not a valid accelerator"):
        bundle.validated_data()


def test_memory_alias_typo_is_rejected():
    """A mistyped alias would stop deduplicating and quietly inflate the modelled area."""
    data = HardwareBundle.from_yaml(TPU_V7).to_data()
    data["memory_aliases"] = [["9.vmem", "0.not_a_memory"]]
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
        ctx = optimize_allocation_co_generic(
            hardware=str(accelerator_path),
            workload=workload_path,
            experiment_id="test-bundle-asymmetric",
            output_path=tmpdir,
        )
    assert ctx.get("total_latency") > 0


# ── C4: the cost model ──────────────────────────────────────────────────────────────────────────


def test_tpu_v7_baseline_area_is_defensible():
    """The baseline must land where an Ironwood-class TensorCore slice plausibly lands.

    The dominant term is four 64 MiB VMEM scratchpads. At the N3 bitcell (0.0199 um^2), 70% array
    efficiency and the 1R1W port factor that works out near 24 Mb/mm^2 of macro, so each is tens of
    mm^2 -- not the 0.01 mm^2 an unpriced model would imply, and not the hundreds a triple-counted
    one would.
    """
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))

    assert report.technology_node == "n3"
    assert report.technology_declared

    vmem = _vmem(report)
    assert vmem.counted
    assert 20.0 < vmem.area_mm2 < 80.0, vmem.area_mm2
    # Macro density implied by the model, in Mb/mm^2 -- the number to argue with.
    assert 15.0 < vmem.total_bits / 1e6 / vmem.area_mm2 < 40.0

    # Four TensorCores + four scratchpads, no SparseCores / PHY / NoC / control.
    assert 100.0 < report.total_area_mm2 < 400.0, report.total_area_mm2
    assert report.memory_area_mm2 > report.compute_area_mm2  # SRAM dominates a modern accelerator
    assert report.off_die_memory_bits == 3298534883328  # HBM is priced as energy, not die area


def test_aliased_views_are_deduped_to_physical_capacity():
    """One physical VMEM seen as `vmem` (memory core) plus `operand_buffer` on each of the TensorCore's
    two compute cores must be billed once. Without the de-dup the four 128 MiB scratchpads would be
    counted as twelve views (~1536 MiB), triple the ~512 MiB the hardware actually has."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7_ALIASED))

    views = [m for c in report.cores for m in c.memories if m.bits_per_instance == _ALIASED_VMEM_BITS]
    assert len(views) == 12  # 4 vmem cores + 8 operand_buffer views (2 compute cores per TensorCore)
    counted = [m for m in views if m.counted]
    assert len(counted) == 4, "exactly the four physical VMEMs, one per TensorCore"
    assert all(m.memory_name == "vmem" for m in counted)
    assert sorted(m.core_id for m in counted) == [8, 9, 10, 11]
    assert all(m.alias_group and not m.counted for m in views if m.memory_name == "operand_buffer")

    # Physical on-die memory is the four counted VMEMs (plus small wreg/accumulator arrays), well below
    # the ~1536 MiB a non-deduped sum of all twelve views would report.
    assert 4 * _ALIASED_VMEM_BITS <= report.on_die_memory_bits < 6 * _ALIASED_VMEM_BITS


def test_mutating_a_non_counted_alias_view_does_not_change_area():
    """The de-dup, exercised under mutation: growing an aliased *view* (not the billed owner) must not
    move the priced area, while growing the physical owner does. This is what stops a search from
    'adding' silicon that does not exist by editing a second view of the same macro."""
    bundle = HardwareBundle.from_yaml(TPU_V7_ALIASED)
    baseline = evaluate_bundle_cost(bundle).total_area_mm2

    view = bundle.copy()
    view.cores[0]["memories"]["operand_buffer"]["size"] *= 2  # 0.operand_buffer is a non-owner view
    assert evaluate_bundle_cost(view).total_area_mm2 == pytest.approx(baseline)

    owner = bundle.copy()
    owner.cores[8]["memories"]["vmem"]["size"] *= 2  # 8.vmem is the billed owner of that group
    assert evaluate_bundle_cost(owner).total_area_mm2 > baseline


def test_memory_aliases_do_not_change_scheduling_capacity():
    """`memory_aliases` is a cost-model annotation only, never read on a scheduling/capacity path: the
    per-core capacity the allocator and tiler see must be identical with and without the aliases."""
    bundle = HardwareBundle.from_yaml(TPU_V7_ALIASED)
    assert bundle.memory_aliases  # the fixture declares them
    with_aliases = {c.id: c.get_memory_capacity() for c in bundle.to_accelerator().core_list}

    stripped = bundle.copy()
    stripped.accelerator.pop("memory_aliases")
    without = {c.id: c.get_memory_capacity() for c in stripped.to_accelerator().core_list}

    assert with_aliases == without


def test_vmem_is_one_core_per_tensorcore_priced_once():
    """Each TensorCore's 64 MiB VMEM is a single memory core, not aliased views inside the compute
    cores, so it is priced once with no memory_aliases needed."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))
    vmems = [m for c in report.cores for m in c.memories if m.total_bits == _VMEM_BITS]
    assert len(vmems) == 4  # one per TensorCore, no compute-core duplicate
    assert all(m.counted for m in vmems)
    assert sorted(m.core_id for m in vmems) == [9, 19, 29, 39]


def test_doubling_a_memory_raises_the_reported_area():
    """(a) The C4 done-condition: growth is not free."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    baseline = evaluate_bundle_cost(bundle)

    variant = bundle.copy()
    variant.cores[9]["memories"]["vmem"]["size"] *= 2
    grown = evaluate_bundle_cost(variant)

    delta = grown.total_area_mm2 - baseline.total_area_mm2
    vmem_area = _vmem(baseline).area_mm2
    # One of four scratchpads doubled: the added area is that scratchpad's, to within the
    # width-driven IO term which does not double with capacity -- a ~8% fraction at 64 MiB.
    assert delta == pytest.approx(vmem_area, rel=0.1)
    assert grown.peak_access_energy_pj_per_cycle > baseline.peak_access_energy_pj_per_cycle


def test_widening_a_port_costs_area_at_unchanged_capacity():
    """Capacity is not the only unpriced axis a search would otherwise exploit."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    baseline = evaluate_bundle_cost(bundle)

    variant = bundle.copy()
    for port in variant.cores[9]["memories"]["vmem"]["ports"]:
        port["bandwidth_max"] *= 4
    wider = evaluate_bundle_cost(variant)

    assert wider.on_die_memory_bits == baseline.on_die_memory_bits
    assert wider.total_area_mm2 > baseline.total_area_mm2
    assert wider.peak_access_energy_pj_per_cycle > baseline.peak_access_energy_pj_per_cycle


def test_adding_a_port_costs_area():
    bundle = HardwareBundle.from_yaml(TPU_V7)
    baseline = evaluate_bundle_cost(bundle)

    variant = bundle.copy()
    vmem = variant.cores[9]["memories"]["vmem"]
    extra = dict(vmem["ports"][0])
    extra["name"] = "r_port_2"
    vmem["ports"].append(extra)
    ported = evaluate_bundle_cost(variant)

    assert ported.total_area_mm2 > baseline.total_area_mm2


def test_enlarging_the_array_costs_area_quadratically_in_precision():
    """Compute area follows dimensions and operand precision, not a stored `unit_area`."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    baseline = evaluate_bundle_cost(bundle)
    mxu_area = _mxu(baseline).area_mm2

    doubled = bundle.copy()
    doubled.cores[0]["operational_array"]["sizes"] = [512, 512]
    grown = _mxu(evaluate_bundle_cost(doubled)).area_mm2
    assert grown == pytest.approx(4 * mxu_area, rel=1e-6)

    wider = bundle.copy()
    wider.cores[0]["operand_precision"]["input"] = "fp32"
    precise = _mxu(evaluate_bundle_cost(wider)).area_mm2
    # bf16 multiplies an 8-bit significand, fp32 a 24-bit one: the multiplier grows ~9x.
    assert 5.0 < precise / mxu_area < 12.0


def test_offchip_and_shim_cores_carry_no_die_area():
    """An HBM stack and an AIE shim front external memory; billing their capacity as SRAM would
    swamp everything else."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))
    hbm = next(c for c in report.cores if c.core_id == 40)
    assert not hbm.on_die
    assert hbm.area_mm2 == 0.0
    assert report.off_die_access_energy_pj_per_cycle > 0


def test_authored_energies_are_preserved_and_the_disagreement_is_reported():
    """The authored r_cost values keep driving the engine; the model says where it disagrees."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))
    vmem = _vmem(report)
    assert vmem.authored_read_energy_pj == 200.0  # unchanged, as authored
    assert vmem.read_energy_pj > vmem.authored_read_energy_pj
    assert any("disagree" in w for w in report.warnings)


def test_access_energy_is_sublinear_in_capacity_and_linear_in_width():
    bundle = HardwareBundle.from_yaml(TPU_V7)

    def vmem_read(b):
        return _vmem(evaluate_bundle_cost(b)).read_energy_pj

    base = vmem_read(bundle)

    bigger = bundle.copy()
    bigger.cores[9]["memories"]["vmem"]["size"] *= 4
    assert base < vmem_read(bigger) < 4 * base  # grows, but sub-linearly

    wider = bundle.copy()
    for port in wider.cores[9]["memories"]["vmem"]["ports"]:
        port["bandwidth_max"] *= 2
    assert vmem_read(wider) > 1.9 * base  # linear in access width


def test_aie_bundle_is_priceable_and_reports_what_it_cannot_model():
    """The AIE2 array must be priceable too -- and its unmodelled compute must read as unknown,
    not as zero."""
    bundle = HardwareBundle.from_yaml(AIE2_STRIX)
    report = evaluate_bundle_cost(bundle)

    assert report.memory_area_mm2 > 0
    assert not report.technology_declared
    assert report.technology_node == "n5"
    compute_tiles = [c for c in report.cores if c.compute and c.on_die]
    assert compute_tiles and all(not c.compute.modelled for c in compute_tiles)


def test_unknown_technology_node_is_an_error_not_a_default():
    bundle = HardwareBundle.from_yaml(TPU_V7)
    with pytest.raises(ValueError, match="Unknown technology node"):
        evaluate_bundle_cost(bundle, technology_node="n0")
    assert set(TECH_NODES) >= {"n3", "n5", "n7"}


def test_api_hardware_cost_report_round_trips(tmp_path):
    out = tmp_path / "hardware_cost.json"
    report = hardware_cost_report(TPU_V7, str(out))
    assert out.exists()
    assert report["total_area_mm2"] > 0
    assert report["cores"][0]["memories"][0]["memory_name"]


# ── C4: the budget guard ────────────────────────────────────────────────────────────────────────


def test_baseline_is_its_own_budget():
    bundle = HardwareBundle.from_yaml(TPU_V7)
    budget = HardwareBudget.from_bundle(bundle)
    verdict = check_budget(bundle, budget)
    assert isinstance(verdict, BudgetVerdict)
    assert verdict.ok, verdict.violations


def test_over_budget_variant_is_rejected_without_a_solve(monkeypatch):
    """(c) The C4 done-condition: rejection happens before anything is scheduled."""
    bundle = HardwareBundle.from_yaml(TPU_V7)
    budget = HardwareBudget.from_bundle(bundle)

    variant = bundle.copy()
    variant.cores[9]["memories"]["vmem"]["size"] *= 2
    verdict = check_budget(variant, budget)
    assert not verdict.ok
    assert "area" in verdict.violations[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        accelerator_path = variant.materialize(tmpdir)

        # Any stage running at all would mean the budget was checked too late.
        def _fail(*args, **kwargs):
            raise AssertionError("the pipeline must not start for an over-budget variant")

        monkeypatch.setattr(MainStage, "run", _fail)
        with pytest.raises(HardwareBudgetExceededError):
            optimize_allocation_co_generic(
                hardware=str(accelerator_path),
                workload="stream/inputs/examples/workload/resnet18.onnx",
                experiment_id="test-budget-reject",
                output_path=tmpdir,
                hardware_budget=budget,
            )


def test_budget_headroom_admits_a_bounded_increase():
    bundle = HardwareBundle.from_yaml(TPU_V7)
    variant = bundle.copy()
    variant.cores[9]["memories"]["vmem"]["size"] *= 2
    growth = evaluate_bundle_cost(variant).total_area_mm2 / evaluate_bundle_cost(bundle).total_area_mm2

    assert not check_budget(variant, HardwareBudget.from_bundle(bundle, headroom=0.05)).ok
    generous = HardwareBudget.from_bundle(bundle, headroom=math.ceil(growth * 100) / 100)
    assert check_budget(variant, generous).ok
    assert assert_within_budget(variant, generous).total_area_mm2 > 0


# ── T1-5: the model's own caveats, no longer discarded ──────────────────────────────────────────


def test_the_report_quotes_its_own_published_tolerance():
    """The module publishes a tolerance under an ACCURACY CLAIM heading; carrying it on the report is
    quoting the producer, not inventing an error bar."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))
    assert report.accuracy_claim == ACCURACY_CLAIM
    assert "2x" in report.accuracy_claim
    assert report.to_dict()["accuracy_claim"] == ACCURACY_CLAIM


def test_authored_disagreements_are_countable_not_only_a_warning_sentence():
    """The count is the reportable fact: "the model and the YAML disagree about N memories" says
    how much of the energy accounting to believe, and a prose warning cannot be counted."""
    report = evaluate_bundle_cost(HardwareBundle.from_yaml(TPU_V7))
    assert report.authored_disagreements, "TPU7x authors r_cost values two orders of magnitude off"
    assert all(": authored " in line for line in report.authored_disagreements)
    # Still summarised into `warnings` for a reader who only reads those.
    assert any("authored access energies disagree" in w for w in report.warnings)


def test_a_bundle_with_no_declared_array_says_its_compute_was_not_modelled():
    """An aie2 tile declares no operational_array, so its compute area is UNKNOWN and the bundle's
    total is a lower bound. Reporting it as an estimate would understate the silicon silently."""
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
