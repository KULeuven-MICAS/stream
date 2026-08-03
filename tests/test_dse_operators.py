"""Golden fixtures for the guarded operator registry (Phase D).

The whole claim of the registry is that the preconditions live in code rather than in prompt text.
That claim is only worth anything if each rule is pinned by a test that asserts the operator IS
offered on evidence that satisfies it and IS NOT offered on evidence that does not — including the
two cases a careless implementation gets backwards:

* ``evidence: "none"`` (no CME) read as "no stall", i.e. a modelling gap read as a licence to grow;
* an over-budget hardware edit reaching a solve because the engine's own ``unit_area: 0`` makes
  growth free.

The fixtures below are the real TPU7x SwiGLU numbers (exploration 198, seq=2048, fully fused):
``Elt_Mul`` = 896 ideal cycles + 1326 stall at ``vregs`` on ports rw_port_1/rw_port_2 = 1326/863,
``Silu`` with no CME at all, four MXU cores binding the overlap at 14,942 cycles of slack, and a
solve that reports ``mip_gap: null``.
"""

import copy
from dataclasses import replace

import pytest

from stream.dse import (
    Objective,
    ObjectiveKind,
    OperatorScorecard,
    Residual,
    RunEvidence,
    apply_operator,
    offer_operators,
    post_hoc_check,
    post_hoc_reduction_check,
)
from stream.dse.evidence import DEFAULT_NOISE_FLOOR_RELATIVE
from stream.dse.operators import GROWTH_FACTORS, SHRINK_HEADROOM
from stream.dse.residual import MIN_TRUST
from stream.hardware.bundle import HardwareBundle
from stream.hardware.cost import HardwareBudget, evaluate_bundle_cost

TPU_V7 = "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml"

MXU_CORES = [0, 2, 4, 6]
VPU_CORES = [1, 3, 5, 7]


# ── Fixtures: the artifacts a run writes ────────────────────────────────────────────────────────


def _level(index, name, operands, stall=0.0, per_port=None, utilization=1.0):
    per_port = per_port if per_port is not None else {"r_port_1": stall}
    return name, {
        "index": index,
        "operands": operands,
        "stall_cycles": max(stall, 0.0),
        "slack_cycles": max(-stall, 0.0),
        "per_port": per_port,
        "utilization": utilization,
    }


def _mxu_node(name):
    """A GEMM on the MXU quad: at its compute ideal but for a 36-cycle scratchpad stall."""
    return {
        "name": name,
        "op": "Gemm",
        "cores": MXU_CORES,
        "evidence": "cme",
        "estimator": "zigzag",
        "latency": 57390.0,
        "ideal_cycle": 57344,
        "efficiency": 0.9991984666318174,
        "mac_utilization": 1.0,
        "memory_levels": dict(
            [
                _level(0, "wreg", ["I2"], per_port={"r_port_1": 0.0, "w_port_1": 0.0}),
                _level(1, "accumulator", ["O"], per_port={"r_port_1": 0.0, "r_port_2": -6.0}),
                _level(
                    2, "operand_buffer", ["I1", "I2", "O"], stall=36.0, per_port={"r_port_1": 0.0, "w_port_1": 36.0}
                ),
            ]
        ),
        "operand_memory_utilization": {
            "basis": "per-operand active memory level index; not the flat mem_level_list index",
            "shared": {"O": [1.0, 0.248046875], "A": [0.248046875], "B": [1.0, 0.248046875]},
        },
    }


ELT_MUL = {
    "name": "Elt_Mul",
    "op": "Mul",
    "cores": VPU_CORES,
    "evidence": "cme",
    "estimator": "zigzag",
    "latency": 2270.0,
    "ideal_cycle": 896,
    "efficiency": 0.3947136563876652,
    "mac_utilization": 0.3947136563876652,
    "memory_levels": dict(
        [
            _level(
                0,
                "vregs",
                ["I1", "I2", "O"],
                stall=1326.0,
                per_port={"rw_port_1": 1326.0, "rw_port_2": 863.0},
                utilization=2.5,
            ),
            _level(1, "operand_buffer", ["I1", "I2", "O"], per_port={"r_port_1": 0.0, "w_port_1": -810.0}),
        ]
    ),
    "operand_memory_utilization": {
        "basis": "per-operand active memory level index; not the flat mem_level_list index",
        "shared": {"O": [0.75, 0.041015625], "A": [0.75, 0.041015625], "B": [0.75, 0.041015625]},
    },
}

SILU_NO_CME = {
    "name": "Silu",
    "op": "Silu",
    "cores": VPU_CORES,
    "evidence": "none",
    "estimator": "ideal-cycle",
    "latency": 3584.0,
    "ideal_cycle": 3584.0,
    "efficiency": 1.0,
    "mac_utilization": None,
    "memory_levels": None,
    "operand_memory_utilization": None,
}


def progress_json(nodes):
    return {"stages": [{"key": "core_cost", "artifact": {"groups": [{"group": 0, "nodes": nodes}]}}]}


# Per-node performance, as only the AllocationIR reports it. `mac_spatial_utilization` is the array
# FILL and is 1.0 for both the GEMMs and the elementwise multiply; the tracker's `mac_utilization`
# (ZigZag's mac_utilization2) is a different quantity — ideal/actual cycles — and reading it as the
# fill would say Elt_Mul leaves 61% of the VPU idle when in fact it fills it and waits on vregs.
PERFORMANCE_NODES = {
    "Gemm_Left": {
        "latency_cycles": 57390,
        "ideal_compute_cycles": 57344.0,
        "mac_spatial_utilization": 1.0,
        "compute_efficiency": 0.9991984666318174,
        "fallback": False,
    },
    "Gemm_Right": {
        "latency_cycles": 57390,
        "ideal_compute_cycles": 57344.0,
        "mac_spatial_utilization": 1.0,
        "compute_efficiency": 0.9991984666318174,
        "fallback": False,
    },
    "Gemm_Down": {
        "latency_cycles": 57360,
        "ideal_compute_cycles": 57344.0,
        "mac_spatial_utilization": 1.0,
        "compute_efficiency": 0.999721059972106,
        "fallback": False,
    },
    "Elt_Mul": {
        "latency_cycles": 2270,
        "ideal_compute_cycles": 896.0,
        "mac_spatial_utilization": 1.0,
        "compute_efficiency": 0.3947136563876652,
        "fallback": False,
    },
    "Silu": {
        "latency_cycles": 3584,
        "ideal_compute_cycles": 3584.0,
        "mac_spatial_utilization": None,
        "compute_efficiency": 1.0,
        "fallback": False,
    },
}


def allocation_json(  # noqa: PLR0913 -- a fixture builder; each argument is one knob a test varies
    *,
    binding=("Core(0, zigzag.compute)",),
    slack=((("Core(0, zigzag.compute)", "core"), 14942),),
    compute_bound=177994,
    transfer_bound=9088,
    recurrence=0,
    mip_gap=None,
    layers=("Gemm_Left", "Silu", "Gemm_Right", "Elt_Mul", "Gemm_Down"),
    n_groups=1,
    nodes=None,
    occupancy=(),
):
    groups = [
        {"name": f"group_{i}", "layers": list(chunk), "intra_core_tiling": [["z2", 256]]}
        for i, chunk in enumerate(_split(layers, n_groups))
    ]
    return {
        "total_latency": 1392195.0,
        "groups": [
            {
                "name": "group_0",
                "latency": 1392195.0,
                "allocation": {
                    "latency": {"total": 1392195, "per_iteration": 187082, "overlap_between_iterations": 14923},
                    "solve": {"status": "OPTIMAL", "solver": "gurobi", "mip_gap": mip_gap},
                    "fused_groups": groups,
                    "performance": {
                        "latency": {
                            "total": 1392195,
                            "per_iteration": 187082,
                            "overlap_between_iterations": 14923,
                        },
                        "bottleneck": {
                            "compute_bound_cycles": compute_bound,
                            "transfer_bound_cycles": transfer_bound,
                        },
                        "aggregate": {
                            "compute_cores_available": 12,
                            "compute_cores_used": 8,
                            "end_to_end_mac_utilization": 0.9885511727882947,
                        },
                        "nodes": {
                            name: {"kind": "compute", "n_cores": 4, **row}
                            for name, row in (PERFORMANCE_NODES if nodes is None else nodes).items()
                        },
                        "overlap": {
                            "overlap_cycles": 14923,
                            "binding_resources": list(binding),
                            "per_resource_slack": [
                                {"resource": name, "kind": kind, "slack_cycles": cycles}
                                for (name, kind), cycles in slack
                            ],
                            "recurrence_bound_cycles": recurrence,
                        },
                        "memory_occupancy": [
                            {
                                "core_id": core_id,
                                "core_name": f"core_{core_id}",
                                "resident_bits": resident,
                                "capacity_bits": capacity,
                                "utilization": resident / capacity if capacity else None,
                                "tensors": [],
                            }
                            for core_id, resident, capacity in occupancy
                        ],
                    },
                },
            }
        ],
    }


def _split(layers, n_groups):
    if n_groups <= 1:
        return [layers]
    return [[layer] for layer in layers]


CORE_SLACK = tuple(((f"Core({c}, zigzag.compute)", "core"), 14942) for c in MXU_CORES) + tuple(
    ((f"Core({c}, zigzag.compute)", "core"), 181228) for c in VPU_CORES
)
CORE_BINDING = tuple(f"Core({c}, zigzag.compute)" for c in MXU_CORES)


@pytest.fixture
def bundle():
    return HardwareBundle.from_yaml(TPU_V7)


@pytest.fixture
def swiglu_evidence():
    """The real fused SwiGLU run: MXU cores bind, no link does, Silu has no CME."""
    return RunEvidence.from_artifacts(
        progress=progress_json(
            [_mxu_node("Gemm_Left"), SILU_NO_CME, _mxu_node("Gemm_Right"), ELT_MUL, _mxu_node("Gemm_Down")]
        ),
        allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK),
    )


MAPPING = {"nb_cols_to_use": 4, "intra_core_tiling": [{"dim": "Gemm_Left.D0", "tile": 256}]}


def _offer(result, operator_id, **target_match):
    for offer in result.offered:
        if offer.operator_id != operator_id:
            continue
        if all(offer.target.get(k) == v or offer.args.get(k) == v for k, v in target_match.items()):
            return offer
    return None


def _reduction_veto(result, operator_id, **target_match):
    """A veto recorded under EXACTLY this operator id.

    The reduction operators share their target with the growth operators, and `_veto`'s family
    wildcard would happily return `core.memory.*`'s Rule 1 refusal when the test is about Rule 1's
    dual — two different rules reaching the same conclusion for different reasons.
    """
    for veto in result.vetoed:
        if veto.operator_id != operator_id:
            continue
        if all(veto.target.get(k) == v for k, v in target_match.items()):
            return veto
    return None


def _veto(result, operator_id, **target_match):
    """A veto whose id matches exactly, where a trailing `*` in the record is a family wildcard."""
    wanted = operator_id.split(".")
    for veto in result.vetoed:
        recorded = veto.operator_id.split(".")
        if len(recorded) != len(wanted):
            continue
        if any(r not in ("*", w) and w not in ("*", r) for r, w in zip(recorded, wanted, strict=True)):
            continue
        if all(veto.target.get(k) == v for k, v in target_match.items()):
            return veto
    return None


# ── D7: the predicted delta ─────────────────────────────────────────────────────────────────────


def test_predicted_delta_is_the_gap_to_the_second_largest_port_stall(bundle, swiglu_evidence):
    """`stall_slack_comb = max(...)` over ports, so relieving rw_port_1 (1326) only exposes
    rw_port_2 (863). The saving is 463 cycles, NOT 1326."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    offer = _offer(result, "core.memory.bandwidth", memory="vregs")
    assert offer is not None, "the textbook bandwidth-growth candidate must be offered"
    assert offer.predicted_delta.cycles == pytest.approx(1326.0 - 863.0)
    assert offer.predicted_delta.scope == "node"
    assert offer.predicted_delta.bound == "upper"
    assert "second largest" in offer.predicted_delta.derivation


def test_the_delta_is_a_whole_node_figure_and_says_so(bundle, swiglu_evidence):
    """`stall_or_slack` already carries (period_count - 1). A consumer that multiplies by the
    8 steady-state iterations would report 3704 cycles for a 463-cycle move."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    offer = _offer(result, "core.memory.bandwidth", memory="vregs")
    assert "must not be multiplied by an iteration count" in offer.predicted_delta.derivation


# ── D2: Rule 1, the veto ────────────────────────────────────────────────────────────────────────


def test_a_level_that_does_not_stall_is_never_offered_for_either_knob(bundle, swiglu_evidence):
    """The MXU's weight registers are saturated (shared utilization 1.0) but do not stall. Neither
    more capacity nor more bandwidth can change latency there, so neither is offered."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(result, "core.memory.bandwidth", memory="wreg") is None
    assert _offer(result, "core.memory.capacity", memory="wreg") is None
    veto = _veto(result, "core.memory.*", memory="wreg")
    assert veto is not None and veto.rule == "rule-1"
    assert "does not stall" in veto.reason


def test_evidence_none_is_not_no_stall(bundle):
    """A node with no CME must produce NO memory operator at all. Absence of evidence is not
    proof of legality -- this is the case a careless implementation reads as 'nothing stalls'."""
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([SILU_NO_CME]),
        allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK),
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    assert [o for o in result.offered if o.operator_id.startswith("core.")] == []
    veto = _veto(result, "core.memory.*", node="Silu")
    assert veto is not None and "evidence=none" in veto.reason


def test_a_stall_alone_does_not_select_capacity(bundle, swiglu_evidence):
    """`vregs` stalls 1326 cycles, so the veto passes -- but the stall vector is bandwidth-derived
    by construction and cannot choose between the two knobs. Capacity stays unoffered until
    saturation or an infeasibility report selects it."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(result, "core.memory.bandwidth", memory="vregs") is not None
    assert _offer(result, "core.memory.capacity", memory="vregs") is None
    veto = _veto(result, "core.memory.capacity", memory="vregs")
    assert veto is not None and "cannot choose between the two knobs" in veto.reason


def test_a_stalling_and_saturated_level_does_select_capacity(bundle):
    """Both halves of Rule 1: the stall lets it through, saturation picks the knob."""
    node = {
        **ELT_MUL,
        "operand_memory_utilization": {"shared": {"A": [1.0, 0.04], "B": [1.0, 0.04], "O": [1.0, 0.04]}},
    }
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([node]), allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK)
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    offer = _offer(result, "core.memory.capacity", memory="vregs")
    assert offer is not None
    assert "is full" in offer.evidence
    # Capacity does not enter the stall arithmetic, so the ceiling is the level's whole stall and
    # the derivation has to say the realised saving may be zero.
    assert offer.predicted_delta.cycles == pytest.approx(1326.0)
    assert "may be zero" in offer.predicted_delta.derivation


def test_an_infeasible_solve_selects_capacity_from_the_typed_report(bundle):
    """The other capacity selector: an unmet memory_capacity constraint names the resource and the
    exact shortfall, which no feasible run's stall vector can."""
    infeasibility = {
        "status": "INFEASIBLE",
        "resources": [
            {
                "resource": {"kind": "core", "id": "1", "label": "Core 1"},
                "unmet": {
                    "family": "memory_capacity",
                    "demand_value": 3_500_000.0,
                    "bound_value": 2_097_152.0,
                    "gap": 1_402_848.0,
                    "unit": "bits",
                    "levers": ["reduce the intra-core tile", "grow Core 1 vregs"],
                },
            }
        ],
        "summary": "Core 1 on-chip memory capacity exceeded",
    }
    evidence = RunEvidence.from_artifacts(progress=progress_json([]), infeasibility=infeasibility)
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    offer = _offer(result, "core.memory.capacity", core=1)
    assert offer is not None
    assert offer.args["factor"] == GROWTH_FACTORS[0], "1.67x short rounds up to the smallest declared step"
    assert "INFEASIBLE" in offer.evidence and "3.5e+06" in offer.evidence
    assert offer.predicted_delta.cycles == 0.0, "an infeasible solve has no latency to improve on"


# ── D2/D4: Rule 2, ordering and the discrepancy gate ────────────────────────────────────────────


def test_core_operators_come_before_system_ones(bundle, swiglu_evidence):
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    tiers = [str(o.tier) for o in result.offered]
    assert tiers == sorted(tiers, key=["core", "system", "link"].index)
    assert tiers[0] == "core", "a measured stall outranks a system-level reshuffle by rule, not by size"


def test_a_node_at_its_compute_ideal_gets_no_core_operator(bundle):
    """Rule 2: no discrepancy, nothing for a core-level operator to close."""
    ideal = {**ELT_MUL, "name": "Perfect", "efficiency": 1.0, "latency": 896.0}
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([ideal]), allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK)
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    assert [o for o in result.offered if o.operator_id.startswith("core.")] == []
    assert any(v.rule == "rule-2" for v in result.vetoed)


# ── D3: Rule 1's exception, the coupled array resize ────────────────────────────────────────────


def test_a_resize_scales_core_local_levels_that_serve_the_resized_dimension(bundle):
    """Per level, not globally. The MXU accumulator is served along D2, so a D2 doubling doubles its
    capacity AND its port width; the per-PE weight registers serve no dimension (they replicate with
    the array, which the cost model already counts) and must not be touched."""
    applied = apply_operator("core.array.resize", {"cores": [0], "dims": {"D2": 2}}, bundle=bundle)
    before, after = bundle.cores[0], applied.bundle.cores[0]

    assert after["operational_array"]["sizes"] == [256, 512]
    assert after["memories"]["accumulator"]["size"] == before["memories"]["accumulator"]["size"] * 2
    assert (
        after["memories"]["accumulator"]["ports"][0]["bandwidth_max"]
        == before["memories"]["accumulator"]["ports"][0]["bandwidth_max"] * 2
    )
    assert after["memories"]["wreg"]["size"] == before["memories"]["wreg"]["size"]


def test_a_resize_never_touches_a_shared_scratchpad(bundle):
    """The disaster this rule exists to prevent. `operand_buffer` is served along D1 AND D2, so a
    naive per-dimension coupling would scale it -- but it is an alias of the VMEM core's memory, and
    a 256->512 resize on both axes would then demand 1,048,576 bits/cycle (~228 TB/s, 4x the part)."""
    applied = apply_operator("core.array.resize", {"cores": [0], "dims": {"D1": 2, "D2": 2}}, bundle=bundle)
    before, after = bundle.cores[0], applied.bundle.cores[0]
    assert after["memories"]["operand_buffer"]["size"] == before["memories"]["operand_buffer"]["size"]
    assert (
        after["memories"]["operand_buffer"]["ports"][0]["bandwidth_max"]
        == before["memories"]["operand_buffer"]["ports"][0]["bandwidth_max"]
    )
    assert applied.bundle.cores[8]["memories"]["vmem"]["size"] == bundle.cores[8]["memories"]["vmem"]["size"]


def test_a_resize_is_not_offered_for_a_node_that_is_waiting_on_memory(bundle, swiglu_evidence):
    """Elt_Mul FILLS the VPU array (spatial utilization 1.0) and still runs at 0.39 efficiency: its
    cycles go to vregs stalls, not to array size. A bigger array shortens the term that is not
    binding — the bandwidth operator is what addresses this node."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(result, "core.array.resize", node="Elt_Mul") is None
    veto = _veto(result, "core.array.resize", node="Elt_Mul")
    assert veto is not None and veto.rule == "rule-1-exception"
    assert "set by memory stalls" in veto.reason
    # ...and the operator that DOES address it is offered on the same node.
    assert _offer(result, "core.memory.bandwidth", node="Elt_Mul") is not None


def test_a_resize_is_offered_for_a_full_array_on_a_compute_bound_node(bundle, swiglu_evidence):
    """The MXU GEMMs fill their array and run at 0.9992 efficiency, so the 57,344-cycle compute
    ideal IS the latency. Doubling one array axis halves it."""
    generous = HardwareBudget.from_bundle(bundle, headroom=3.0)
    result = offer_operators(swiglu_evidence, bundle=bundle, budget=generous, mapping_params=MAPPING)
    offer = _offer(result, "core.array.resize", node="Gemm_Left", dim="D2")
    assert offer is not None
    assert offer.target["from"] == 256 and offer.target["to"] == 512
    assert offer.predicted_delta.cycles == pytest.approx(57344.0 * 0.5)
    # The coupling has to be stated on the offer, not only performed on application.
    assert any("served_dimensions" in c for c in offer.couples)
    assert any("memory_aliases" in c for c in offer.couples)


def test_the_tracker_mac_utilization_is_not_mistaken_for_the_array_fill(swiglu_evidence):
    """progress.json's `mac_utilization` is ZigZag's mac_utilization2 — ideal/actual cycles, the
    same number as compute_efficiency. Only the AllocationIR reports the spatial fill, and reading
    the wrong one would say Elt_Mul leaves 61% of its array idle."""
    node = swiglu_evidence.node("Elt_Mul")
    assert node.mac_spatial_utilization == 1.0
    assert node.compute_efficiency == pytest.approx(0.3947136563876652)


def test_the_post_hoc_guard_rejects_a_resize_whose_utilization_fell(swiglu_evidence):
    """ZigZag's validator checks none of the coupling and degrades silently, so the only reliable
    test is the outcome. A resized array that is not being fed must be rejected, not scored."""
    starved = {**PERFORMANCE_NODES["Gemm_Left"], "mac_spatial_utilization": 0.5}
    worse = RunEvidence.from_artifacts(
        progress=progress_json([_mxu_node("Gemm_Left")]),
        allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK, nodes={"Gemm_Left": starved}),
    )
    reason = post_hoc_check("core.array.resize", {"node": "Gemm_Left"}, swiglu_evidence, worse)
    assert reason is not None and "mac_spatial_utilization fell" in reason

    same = RunEvidence.from_artifacts(
        progress=progress_json([_mxu_node("Gemm_Left")]),
        allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK),
    )
    assert post_hoc_check("core.array.resize", {"node": "Gemm_Left"}, swiglu_evidence, same) is None


# ── D1/C4: the budget, enforced before any solve ────────────────────────────────────────────────


def test_an_over_budget_edit_is_rejected_before_a_solve(bundle, swiglu_evidence):
    """Widening the MXU scratchpad means widening the whole aliased 128 MiB VMEM. The engine's own
    `unit_area: 0` would report that as free; the computed cost model does not."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(result, "core.memory.bandwidth", memory="operand_buffer") is None
    veto = next(
        (v for v in result.vetoed if v.operator_id == "core.memory.bandwidth" and v.rule == "budget"),
        None,
    )
    assert veto is not None and "before any solve" in veto.reason


def test_a_generous_budget_admits_what_a_tight_one_refuses(bundle, swiglu_evidence):
    """The guard is the budget, not a hard-coded refusal: the same edit passes at 4x headroom."""
    generous = HardwareBudget.from_bundle(bundle, headroom=3.0)
    result = offer_operators(swiglu_evidence, bundle=bundle, budget=generous, mapping_params=MAPPING)
    offer = _offer(result, "core.memory.bandwidth", memory="operand_buffer")
    assert offer is not None
    assert offer.cost["area_mm2"] > evaluate_bundle_cost(bundle).total_area_mm2


def test_no_hardware_operator_is_offered_without_a_bundle_to_price(swiglu_evidence):
    """An edit that cannot be priced cannot be certified within budget, so it is not offered --
    the same discipline as `evidence: "none"`."""
    result = offer_operators(swiglu_evidence, bundle=None, mapping_params=MAPPING)
    assert [o for o in result.offered if o.kind == "hardware"] == []


def test_the_alias_group_moves_together(bundle):
    """A level named in `memory_aliases` is one physical memory. Scaling only the view the stalling
    node sits behind would change what the solve sees without changing what the cost model bills."""
    applied = apply_operator(
        "core.memory.bandwidth", {"cores": MXU_CORES, "memory": "operand_buffer", "factor": 2}, bundle=bundle
    )
    widened = applied.bundle
    assert widened.cores[0]["memories"]["operand_buffer"]["ports"][0]["bandwidth_max"] == 262144 * 2
    assert widened.cores[1]["memories"]["operand_buffer"]["ports"][0]["bandwidth_max"] == 262144 * 2
    assert widened.cores[8]["memories"]["vmem"]["ports"][0]["bandwidth_max"] == 262144 * 2


# ── D5: Rule 3, the system tier ─────────────────────────────────────────────────────────────────


def test_a_recurrence_bound_schedule_gets_no_system_operator(bundle):
    """II = per_iteration - overlap = 172,159 cycles. When RecMII reaches that, no allocation,
    fusion or tiling change can overlap iterations a loop-carried state forbids overlapping."""
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([ELT_MUL]),
        allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK, recurrence=172_159),
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    assert [o for o in result.offered if str(o.tier) == "system"] == []
    veto = next(v for v in result.vetoed if v.rule == "rule-3")
    assert "recurrence bound" in veto.reason


def test_the_tile_operator_needs_a_named_dimension(bundle, swiglu_evidence):
    """The mapper's tile namespace is `<node>.D<n>`; the solved IR only reports the global `z` dims.
    With no spec to edit there is no legal target, so nothing is offered rather than guessed."""
    with_spec = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(with_spec, "system.tiling.intra_core") is not None

    without = offer_operators(swiglu_evidence, bundle=bundle, mapping_params={"nb_cols_to_use": 4})
    assert _offer(without, "system.tiling.intra_core") is None
    assert any(v.operator_id == "system.tiling.intra_core" for v in without.vetoed)


def test_a_feasible_fully_fused_run_is_not_offered_a_cut(bundle, swiglu_evidence):
    """Nothing measured says one group is too large: it solved. Cutting it can only add traffic."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(result, "system.fusion.cut") is None
    veto = next(v for v in result.vetoed if v.operator_id == "system.fusion.cut")
    assert "already one fused group and it solved" in veto.reason


def test_a_layer_by_layer_transfer_bound_run_is_offered_re_fusion(bundle):
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([_mxu_node("Gemm_Left")]),
        allocation=allocation_json(
            binding=CORE_BINDING, slack=CORE_SLACK, n_groups=5, compute_bound=1000, transfer_bound=9088
        ),
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    offer = _offer(result, "system.fusion.cut")
    assert offer is not None
    assert offer.args["fusion_cut_points"] is None
    assert offer.predicted_delta.cycles == pytest.approx(9088.0)
    assert offer.predicted_delta.scope == "iteration"


def test_the_rebalance_bound_is_the_load_balance_bound(bundle, swiglu_evidence):
    """busy_i = per_iteration - slack_i. The MXU quad is busy 172,140 cycles against an 88,997-cycle
    mean over the eight active cores, so the ceiling on any redistribution is their difference."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    offer = _offer(result, "system.alloc.cores")
    assert offer is not None
    busy_mxu, busy_vpu = 187082 - 14942, 187082 - 181228
    mean = (4 * busy_mxu + 4 * busy_vpu) / 8
    assert offer.predicted_delta.cycles == pytest.approx(busy_mxu - mean)
    assert "operator_types may forbid it" in offer.predicted_delta.derivation


def test_no_rebalance_when_the_mapper_already_reaches_every_core(bundle, swiglu_evidence):
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params={**MAPPING, "nb_cols_to_use": 8})
    assert _offer(result, "system.alloc.cores") is None
    veto = next(v for v in result.vetoed if v.operator_id == "system.alloc.cores")
    assert "no wider allocation" in veto.reason


# ── D6: Rule 4, the NoC / off-chip veto ─────────────────────────────────────────────────────────


def test_a_link_with_slack_to_spare_is_never_offered(bundle, swiglu_evidence):
    """The real run: every link has more slack than the four cores that cap the overlap, so its
    transfers are already hidden. This is the knob that always looks helpful and is not."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(result, "link.bandwidth") is None
    veto = next(v for v in result.vetoed if v.operator_id == "link.bandwidth")
    assert veto.rule == "rule-4" and "no link is in the solver's binding set" in veto.reason


def test_a_binding_link_on_a_compute_bound_schedule_is_still_vetoed(bundle):
    """Even when the solver says a link binds the overlap: a wider link cannot shorten a slot whose
    latency is set by the array."""
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([ELT_MUL]),
        allocation=allocation_json(
            binding=("CL(Any, Any, bw=65536)",),
            slack=((("CL(Any, Any, bw=65536)", "link"), 100),) + CORE_SLACK,
            compute_bound=177994,
            transfer_bound=9088,
        ),
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(result, "link.bandwidth") is None
    veto = next(v for v in result.vetoed if v.operator_id == "link.bandwidth")
    assert "compute-bound" in veto.reason


def test_a_binding_link_on_a_transfer_bound_schedule_is_offered(bundle):
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([ELT_MUL]),
        allocation=allocation_json(
            binding=("CL(Any, Any, bw=65536)",),
            slack=((("CL(Any, Any, bw=65536)", "link"), 100),) + CORE_SLACK,
            compute_bound=20_000,
            transfer_bound=160_000,
        ),
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    offer = _offer(result, "link.bandwidth")
    assert offer is not None
    assert offer.args["link_bandwidth"] == 65536, "the bandwidth in the solver's key is the join to the bundle"
    exposed = 187082 - 100
    assert offer.predicted_delta.cycles == pytest.approx(min(160_000, exposed * 0.5))


def test_a_saving_below_the_noise_floor_is_vetoed(bundle):
    """Rule 4's second half. The link binds and the schedule is transfer-bound, but the exposed
    cycles are small enough that the predicted saving is inside the solver's own tolerance."""
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([ELT_MUL]),
        allocation=allocation_json(
            binding=("CL(Any, Any, bw=65536)",),
            slack=((("CL(Any, Any, bw=65536)", "link"), 186_000),) + CORE_SLACK,
            compute_bound=20_000,
            transfer_bound=160_000,
        ),
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING)
    assert _offer(result, "link.bandwidth") is None
    veto = next(v for v in result.vetoed if "noise floor" in v.reason)
    assert veto.rule == "rule-4"


def test_a_null_mip_gap_means_the_floor_is_unknown_not_zero(swiglu_evidence):
    """Gurobi defines no gap for the lexicographic multi-objective model, so `mip_gap` is null on
    the default path. Reading null as 0 would make every rounding difference count as evidence."""
    floor = swiglu_evidence.noise_floor
    assert floor.known is False
    assert floor.relative == DEFAULT_NOISE_FLOOR_RELATIVE
    assert floor.cycles == pytest.approx(DEFAULT_NOISE_FLOOR_RELATIVE * 187082)
    assert "no optimality gap" in floor.source


def test_a_reported_mip_gap_is_used_verbatim(bundle):
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([ELT_MUL]),
        allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK, mip_gap=0.001),
    )
    del bundle
    assert evidence.noise_floor.known is True
    assert evidence.noise_floor.cycles == pytest.approx(0.001 * 187082)


# ── D1: the record every offer carries ──────────────────────────────────────────────────────────


def test_every_offer_declares_the_full_quadruple(bundle, swiglu_evidence):
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)
    assert result.offered, "the fixture must produce at least one legal move"
    for offer in result.offered:
        assert offer.evidence and any(c.isdigit() for c in offer.evidence), "preconditions cite numbers"
        assert offer.effect
        assert offer.predicted_delta.derivation
        assert offer.predicted_delta.cycles >= 0.0
        if offer.kind == "hardware":
            assert offer.cost and offer.cost["area_mm2"] > 0


def test_the_serialized_form_reports_the_noise_floor_per_offer(bundle, swiglu_evidence):
    """A 463-cycle move on a 187,082-cycle iteration is legal and still invisible end to end. The
    guards do not decide that; they report it so a ranker can."""
    payload = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING).as_dict()
    by_id = {o["operator"]: o for o in payload["offered"]}
    assert by_id["core.memory.bandwidth"]["clears_noise_floor"] is False
    assert by_id["system.alloc.cores"]["clears_noise_floor"] is True
    assert payload["noise_floor"]["known"] is False


# ── E0: Rule 1's dual — shrink what the measurement says is over-provisioned ─────────────────────

# The TPU7x scratchpad is the case this whole rule exists for: 4 x 128 MiB of VMEM is 165.6 of the
# bundle's 172.8 mm2, and the fused SwiGLU run reports it 24.8% occupied (28 MiB of weights + a
# 2 MiB input tile + a 1.75 MiB output tile per core). Every number below is that run's.
VMEM_BITS = 1073741824
SWIGLU_WORKING_SET_BITS = 266338304  # 0.248046875 x 128 MiB, the CME's own figure
BASELINE_AREA_MM2 = 172.78


def _area_objective(evidence, area=BASELINE_AREA_MM2):
    return Objective.from_baseline("area", baseline_latency_cycles=evidence.latency_total, baseline_area_mm2=area)


def test_an_over_provisioned_scratchpad_is_offered_a_shrink_priced_in_area(bundle, swiglu_evidence):
    """The dual of Rule 1, on the run that motivated it: a 128 MiB scratchpad a quarter full.

    Sized at bank granularity rather than by halvings: 24.8% of 128 MiB plus 10% of margin is 3 of
    the scratchpad's 8 banks, and a rule that could only offer powers of two would round that up to
    4 and hand back a quarter less silicon than the measurement allows.
    """
    result = offer_operators(
        swiglu_evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(swiglu_evidence)
    )
    offer = _offer(result, "core.memory.shrink", memory="operand_buffer", banks=3)
    assert offer is not None, "a scratchpad measured 24.8% full must be offered a reduction to 3/8"
    assert offer.predicted_delta.unit == "mm2"
    assert offer.predicted_delta.cycles == 0.0, "capacity above the working set buys no cycles, and must not claim any"
    # 172.78 -> 71.86 mm2: the whole point of the operator, and it is arithmetic, not an estimate.
    assert offer.predicted_delta.value == pytest.approx(100.9, abs=0.5)
    assert "266338304" in offer.evidence or "24.8%" in offer.evidence
    # ... and the next size up, so the choice between "as tight as measured" and "one bank of
    # margin" is the proposer's rather than the registry's.
    assert _offer(result, "core.memory.shrink", memory="operand_buffer", banks=4) is not None


def test_no_offered_capacity_is_below_the_measured_working_set(bundle, swiglu_evidence):
    """The one guard that makes a reduction a design rather than a gamble: 2 of 8 banks would leave
    32 MiB against a measured 31.75 MiB working set with no margin at all, and 1 would not fit."""
    result = offer_operators(
        swiglu_evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(swiglu_evidence)
    )
    scratchpad = [
        o for o in result.offered if o.operator_id == "core.memory.shrink" and o.target["memory"] == "operand_buffer"
    ]
    assert scratchpad, "the fixture must offer at least one reduction of the scratchpad"
    for offer in scratchpad:
        assert offer.target["to_bits"] >= SWIGLU_WORKING_SET_BITS * SHRINK_HEADROOM
    assert _offer(result, "core.memory.shrink", memory="operand_buffer", banks=2) is None
    assert _offer(result, "core.memory.shrink", memory="operand_buffer", banks=1) is None


def test_an_unmeasured_memory_is_never_shrunk(bundle, swiglu_evidence):
    """The mirror of `evidence: "none"`. Nothing measured how full HBM gets on this run, and an
    unmeasured memory is not a safe one to cut — it is one nothing is known about."""
    result = offer_operators(
        swiglu_evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(swiglu_evidence)
    )
    assert _offer(result, "core.memory.shrink", memory="hbm3e") is None
    veto = _reduction_veto(result, "core.memory.shrink", memory="hbm3e")
    assert veto is not None and veto.rule == "rule-1-dual"
    assert "nothing measured" in veto.reason


def test_a_full_memory_is_never_shrunk(bundle, swiglu_evidence):
    """The MXU accumulator runs at 100% of its 8192 bits. Halving it is arithmetically illegal."""
    result = offer_operators(
        swiglu_evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(swiglu_evidence)
    )
    assert _offer(result, "core.memory.shrink", memory="accumulator") is None
    veto = _reduction_veto(result, "core.memory.shrink", memory="accumulator")
    assert veto is not None and "100.0% occupied" in veto.reason
    assert "all 8 of its banks" in veto.reason


def test_the_allocator_residency_is_evidence_even_without_a_cme(bundle):
    """An AIE tile has no CME at all, so the ZigZag half of the signal is absent — but the MILP's
    own memory-capacity constraint still measured what it placed. The two are different
    measurements, and a missing CME must not be read as "nothing is known"."""
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([SILU_NO_CME]),
        allocation=allocation_json(
            binding=CORE_BINDING,
            slack=CORE_SLACK,
            # The scratchpad's alias owner is core 0's operand_buffer; the allocator constrains
            # cores 0-11 against the same 128 MiB.
            occupancy=tuple((core_id, SWIGLU_WORKING_SET_BITS, VMEM_BITS) for core_id in range(12)),
        ),
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(evidence))
    offer = _offer(result, "core.memory.shrink", memory="operand_buffer", banks=3)
    assert offer is not None, "the allocator's solved residency is a measurement in its own right"
    assert "allocator residency" in offer.evidence


def test_the_floor_is_the_worst_fused_group_not_the_slowest_one(bundle):
    """Each fused group solves separately and the memory has to hold whichever needs the most.
    Reading only the dominant group's residency would authorise a shrink another group cannot fit."""
    allocation = allocation_json(binding=CORE_BINDING, slack=CORE_SLACK, occupancy=((0, VMEM_BITS // 8, VMEM_BITS),))
    # A second, faster group that needs three quarters of the scratchpad.
    hungry = allocation_json(binding=CORE_BINDING, slack=CORE_SLACK, occupancy=((0, 3 * VMEM_BITS // 4, VMEM_BITS),))
    hungry["groups"][0]["allocation"]["latency"]["total"] = 1
    hungry["groups"][0]["name"] = "group_1"
    allocation["groups"].extend(hungry["groups"])

    evidence = RunEvidence.from_artifacts(progress=progress_json([SILU_NO_CME]), allocation=allocation)
    assert evidence.occupancy_of(0).resident_bits == 3 * VMEM_BITS // 4
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(evidence))
    offer = _offer(result, "core.memory.shrink", memory="operand_buffer")
    # Read off the eighth-full group alone this would be a reduction to 2 of 8 banks. The hungry
    # group needs six, and the capacity has to hold that too.
    assert offer is not None and offer.target["banks"] >= 7
    assert offer.target["to_bits"] >= 3 * VMEM_BITS // 4


def test_a_shrink_carries_every_aliased_view(bundle):
    """The sharp version of the alias coupling: leaving one view at 128 MiB would let the solver keep
    placing tensors the shrunken silicon cannot hold."""
    applied = apply_operator(
        "core.memory.shrink",
        {"cores": [0, 2, 4, 6], "memory": "operand_buffer", "to_bits": VMEM_BITS // 4},
        bundle=bundle,
    )
    sizes = {
        (core_id, name): mem["size"]
        for core_id, core in applied.bundle.cores.items()
        for name, mem in (core.get("memories") or {}).items()
        if name in ("operand_buffer", "vmem")
    }
    assert len(sizes) == 12, "8 operand_buffer views + 4 vmem cores are one scratchpad seen twelve ways"
    assert set(sizes.values()) == {VMEM_BITS // 4}
    assert evaluate_bundle_cost(applied.bundle).total_area_mm2 == pytest.approx(51.7, abs=0.5)


def test_a_stalling_level_is_never_narrowed(bundle, swiglu_evidence):
    """The exact dual of Rule 1's positive direction: a level that is not keeping up with the width
    it already has cannot be given less of it."""
    result = offer_operators(
        swiglu_evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(swiglu_evidence)
    )
    assert _offer(result, "core.memory.narrow", memory="vregs") is None
    veto = _reduction_veto(result, "core.memory.narrow", memory="vregs")
    assert veto is not None and "stalls 1326 cycles" in veto.reason


def test_slack_alone_cannot_choose_a_narrowing_factor(bundle):
    """Slack says the level kept up. It does not say by how much, and a factor needs the how much."""
    node = copy.deepcopy(ELT_MUL)
    node["memory_levels"]["vregs"].update(stall_cycles=0.0, slack_cycles=800.0, per_port={"rw_port_1": -800.0})
    node["memory_levels"]["vregs"]["utilization"] = None
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([node]), allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK)
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(evidence))
    veto = _reduction_veto(result, "core.memory.narrow", memory="vregs")
    assert veto is not None and "no measured port occupancy" in veto.reason


def test_a_level_with_slack_and_spare_occupancy_is_narrowed(bundle):
    """Both halves present: no stall, and a port measured 30% busy over the node's span. Halving the
    width takes it to 0.6 and is offered; quartering takes it to 1.2, past the ceiling, and is not."""
    node = copy.deepcopy(ELT_MUL)
    node["memory_levels"]["vregs"].update(
        stall_cycles=0.0, slack_cycles=800.0, per_port={"rw_port_1": -800.0}, utilization=0.3
    )
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([node]), allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK)
    )
    result = offer_operators(evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(evidence))
    offer = _offer(result, "core.memory.narrow", memory="vregs", divisor=2)
    assert offer is not None
    assert offer.predicted_delta.unit == "mm2" and offer.predicted_delta.value > 0
    assert _offer(result, "core.memory.narrow", memory="vregs", divisor=4) is None
    assert _reduction_veto(result, "core.memory.narrow", divisor=4).reason.endswith("0.9 ceiling")


def test_a_narrowing_never_drops_below_the_declared_minimum_access(bundle):
    """`bandwidth_min` is the smallest access the memory supports; a maximum below it would describe
    hardware that cannot service its own minimum request."""
    applied = apply_operator(
        "core.memory.narrow", {"cores": [8, 9, 10, 11], "memory": "vmem", "to_bandwidth": 262144 // 4}, bundle=bundle
    )
    ports = applied.bundle.cores[8]["memories"]["vmem"]["ports"]
    assert [p["bandwidth_max"] for p in ports] == [262144 // 4] * len(ports)
    assert [p["bandwidth_min"] for p in ports] == [256] * len(ports), "the floor holds"


def test_a_reduction_applied_to_an_already_smaller_design_cannot_grow_it(bundle):
    """A candidate applies its operator to the run's BASELINE bundle, while the offer was computed
    on the bundle the evidence came from — the same design only in the first wave.

    An absolute target is what makes the edit that runs the edit that was priced. It is also applied
    as a cap, so a reduction reaching a design already below its target is a no-op rather than a
    silent growth that no budget guard ever looked at.
    """
    smaller = apply_operator(
        "core.memory.shrink", {"cores": [1, 3, 5, 7], "memory": "vregs", "to_bits": 524288}, bundle=bundle
    ).bundle
    again = apply_operator(
        "core.memory.shrink", {"cores": [1, 3, 5, 7], "memory": "vregs", "to_bits": 1835008}, bundle=smaller
    ).bundle
    assert again.cores[1]["memories"]["vregs"]["size"] == 524288
    narrower = apply_operator(
        "core.memory.narrow", {"cores": [1, 3, 5, 7], "memory": "vregs", "to_bandwidth": 4096}, bundle=bundle
    ).bundle
    widened = apply_operator(
        "core.memory.narrow", {"cores": [1, 3, 5, 7], "memory": "vregs", "to_bandwidth": 32768}, bundle=narrower
    ).bundle
    assert [p["bandwidth_max"] for p in widened.cores[1]["memories"]["vregs"]["ports"]] == [4096, 4096]


def test_a_reduction_that_made_the_mapping_infeasible_is_rejected(swiglu_evidence):
    """The pre-solve floor is arithmetic over the PREVIOUS placement. The placement is re-derived on
    the smaller hardware, so the only proof the cut held is the run that followed it."""
    after = RunEvidence.from_artifacts(
        progress=progress_json([ELT_MUL]),
        allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK),
        infeasibility={"status": "INFEASIBLE", "resources": []},
    )
    reason = post_hoc_check(
        "core.memory.shrink", {"memory": "operand_buffer", "to_bits": 268435456}, swiglu_evidence, after
    )
    assert reason is not None and "INFEASIBLE" in reason
    # ... and a reduction that stayed feasible at the same latency is accepted.
    assert post_hoc_check("core.memory.shrink", {"memory": "operand_buffer"}, swiglu_evidence, swiglu_evidence) is None


def test_a_reduction_that_cost_visible_latency_is_rejected(swiglu_evidence):
    """Feasible but slower. Trading latency for area is what the objective's ceiling bounds; a
    candidate over it is a regression reported as such, not an area win."""
    slower = replace(swiglu_evidence, latency_total=swiglu_evidence.latency_total * 1.5)
    reason = post_hoc_reduction_check(
        "core.memory.shrink",
        {"memory": "operand_buffer", "to_bits": 268435456},
        slower,
        parent_latency_cycles=swiglu_evidence.latency_total,
        objective=_area_objective(swiglu_evidence),
    )
    assert reason is not None and "ceiling" in reason


# ── E1: the objective is a filter, not a ranking ─────────────────────────────────────────────────


def test_a_latency_search_is_never_offered_an_area_saving(bundle, swiglu_evidence):
    """And it is told why, rather than being handed a move it cannot score."""
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING)  # default: latency
    assert result.objective.kind is ObjectiveKind.LATENCY
    assert not [o for o in result.offered if o.predicted_delta.unit == "mm2"]
    veto = _reduction_veto(result, "core.memory.shrink", banks=3)
    assert veto is not None and veto.rule == "objective"


def test_an_area_search_is_never_offered_a_way_to_spend_silicon(bundle, swiglu_evidence):
    result = offer_operators(
        swiglu_evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(swiglu_evidence)
    )
    assert {o.operator_id for o in result.offered} == {"core.memory.shrink"}
    veto = _reduction_veto(result, "core.memory.bandwidth", memory="vregs")
    assert veto is not None and veto.rule == "objective"


def test_efficiency_admits_both_directions(bundle, swiglu_evidence):
    objective = Objective.from_baseline(
        "efficiency",
        baseline_latency_cycles=swiglu_evidence.latency_total,
        baseline_area_mm2=BASELINE_AREA_MM2,
    )
    result = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING, objective=objective)
    units = {o.predicted_delta.unit for o in result.offered}
    assert units == {"cycles", "mm2"}


def test_an_infeasible_run_keeps_every_repair_whatever_the_objective(bundle):
    """There is no objective value to improve on a design that produced no schedule. A search that
    refused to spend area on feasibility would simply never solve."""
    evidence = RunEvidence.from_artifacts(
        progress=progress_json([_mxu_node("Gemm_Left")]),
        allocation=allocation_json(binding=CORE_BINDING, slack=CORE_SLACK),
        infeasibility={
            "status": "INFEASIBLE",
            "resources": [
                {
                    "resource": {"kind": "core", "id": "8", "label": "Core(8)"},
                    "unmet": {
                        "family": "memory_capacity",
                        "demand_value": 1610612736,
                        "bound_value": 1073741824,
                        "gap": 536870912,
                        "unit": "bits",
                        "levers": [],
                    },
                }
            ],
        },
    )
    objective = Objective.from_baseline("area", baseline_latency_cycles=None, baseline_area_mm2=BASELINE_AREA_MM2)
    result = offer_operators(
        evidence,
        bundle=bundle,
        mapping_params=MAPPING,
        objective=objective,
        # Growing a 128 MiB scratchpad busts the default 10% area headroom, and that is a separate
        # guard with a separate reason. This test is about the objective filter, so the budget is
        # widened to let the repair through and leave the objective as the only thing that could
        # refuse it.
        budget=HardwareBudget.from_bundle(bundle, 4.0),
    )
    assert _offer(result, "core.memory.capacity", core=8) is not None
    assert not [v for v in result.vetoed if v.rule == "objective"]


def test_the_serialized_form_declares_the_objective_and_the_scorecard(bundle, swiglu_evidence):
    """A comparison that cannot see the objective silently mixes two different questions."""
    payload = offer_operators(
        swiglu_evidence, bundle=bundle, mapping_params=MAPPING, objective=_area_objective(swiglu_evidence)
    ).as_dict()
    assert payload["objective"]["kind"] == "area"
    assert payload["objective"]["unit"] == "mm2"
    assert payload["objective"]["max_latency_cycles"] == pytest.approx(1392195 * 1.02)
    assert payload["objective"]["max_area_mm2"] == pytest.approx(BASELINE_AREA_MM2)
    assert "trust" in payload["offered"][0] and payload["offered"][0]["trust"] == 1.0
    # An area saving is arithmetic over the cost model, not a solver estimate, so the cycle noise
    # floor says nothing about it and it must not be reported as invisible.
    assert payload["offered"][0]["clears_noise_floor"] is True


def test_a_persistent_over_predictor_is_discounted_not_removed(bundle, swiglu_evidence):
    """The live case: predicted 9,088, achieved -24,178. The operator stays on the menu — the next
    evidence may be the case it is right about — but its prediction is no longer taken at face."""
    scorecard = OperatorScorecard()
    scorecard.record(
        Residual(
            operator_id="system.tiling.intra_core",
            predicted=9088.0,
            achieved=-24178.0,
            unit="cycles",
        )
    )
    payload = offer_operators(swiglu_evidence, bundle=bundle, mapping_params=MAPPING, scorecard=scorecard).as_dict()
    tiling = next(o for o in payload["offered"] if o["operator"] == "system.tiling.intra_core")
    assert tiling["trust"] == pytest.approx(MIN_TRUST)
    assert tiling["discounted_delta"] == pytest.approx(9088.0 * MIN_TRUST)
    untouched = next(o for o in payload["offered"] if o["operator"] == "system.alloc.cores")
    assert untouched["trust"] == 1.0, "untried is not unreliable"
