"""
ComputeAllocator — a fully-modular MILP wrapper for node-to-core allocation
with *all* solver variables and constants exposed for inspection.

Highlights
----------
* Constants collected once and stored under ``self.const`` (see
  :class:`ComputeAllocatorConstants`).
* Every Gurobi variable handle is a public attribute—type-annotated precisely
  (``gp.tupledict`` *vs.* ``gurobipy.Var``).
* Helper methods assert the tupledicts are non-``None`` **before** subscripting,
  satisfying static type checkers.
* Public API remains: :py:meth:`ComputeAllocator.get_optimal_allocations`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import gurobipy as gp
from gurobipy import GRB, Var
from zigzag.datatypes import LayerOperand, MemoryOperand

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.utils import get_core_capacities
from stream.opt.allocation.constraint_optimization.utils import (
    convert_ids,
    get_energies,
    get_latencies,
    invert_ids_list,
)
from stream.utils import CostModelEvaluationLUT
from stream.workload.onnx_workload import ComputationNodeWorkload

# --------------------------------------------------------------------------- #
# Type aliases                                                                #
# --------------------------------------------------------------------------- #
ALLOCATION_T: TypeAlias = list[tuple[int, int, tuple[int, int]]]
ALLOCATION_INTERNAL_T: TypeAlias = list[tuple[int, str, int]]

LatDict = dict[tuple[int, str, int], int]
EnergyDict = dict[tuple[int, str], float]
WeightDict = dict[int, int]
DepDict = dict[tuple[int, int], int]
CapDict = dict[str, float]
GroupDict = dict[int, list[int]]
SplitDict = dict[int, dict[str, dict[int, int]]]


# --------------------------------------------------------------------------- #
# Constants container                                                         #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ComputeAllocatorConstants:
    """Immutable data needed to build the MILP model."""

    lat: LatDict
    energies: EnergyDict
    weights: WeightDict
    deps: DepDict
    capacities: CapDict
    groups: GroupDict
    splits: SplitDict
    node_count: int


# --------------------------------------------------------------------------- #
# Allocator                                                                   #
# --------------------------------------------------------------------------- #
class ComputeAllocator:
    """Constraint-optimization wrapper exposing solver internals."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        cost_lut: CostModelEvaluationLUT,
        *,
        iterations: int = 1,
        gap: float = 0.5,
        time_limit: int = 600,
        latency_attr: str = "latency_total1",
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.cost_lut = cost_lut

        self.iterations = iterations
        self.gap = gap
        self.time_limit = time_limit
        self.latency_attr = latency_attr

        # Gurobi model and main assignment tensor
        self.model: gp.Model | None = None
        self.asgn: gp.tupledict | None = None

        # ---------------- variable handles (precise typing) ------------- #
        self.k_vec: gp.tupledict | None = None
        self.k_splits: gp.tupledict | None = None
        self.core_asgn: gp.tupledict | None = None
        self.lat_id_core: gp.tupledict | None = None
        self.slot_asgn: gp.tupledict | None = None
        self.slot_idx: gp.tupledict | None = None
        self.w_split: gp.tupledict | None = None
        self.w_core: gp.tupledict | None = None
        self.lat_core_slot: gp.tupledict | None = None
        self.lat_slot: gp.tupledict | None = None
        self.lat_iter: Var | None = None
        self.idle_start: gp.tupledict | None = None
        self.idle_end: gp.tupledict | None = None
        self.idle_sum: gp.tupledict | None = None
        self.idle_min: Var | None = None
        self.total_lat: Var | None = None

        # ---------------- sets populated during model build ------------- #
        self.node_ids: list[int] = []
        self.cores: list[str] = []
        self.slots: list[int] = []
        self.p_vals: list[int] = []

        # Constants container (populated in _prepare_constants)
        self.const: ComputeAllocatorConstants | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def get_optimal_allocations(self) -> ALLOCATION_T:
        """Return sorted ``(slot, core_name, node_id)`` tuples."""
        self.const = self._prepare_constants()
        self._build_constraint_model(self.const)
        alloc = self._solve_model(self.const.node_count)
        return invert_ids_list(alloc, self.const.node_count)  # type: ignore

    # ------------------------------------------------------------------ #
    # Prepare constants                                                  #
    # ------------------------------------------------------------------ #
    def _prepare_constants(self) -> ComputeAllocatorConstants:
        core_ids, cores, caps = self._collect_core_data()
        nodes, ids = self._collect_node_data()
        lat, splits = self._calculate_latencies(nodes, core_ids, cores, ids)
        en = self._calculate_energies(nodes, core_ids, ids)
        deps = self._build_dependency_map(nodes, ids)
        groups = self._group_nodes(nodes, ids)
        weights = self._calculate_weights(groups, ids)
        caps = self._relax_caps_if_single_node(nodes, caps)

        return ComputeAllocatorConstants(
            lat=lat,
            energies=en,
            weights=weights,
            deps=deps,
            capacities=caps,
            groups=groups,
            splits=splits,
            node_count=len(nodes),
        )

    # ----------------- low-level helpers -------------------------------- #
    def _collect_core_data(self) -> tuple[list[int], list, CapDict]:
        core_ids = sorted(c.id for c in self.accelerator.cores.node_list if c.id != self.accelerator.offchip_core_id)
        cores = [self.accelerator.get_core(cid) for cid in core_ids]
        caps = get_core_capacities(self.accelerator, MemoryOperand("I2"), core_ids)
        return core_ids, cores, caps

    def _collect_node_data(self) -> tuple[list, dict]:
        nodes = sorted(self.workload.node_list)
        return nodes, convert_ids(nodes)

    def _calculate_latencies(
        self,
        nodes,
        core_ids: list[int],
        cores,
        ids: dict,
    ) -> tuple[LatDict, SplitDict]:
        raw_lat, raw_split = get_latencies(
            nodes,
            core_ids,
            self.accelerator,
            self.cost_lut,
            impossible_lat=0,
            latency_attr=self.latency_attr,
        )
        lat: LatDict = {(ids[n], f"Core {c.id}", k): v for (n, c, k), v in raw_lat.items()}
        split: SplitDict = {
            ids[n]: {f"Core {c.id}": {k: raw_split[n][c][k] for k in raw_split[n][c]} for c in cores} for n in nodes
        }
        return lat, split

    def _calculate_energies(
        self,
        nodes,
        core_ids: list[int],
        ids: dict,
    ) -> EnergyDict:
        return get_energies(nodes, core_ids, self.accelerator, self.cost_lut, 0, ids)

    def _build_dependency_map(self, nodes, ids: dict) -> DepDict:
        out_op = LayerOperand("O")
        return {
            (ids[p], ids[c]): cast(int, p.operand_size_bit[out_op])
            for p, c in self.workload.edges()
            if p in nodes and c in nodes
        }

    @staticmethod
    def _group_nodes(nodes, ids: dict) -> GroupDict:
        groups: GroupDict = {}
        for n in nodes:
            groups.setdefault(n.id, []).append(ids[n])
        return groups

    @staticmethod
    def _calculate_weights(groups: GroupDict, ids: dict) -> WeightDict:
        weights: WeightDict = {}
        for grp in groups.values():
            ref_sz = None
            for idx, nid in enumerate(grp):
                node = next(k for k, v in ids.items() if v == nid)
                i2_ops = [op for op in node.constant_operands if node.memory_operand_links[op] == MemoryOperand("I2")]
                bits = sum(node.operand_size_bit[op] for op in i2_ops)
                if idx == 0:
                    ref_sz = bits
                    weights[nid] = bits
                else:
                    assert bits == ref_sz, "Mismatch in grouped weight sizes"
                    weights[nid] = 0
        return weights

    @staticmethod
    def _relax_caps_if_single_node(
        nodes,
        caps: CapDict,
    ) -> CapDict:
        if len(nodes) == 1:
            return {c: 1e11 for c in caps}
        return caps

    # ------------------------------------------------------------------ #
    # Model construction                                                 #
    # ------------------------------------------------------------------ #
    def _build_constraint_model(self, const: ComputeAllocatorConstants) -> None:
        self._create_basic_sets(const.lat, const.capacities)
        self._create_variables()
        self._add_k_split_constraints()
        self._add_core_assignment_constraints(const.splits)
        self._add_slot_assignment_constraints()
        self._add_group_constraints(const.groups)
        self._add_dependency_constraints(const.deps)
        self._add_weight_constraints(const.weights, const.capacities)
        self._add_latency_constraints(const.lat)
        self._add_overlap_constraints()
        self._set_objective()

    # -------------------- basic sets ----------------------------------- #
    def _create_basic_sets(self, lat: LatDict, caps: CapDict) -> None:
        node_core_k = list(lat.keys())
        self.node_ids = sorted({n for n, _, _ in node_core_k})
        self.cores = sorted(caps)
        self.p_vals = list(range(1, len(self.cores) + 1))
        self.slots = list(range(len(self.node_ids)))

        self.model = gp.Model("compute_alloc")
        self.model.Params.OutputFlag = 0
        self.model.Params.TimeLimit = self.time_limit
        self.model.Params.Threads = 1
        self.model.Params.PoolGap = self.gap

    # -------------------- variable creation ---------------------------- #
    def _create_variables(self) -> None:
        assert self.model is not None
        m = self.model

        # tupledicts
        self.k_vec = m.addVars(self.node_ids, self.p_vals, vtype=GRB.BINARY, name="k_vec")
        self.k_splits = m.addVars(self.node_ids, vtype=GRB.INTEGER, name="k_splits")
        self.core_asgn = m.addVars(self.cores, self.node_ids, vtype=GRB.BINARY, name="core_asgn")
        self.lat_id_core = m.addVars(self.node_ids, self.cores, vtype=GRB.INTEGER, name="lat_id_core")
        self.slot_asgn = m.addVars(self.slots, self.node_ids, vtype=GRB.BINARY, name="slot_asgn")
        self.asgn = m.addVars(self.cores, self.slots, self.node_ids, vtype=GRB.BINARY, name="asgn")
        self.slot_idx = m.addVars(self.node_ids, vtype=GRB.INTEGER, name="slot_idx")
        self.w_split = m.addVars(self.node_ids, vtype=GRB.INTEGER, name="w_split")
        self.w_core = m.addVars(self.cores, vtype=GRB.INTEGER, name="w_core")
        self.lat_core_slot = m.addVars(self.cores, self.slots, vtype=GRB.INTEGER, name="lat_core_slot")
        self.lat_slot = m.addVars(self.slots, vtype=GRB.INTEGER, name="lat_slot")
        self.idle_start = m.addVars(self.cores, self.slots, vtype=GRB.BINARY, name="idle_start")
        self.idle_end = m.addVars(self.cores, self.slots, vtype=GRB.BINARY, name="idle_end")
        self.idle_sum = m.addVars(self.cores, vtype=GRB.INTEGER, name="idle_sum")

        # Scalars
        self.lat_iter = m.addVar(vtype=GRB.INTEGER, name="lat_iter")
        self.idle_min = m.addVar(vtype=GRB.INTEGER, name="idle_min")
        self.total_lat = m.addVar(vtype=GRB.INTEGER, name="total_lat")

    # -------------------- constraint layers ---------------------------- #
    def _add_k_split_constraints(self) -> None:
        assert self.model and self.k_vec and self.k_splits
        m = self.model
        m.addConstrs(self.k_vec.sum(n, "*") == 1 for n in self.node_ids)
        m.addConstrs(self.k_splits[n] == gp.quicksum(self.k_vec[n, k] * k for k in self.p_vals) for n in self.node_ids)

    def _add_core_assignment_constraints(self, splits: SplitDict) -> None:
        assert self.model and self.core_asgn and self.k_splits and self.k_vec
        m = self.model
        m.addConstrs(self.core_asgn.sum("*", n) == self.k_splits[n] for n in self.node_ids)
        for n in self.node_ids:
            m.addConstrs(
                self.core_asgn[c, n] <= gp.quicksum(splits[n][c][k] * self.k_vec[n, k] for k in self.p_vals)
                for c in self.cores
            )

    def _add_slot_assignment_constraints(self) -> None:
        assert self.model and self.slot_asgn and self.asgn and self.core_asgn and self.k_splits
        m = self.model
        m.addConstrs(self.slot_asgn.sum("*", n) == 1 for n in self.node_ids)
        m.addConstrs(self.core_asgn[c, n] == self.asgn.sum(c, "*", n) for c in self.cores for n in self.node_ids)
        for n in self.node_ids:
            for s in self.slots:
                m.addConstr(self.slot_asgn[s, n] * self.asgn.sum("*", s, n) == self.slot_asgn[s, n] * self.k_splits[n])
        m.addConstrs(self.asgn.sum(c, s, "*") <= 1 for c in self.cores for s in self.slots)

    def _add_group_constraints(self, groups: GroupDict) -> None:
        assert self.model and self.asgn
        m = self.model
        for grp in groups.values():
            for n_i, n_j in zip(grp, grp[1:], strict=False):
                m.addConstrs(self.asgn.sum(c, "*", n_i) == self.asgn.sum(c, "*", n_j) for c in self.cores)

    def _add_dependency_constraints(self, deps: DepDict) -> None:
        assert self.model and self.slot_idx and self.slot_asgn
        m = self.model
        m.addConstrs(
            self.slot_idx[n] == gp.quicksum(s * self.slot_asgn[s, n] for s in self.slots) for n in self.node_ids
        )
        for p, c in deps:
            m.addConstr(self.slot_idx[c] >= self.slot_idx[p] + 1)

    def _add_weight_constraints(self, weights: WeightDict, caps: CapDict) -> None:
        assert self.model and self.w_split and self.k_splits and self.w_core and self.asgn
        m = self.model
        m.addConstrs(self.w_split[n] * self.k_splits[n] >= weights[n] for n in self.node_ids)
        m.addConstrs(
            self.w_core[c] >= gp.quicksum(self.w_split[n] * self.asgn.sum(c, "*", n) for n in self.node_ids)
            for c in self.cores
        )
        m.addConstrs(self.w_core[c] <= caps[c] for c in self.cores)

    def _add_latency_constraints(self, lat: LatDict) -> None:
        assert (
            self.model
            and self.lat_id_core
            and self.k_vec
            and self.lat_core_slot
            and self.asgn
            and self.lat_slot
            and self.lat_iter
        )
        m = self.model
        _, lat_param = gp.multidict(lat)
        for n in self.node_ids:
            m.addConstrs(
                self.lat_id_core[n, c] == gp.quicksum(self.k_vec[n, k] * lat_param[n, c, k] for k in self.p_vals)
                for c in self.cores
            )
        m.addConstrs(
            self.lat_core_slot[c, s] == gp.quicksum(self.lat_id_core[n, c] * self.asgn[c, s, n] for n in self.node_ids)
            for c in self.cores
            for s in self.slots
        )
        m.addConstrs(
            self.lat_slot[s] == gp.max_(self.lat_core_slot[c, s] for c in self.cores)  # type: ignore[arg-type]
            for s in self.slots
        )
        m.addConstr(self.lat_iter == gp.quicksum(self.lat_slot))

    def _add_overlap_constraints(self) -> None:
        assert (
            self.model
            and self.idle_start
            and self.idle_end
            and self.idle_sum
            and self.idle_min
            and self.asgn
            and self.lat_slot
            and self.total_lat
        ), "Model and required variables must be initialized before adding overlap constraints"
        m = self.model
        incl = m.addVars(self.cores, self.slots, vtype=GRB.INTEGER, name="incl")
        excl = m.addVars(self.cores, self.slots, vtype=GRB.INTEGER, name="excl")

        m.addConstrs(
            incl[c, s] == gp.quicksum(self.asgn.sum(c, t, "*") for t in range(s + 1))
            for c in self.cores
            for s in self.slots
        )
        m.addConstrs(
            excl[c, s] == gp.quicksum(self.asgn.sum(c, t, "*") for t in range(s))
            for c in self.cores
            for s in self.slots
        )

        eps = 1e-4
        big_m = len(self.node_ids) + eps
        tot_nodes_core = m.addVars(self.cores, vtype=GRB.INTEGER, name="tot_nodes_core")
        m.addConstrs(tot_nodes_core[c] == self.asgn.sum(c, "*", "*") for c in self.cores)

        m.addConstrs(
            1 >= incl[c, s] + eps - big_m * (1 - self.idle_start[c, s]) for c in self.cores for s in self.slots
        )
        m.addConstrs(1 <= incl[c, s] + big_m * self.idle_start[c, s] for c in self.cores for s in self.slots)
        m.addConstrs(
            excl[c, s] >= tot_nodes_core[c] - 1 + eps - big_m * (1 - self.idle_end[c, s])
            for c in self.cores
            for s in self.slots
        )
        m.addConstrs(
            excl[c, s] <= tot_nodes_core[c] - 1 + big_m * self.idle_end[c, s] for c in self.cores for s in self.slots
        )
        m.addConstrs(
            self.idle_sum[c]
            == gp.quicksum((self.idle_start[c, s] + self.idle_end[c, s]) * self.lat_slot[s] for s in self.slots)
            for c in self.cores
        )
        m.addConstr(self.idle_min == gp.min_(self.idle_sum))
        assert self.lat_iter is not None, "lat_iter must be set before this constraint"
        m.addConstr(self.total_lat == self.iterations * self.lat_iter - (self.iterations - 1) * self.idle_min)

    def _set_objective(self) -> None:
        assert self.model and self.total_lat
        self.model.setObjective(self.total_lat, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # Solve & extract                                                    #
    # ------------------------------------------------------------------ #
    def _solve_model(self, node_cnt: int) -> ALLOCATION_INTERNAL_T:
        assert self.model and self.asgn
        self.model.optimize()

        if self.model.SolCount == 0:
            self.model.computeIIS()
            self.model.write("infeasible.ilp")
            raise ValueError("No feasible solution; see infeasible.ilp")
        if self.model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
            raise RuntimeError(f"Solver status {self.model.Status}")

        alloc: ALLOCATION_INTERNAL_T = [
            (slot, core, nid) for (core, slot, nid), var in self.asgn.items() if round(var.X) == 1
        ]
        return sorted(alloc)


# --------------------------------------------------------------------------- #
# Functional façade                                                           #
# --------------------------------------------------------------------------- #
def get_optimal_allocations(
    workload: ComputationNodeWorkload,
    accelerator: Accelerator,
    cost_lut: CostModelEvaluationLUT,
    *,
    iterations: int = 1,
    gap: float = 0.5,
    time_limit: int = 600,
    latency_attr: str = "latency_total1",
) -> ALLOCATION_T:
    """Backwards-compatible helper preserving the original functional API."""
    return ComputeAllocator(
        workload,
        accelerator,
        cost_lut,
        iterations=iterations,
        gap=gap,
        time_limit=time_limit,
        latency_attr=latency_attr,
    ).get_optimal_allocations()
