"""
ComputeAllocator — a fully-modular MILP wrapper for node-to-core allocation
with *all* solver variables and constants exposed for inspection.

Highlights
----------
* Constants collected once and stored under ``self.const`` (see
  :class:`ComputeAllocatorConstants`).
* Every solver variable handle is a public attribute—type-annotated precisely
  (plain ``dict[tuple, SolverVar]``).
* Helper methods assert the variable dicts are non-``None`` **before**
  subscripting, satisfying static type checkers.
* Public API remains: :py:meth:`ComputeAllocator.get_optimal_allocations`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeAlias, cast

from zigzag.datatypes import LayerOperand, MemoryOperand

from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.config import ConstraintOptStageConfig
from stream.opt.allocation.constraint_optimization.context import ConstraintContext, build_constraint_context
from stream.opt.allocation.constraint_optimization.utils import (
    convert_ids,
    get_energies,
    get_latencies,
    invert_ids_list,
)
from stream.opt.solver import (
    SolverBackend,
    SolverModel,
    SolverParams,
    SolverVar,
    SolverVarType,
    create_solver,
)
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Type aliases                                                                #
# --------------------------------------------------------------------------- #
ALLOCATION_T: TypeAlias = list[tuple[int, int, tuple[int, int]]]
ALLOCATION_INTERNAL_T: TypeAlias = list[tuple[int, Core, int]]

LatDict = dict[tuple[int, Core, int], int]
EnergyDict = dict[tuple[int, Core], float]
WeightDict = dict[int, int]
DepDict = dict[tuple[int, int], int]
CapDict = dict[Core, float]
GroupDict = dict[tuple[int, int], list[int]]  # key = (node.id, node.group)
SplitDict = dict[int, dict[Core, dict[int, int]]]


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
    cores: list[Core]


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
        workload: Workload,
        accelerator: Accelerator,
        cost_lut: CoreCostLUT,
        context: ConstraintContext,
        *,
        iterations: int = 1,
        backend: str = "ORTOOLS_GSCIP",
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.cost_lut = cost_lut
        self.context = context
        self.iterations = iterations
        self.backend_str = backend

        # Solver model and main assignment tensor
        self.model: SolverModel | None = None
        self.asgn: dict[tuple[Core, int, int], SolverVar] | None = None

        # ---------------- variable handles (precise typing) ------------- #
        self.k_vec: dict[tuple[int, int], SolverVar] | None = None
        self.k_splits: dict[int, SolverVar] | None = None
        self.core_asgn: dict[tuple[Core, int], SolverVar] | None = None
        self.lat_id_core: dict[tuple[int, Core], SolverVar] | None = None
        self.slot_asgn: dict[tuple[int, int], SolverVar] | None = None
        self.slot_idx: dict[int, SolverVar] | None = None
        self.w_split: dict[int, SolverVar] | None = None
        self.w_core: dict[Core, SolverVar] | None = None
        self.lat_core_slot: dict[tuple[Core, int], SolverVar] | None = None
        self.lat_slot: dict[int, SolverVar] | None = None
        self.lat_iter: SolverVar | None = None
        self.idle_start: dict[tuple[Core, int], SolverVar] | None = None
        self.idle_end: dict[tuple[Core, int], SolverVar] | None = None
        self.idle_sum: dict[Core, SolverVar] | None = None
        self.idle_min: SolverVar | None = None
        self.total_lat: SolverVar | None = None

        # ---------------- sets populated during model build ------------- #
        self.node_ids: list[int] = []
        self.cores: list[Core] = []
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
        cores, caps = self._collect_core_data()
        nodes, ids = self._collect_node_data()
        lat, splits = self._calculate_latencies(nodes, cores, ids)
        en = self._calculate_energies(nodes, cores, ids)
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
            cores=cores,
        )

    # ----------------- low-level helpers -------------------------------- #
    def _collect_core_data(self) -> tuple[list[Core], CapDict]:
        cores = sorted(self.context.compute_cores, key=lambda c: c.id)
        if not cores:
            raise ValueError("No eligible compute cores found for ComputeAllocator.")
        caps = {core: self.context.capacities[core] for core in cores}
        return cores, caps

    def _collect_node_data(self) -> tuple[list, dict]:
        nodes = sorted(self.workload.node_list, key=lambda n: n.id)
        return nodes, convert_ids(nodes)

    def _calculate_latencies(
        self,
        nodes,
        cores: list[Core],
        ids: dict,
    ) -> tuple[LatDict, SplitDict]:
        raw_lat, raw_split = get_latencies(
            nodes,
            [c.id for c in cores],
            self.accelerator,
            self.cost_lut,
            impossible_lat=0,
        )
        lat: LatDict = {(ids[n], c, k): v for (n, c, k), v in raw_lat.items()}
        split: SplitDict = {ids[n]: {c: {k: raw_split[n][c][k] for k in raw_split[n][c]} for c in cores} for n in nodes}
        return lat, split

    def _calculate_energies(
        self,
        nodes,
        cores: list[Core],
        ids: dict,
    ) -> EnergyDict:
        return get_energies(nodes, [c.id for c in cores], self.accelerator, self.cost_lut, 0, ids)

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
            key = (n.id, n.group)
            groups.setdefault(key, []).append(ids[n])
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
        self.cores = sorted(caps, key=lambda c: c.id)
        self.p_vals = list(range(1, len(self.cores) + 1))
        self.slots = list(range(len(self.node_ids)))

        self.model = create_solver(SolverBackend[self.backend_str], "compute_alloc")
        self.model.set_param(SolverParams.VERBOSITY, 1)
        self.model.set_param(SolverParams.TIME_LIMIT, self.compute_cfg.time_limit)
        self.model.set_param(SolverParams.THREADS, 0)
        self.model.set_param(SolverParams.POOL_GAP, self.compute_cfg.gap)

    # -------------------- variable creation ---------------------------- #
    def _create_variables(self) -> None:
        assert self.model is not None
        m = self.model

        # Replace tupledict creation with plain dict loops
        self.k_vec = {
            (n, k): m.add_var(vtype=SolverVarType.BINARY, name=f"k_vec[{n},{k}]")
            for n in self.node_ids
            for k in self.p_vals
        }
        self.k_splits = {n: m.add_var(vtype=SolverVarType.INTEGER, name=f"k_splits[{n}]") for n in self.node_ids}
        self.core_asgn = {
            (c, n): m.add_var(vtype=SolverVarType.BINARY, name=f"core_asgn[{c.id},{n}]")
            for c in self.cores
            for n in self.node_ids
        }
        self.lat_id_core = {
            (n, c): m.add_var(vtype=SolverVarType.INTEGER, name=f"lat_id_core[{n},{c.id}]")
            for n in self.node_ids
            for c in self.cores
        }
        self.slot_asgn = {
            (s, n): m.add_var(vtype=SolverVarType.BINARY, name=f"slot_asgn[{s},{n}]")
            for s in self.slots
            for n in self.node_ids
        }
        self.asgn = {
            (c, s, n): m.add_var(vtype=SolverVarType.BINARY, name=f"asgn[{c.id},{s},{n}]")
            for c in self.cores
            for s in self.slots
            for n in self.node_ids
        }
        self.slot_idx = {n: m.add_var(vtype=SolverVarType.INTEGER, name=f"slot_idx[{n}]") for n in self.node_ids}
        self.w_split = {n: m.add_var(vtype=SolverVarType.INTEGER, name=f"w_split[{n}]") for n in self.node_ids}
        self.w_core = {c: m.add_var(vtype=SolverVarType.INTEGER, name=f"w_core[{c.id}]") for c in self.cores}
        self.lat_core_slot = {
            (c, s): m.add_var(vtype=SolverVarType.INTEGER, name=f"lat_core_slot[{c.id},{s}]")
            for c in self.cores
            for s in self.slots
        }
        self.lat_slot = {s: m.add_var(vtype=SolverVarType.INTEGER, name=f"lat_slot[{s}]") for s in self.slots}
        self.idle_start = {
            (c, s): m.add_var(vtype=SolverVarType.BINARY, name=f"idle_start[{c.id},{s}]")
            for c in self.cores
            for s in self.slots
        }
        self.idle_end = {
            (c, s): m.add_var(vtype=SolverVarType.BINARY, name=f"idle_end[{c.id},{s}]")
            for c in self.cores
            for s in self.slots
        }
        self.idle_sum = {c: m.add_var(vtype=SolverVarType.INTEGER, name=f"idle_sum[{c.id}]") for c in self.cores}

        # Scalars
        self.lat_iter = m.add_var(vtype=SolverVarType.INTEGER, name="lat_iter")
        self.idle_min = m.add_var(vtype=SolverVarType.INTEGER, name="idle_min")
        self.total_lat = m.add_var(vtype=SolverVarType.INTEGER, name="total_lat")

    # -------------------- constraint layers ---------------------------- #
    def _add_k_split_constraints(self) -> None:
        assert self.model and self.k_vec and self.k_splits
        m = self.model
        for n in self.node_ids:
            m.add_constr(
                m.quicksum(self.k_vec[(n, k)]._raw for k in self.p_vals) == 1,
                name=f"k_vec_one_hot_{n}",
            )
        for n in self.node_ids:
            m.add_constr(
                self.k_splits[n]._raw == m.quicksum(self.k_vec[(n, k)]._raw * k for k in self.p_vals)._raw,
                name=f"k_splits_def_{n}",
            )

    def _add_core_assignment_constraints(self, splits: SplitDict) -> None:
        assert self.model and self.core_asgn and self.k_splits and self.k_vec
        m = self.model
        for n in self.node_ids:
            m.add_constr(
                m.quicksum(self.core_asgn[(c, n)]._raw for c in self.cores) == self.k_splits[n]._raw,
                name=f"core_asgn_sum_{n}",
            )
        for n in self.node_ids:
            for c in self.cores:
                if any(splits[n][c][p] > 0 for p in self.p_vals):
                    m.add_constr(
                        self.core_asgn[(c, n)]._raw
                        <= m.quicksum(splits[n][c][k] * self.k_vec[(n, k)]._raw for k in self.p_vals)._raw,
                        name=f"core_asgn_split_{c.id}_{n}",
                    )
                else:
                    # Explicitly zero the assignment
                    m.add_constr(self.core_asgn[(c, n)]._raw == 0, name=f"core_asgn_zero_{c.id}_{n}")

    def _add_slot_assignment_constraints(self) -> None:
        assert self.model and self.slot_asgn and self.asgn and self.core_asgn and self.k_splits
        m = self.model
        for n in self.node_ids:
            m.add_constr(
                m.quicksum(self.slot_asgn[(s, n)]._raw for s in self.slots) == 1,
                name=f"slot_asgn_one_hot_{n}",
            )
        for c in self.cores:
            for n in self.node_ids:
                m.add_constr(
                    self.core_asgn[(c, n)]._raw == m.quicksum(self.asgn[(c, s, n)]._raw for s in self.slots)._raw,
                    name=f"core_asgn_asgn_{c.id}_{n}",
                )
        for n in self.node_ids:
            for s in self.slots:
                m.add_constr(
                    self.slot_asgn[(s, n)]._raw * m.quicksum(self.asgn[(c, s, n)]._raw for c in self.cores)._raw
                    == self.slot_asgn[(s, n)]._raw * self.k_splits[n]._raw,
                    name=f"slot_asgn_ksplits_{s}_{n}",
                )
        for c in self.cores:
            for s in self.slots:
                m.add_constr(
                    m.quicksum(self.asgn[(c, s, n)]._raw for n in self.node_ids) <= 1,
                    name=f"asgn_at_most_one_{c.id}_{s}",
                )

    def _add_group_constraints(self, groups: GroupDict) -> None:
        assert self.model and self.asgn
        m = self.model
        for grp in groups.values():
            for n_i, n_j in zip(grp, grp[1:], strict=False):
                for c in self.cores:
                    m.add_constr(
                        m.quicksum(self.asgn[(c, s, n_i)]._raw for s in self.slots)
                        == m.quicksum(self.asgn[(c, s, n_j)]._raw for s in self.slots),
                        name=f"group_{n_i}_{n_j}_{c.id}",
                    )

    def _add_dependency_constraints(self, deps: DepDict) -> None:
        assert self.model and self.slot_idx and self.slot_asgn
        m = self.model
        for n in self.node_ids:
            m.add_constr(
                self.slot_idx[n]._raw == m.quicksum(s * self.slot_asgn[(s, n)]._raw for s in self.slots)._raw,
                name=f"slot_idx_def_{n}",
            )
        for p, c in deps:
            m.add_constr(self.slot_idx[c]._raw >= self.slot_idx[p]._raw + 1, name=f"dep_{p}_{c}")

    def _add_weight_constraints(self, weights: WeightDict, caps: CapDict) -> None:
        assert self.model and self.w_split and self.k_splits and self.w_core and self.asgn
        m = self.model
        for n in self.node_ids:
            m.add_constr(
                self.w_split[n]._raw * self.k_splits[n]._raw >= weights[n],
                name=f"w_split_{n}",
            )
        for c in self.cores:
            m.add_constr(
                self.w_core[c]._raw
                >= m.quicksum(
                    self.w_split[n]._raw * m.quicksum(self.asgn[(c, s, n)]._raw for s in self.slots)._raw
                    for n in self.node_ids
                )._raw,
                name=f"w_core_{c.id}",
            )
            m.add_constr(self.w_core[c]._raw <= caps[c], name=f"w_cap_{c.id}")

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
        # Remove gp.multidict: lat is already a dict with (n, c, k) tuple keys
        for n in self.node_ids:
            for c in self.cores:
                m.add_constr(
                    self.lat_id_core[(n, c)]._raw
                    == m.quicksum(self.k_vec[(n, k)]._raw * lat[(n, c, k)] for k in self.p_vals)._raw,
                    name=f"lat_id_{n}_{c.id}",
                )
        for c in self.cores:
            for s in self.slots:
                m.add_constr(
                    self.lat_core_slot[(c, s)]._raw
                    == m.quicksum(
                        self.lat_id_core[(n, c)]._raw * self.asgn[(c, s, n)]._raw for n in self.node_ids
                    )._raw,
                    name=f"lat_core_slot_{c.id}_{s}",
                )
        # lat_slot[s] = max(lat_core_slot[c, s] for c in cores)
        # Since we MINIMIZE lat_iter (which sums lat_slot), the objective forces lat_slot down.
        # Therefore: lat_slot[s] >= lat_core_slot[c, s] for all c  is sufficient.
        for s in self.slots:
            for c in self.cores:
                m.add_constr(
                    self.lat_slot[s]._raw >= self.lat_core_slot[(c, s)]._raw,
                    name=f"lat_slot_max_{s}_{c.id}",
                )
        m.add_constr(
            self.lat_iter._raw == m.quicksum(self.lat_slot[s]._raw for s in self.slots)._raw,
            name="lat_iter_def",
        )

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
        incl: dict[tuple[Core, int], SolverVar] = {
            (c, s): m.add_var(vtype=SolverVarType.INTEGER, name=f"incl[{c.id},{s}]")
            for c in self.cores
            for s in self.slots
        }
        excl: dict[tuple[Core, int], SolverVar] = {
            (c, s): m.add_var(vtype=SolverVarType.INTEGER, name=f"excl[{c.id},{s}]")
            for c in self.cores
            for s in self.slots
        }

        for c in self.cores:
            for s in self.slots:
                m.add_constr(
                    incl[(c, s)]._raw
                    == m.quicksum(
                        m.quicksum(self.asgn[(c, t, n)]._raw for n in self.node_ids)._raw for t in range(s + 1)
                    )._raw,
                    name=f"incl_{c.id}_{s}",
                )
        for c in self.cores:
            for s in self.slots:
                m.add_constr(
                    excl[(c, s)]._raw
                    == m.quicksum(
                        m.quicksum(self.asgn[(c, t, n)]._raw for n in self.node_ids)._raw for t in range(s)
                    )._raw,
                    name=f"excl_{c.id}_{s}",
                )

        eps = 1e-4
        big_m = len(self.node_ids) + eps
        tot_nodes_core: dict[Core, SolverVar] = {
            c: m.add_var(vtype=SolverVarType.INTEGER, name=f"tot_nodes_core[{c.id}]") for c in self.cores
        }
        for c in self.cores:
            m.add_constr(
                tot_nodes_core[c]._raw
                == m.quicksum(self.asgn[(c, s, n)]._raw for s in self.slots for n in self.node_ids)._raw,
                name=f"tot_nodes_core_{c.id}",
            )

        for c in self.cores:
            for s in self.slots:
                m.add_constr(
                    1 >= incl[(c, s)]._raw + eps - big_m * (1 - self.idle_start[(c, s)]._raw),
                    name=f"idle_start_ub_{c.id}_{s}",
                )
                m.add_constr(
                    1 <= incl[(c, s)]._raw + big_m * self.idle_start[(c, s)]._raw,
                    name=f"idle_start_lb_{c.id}_{s}",
                )
                m.add_constr(
                    excl[(c, s)]._raw >= tot_nodes_core[c]._raw - 1 + eps - big_m * (1 - self.idle_end[(c, s)]._raw),
                    name=f"idle_end_ub_{c.id}_{s}",
                )
                m.add_constr(
                    excl[(c, s)]._raw <= tot_nodes_core[c]._raw - 1 + big_m * self.idle_end[(c, s)]._raw,
                    name=f"idle_end_lb_{c.id}_{s}",
                )
        for c in self.cores:
            m.add_constr(
                self.idle_sum[c]._raw
                == m.quicksum(
                    (self.idle_start[(c, s)]._raw + self.idle_end[(c, s)]._raw) * self.lat_slot[s]._raw
                    for s in self.slots
                )._raw,
                name=f"idle_sum_{c.id}",
            )

        # idle_min = min(idle_sum[c] for c in cores)
        # Use big-M binary selector pattern (research Pattern 4):
        #   idle_min <= idle_sum[c] for all c  (upper-bound from below)
        #   idle_min >= idle_sum[c] - M*(1-b_c) for all c, with sum(b_c)==1
        big_m_min = len(self.node_ids) * max(1, len(self.slots))  # safe upper bound on any idle_sum
        b_min: dict[Core, SolverVar] = {
            c: m.add_var(vtype=SolverVarType.BINARY, name=f"b_min[{c.id}]") for c in self.cores
        }
        for c in self.cores:
            m.add_constr(self.idle_min._raw <= self.idle_sum[c]._raw, name=f"idle_min_ub_{c.id}")
            m.add_constr(
                self.idle_min._raw >= self.idle_sum[c]._raw - big_m_min * (1 - b_min[c]._raw),
                name=f"idle_min_lb_{c.id}",
            )
        m.add_constr(
            m.quicksum(b_min[c]._raw for c in self.cores) == 1,
            name="idle_min_selector_one_hot",
        )
        assert self.lat_iter is not None, "lat_iter must be set before this constraint"
        m.add_constr(
            self.total_lat._raw == self.iterations * self.lat_iter._raw - (self.iterations - 1) * self.idle_min._raw,
            name="total_lat_def",
        )

    def _set_objective(self) -> None:
        assert self.model and self.total_lat
        self.model.set_objective(self.total_lat._raw, sense="minimize")

    # ------------------------------------------------------------------ #
    # Solve & extract                                                    #
    # ------------------------------------------------------------------ #
    def _solve_model(self, node_cnt: int) -> ALLOCATION_INTERNAL_T:
        assert self.model and self.asgn
        self.model.optimize()

        if self.model.get_sol_count() == 0:
            self.model.compute_iis()
            self.model.write("infeasible.ilp")
            raise ValueError("No feasible solution; see infeasible.ilp")
        if self.model.get_status() not in {"OPTIMAL", "TIME_LIMIT"}:
            raise RuntimeError(f"Solver status {self.model.get_status()}")

        alloc: ALLOCATION_INTERNAL_T = [
            (slot, core, nid) for (core, slot, nid), var in self.asgn.items() if round(var.X) == 1
        ]
        return sorted(alloc, key=lambda t: (t[0], getattr(t[1], "id", t[1]), t[2]))


# --------------------------------------------------------------------------- #
# Functional façade                                                           #
# --------------------------------------------------------------------------- #
def get_optimal_allocations(
    workload: Workload,
    accelerator: Accelerator,
    cost_lut: CoreCostLUT,
    *,
    context: ConstraintContext | None = None,
    stage_config: ConstraintOptStageConfig | None = None,
    iterations: int = 1,
    backend: str = "ORTOOLS_GSCIP",
) -> ALLOCATION_T:
    """Backwards-compatible helper preserving the original functional API."""
    if context is None:
        logger.warning(
            "get_optimal_allocations called without explicit context. "
            "Building defaults; please pass ConstraintOptStageConfig explicitly."
        )
        stage_cfg = stage_config or ConstraintOptStageConfig()
        context = context or build_constraint_context(accelerator, stage_cfg)

    return ComputeAllocator(
        workload,
        accelerator,
        cost_lut,
        context,
        iterations=iterations,
        backend=backend,
    ).get_optimal_allocations()
