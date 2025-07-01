# transfer_allocator.py
from itertools import chain

import gurobipy as gp
from gurobipy import GRB

from stream.opt.allocation.constraint_optimization.timeslot_allocation import TimeSlotAllocation, _resource_key
from stream.workload.steady_state_transfer import SteadyStateTransfer
from stream.workload.steady_state_workload import SteadyStateWorkload


class TransferAllocator:
    """
    Latency-aware routing optimiser.

    *  **Slots** and **resources** of *all* nodes are read from the given
       `TimeSlotAllocation` (`tsa`) – no slot variables any more.
    *  **Decision variables**: one-hot path choice *per transfer-node*.
    """

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        workload: SteadyStateWorkload,
        tsa: TimeSlotAllocation,
        *,
        iterations: int = 1,
    ):
        self.workload = workload
        self.tsa = tsa
        self.iter = iterations

        # Helpful maps ---------------------------------------------------
        self.slot_of = {n: tsa.get_timeslot_of_node(n) for n in tsa.nodes}
        self.max_slot = tsa.slot_max

        # quick topological sanity-check
        for u, v in workload.get_edges():
            if self.slot_of[v] <= self.slot_of[u]:
                raise ValueError(
                    f"Schedule violates precedence: {u.node_name} (slot {self.slot_of[u]}) "
                    f"→ {v.node_name} (slot {self.slot_of[v]})"
                )

        # Node groups ----------------------------------------------------
        self.comp_nodes = workload.computation_nodes
        self.tensor_nodes = workload.tensor_nodes
        self.transfer_nodes: list[SteadyStateTransfer] = workload.transfer_nodes

        # Pre-compute “bare” latencies (runtime for comp., 0 for tensors)
        self.latency = {}
        for n in chain(self.comp_nodes, self.tensor_nodes):
            self.latency[n] = n.runtime if n.runtime not in (None, 0) else 1.0

        # Transfer-node / path → latency  (dependent on link bandwidth)
        for n in self.transfer_nodes:
            for p in n.possible_resource_allocation:
                bw = min(link.bandwidth for link in p)
                self.latency[(n, tuple(p))] = n.tensor.size / bw

        # ----------------------------------------------------------------
        # Gurobi model
        # ----------------------------------------------------------------
        self.m = gp.Model("transfer_allocation")
        self.m.setParam("OutputFlag", 1)

        # --------------- decision vars: path choice  --------------------
        self.pvar = {  # (node, path) → binary
            (n, tuple(p)): self.m.addVar(vtype=GRB.BINARY, name=f"p_{n.node_name}_{_resource_key(p)}")
            for n in self.transfer_nodes
            for p in n.possible_resource_allocation
        }
        # one-hot per transfer node
        for n in self.transfer_nodes:
            self.m.addConstr(
                gp.quicksum(self.pvar[(n, tuple(p))] for p in n.possible_resource_allocation) == 1,
                name=f"one_path_{n.node_name}",
            )

        # --------------- slot latency variables ------------------------
        self.slot_lat = {t: self.m.addVar(vtype=GRB.INTEGER, name=f"slot_lat_{t}") for t in range(self.max_slot + 1)}
        self._add_slot_latency_constraints()

        # --------------- link contention -------------------------------
        self._add_link_contention_constraints()

        # --------------- idle / overlap --------------------------------
        self._add_idle_overlap_constraints()

        # --------------- objective -------------------------------------
        self._add_latency_objective()

    # ------------------------------------------------------------------ #
    # helper: slot-latency constraints                                   #
    # ------------------------------------------------------------------ #
    def _add_slot_latency_constraints(self):
        # fixed-latency nodes -------------------------------------------
        for n in chain(self.comp_nodes, self.tensor_nodes):
            t = self.slot_of[n]
            self.m.addConstr(
                self.slot_lat[t] >= self.latency[n],
                name=f"lat_fixed_{n.node_name}",
            )

        # transfer nodes – latency depends on chosen path ---------------
        for n in self.transfer_nodes:
            t = self.slot_of[n]
            for p in n.possible_resource_allocation:
                key = (n, tuple(p))
                self.m.addConstr(
                    self.slot_lat[t] >= self.latency[key] * self.pvar[key],
                    name=f"lat_tr_{n.node_name}_{_resource_key(p)}",
                )

    # ------------------------------------------------------------------ #
    # helper: link contention                                            #
    # ------------------------------------------------------------------ #
    def _add_link_contention_constraints(self):
        usage = {}  # (link, slot) → linear expr (sum of path vars)
        for n in self.transfer_nodes:
            s = self.slot_of[n]
            for p in n.possible_resource_allocation:
                v = self.pvar[(n, tuple(p))]
                for link in p:
                    usage.setdefault((link, s), gp.LinExpr()).addTerms(1, v)

        for (link, s), expr in usage.items():
            self.m.addConstr(expr <= 1, name=f"link_use_{_resource_key(link)}_{s}")

    # ------------------------------------------------------------------ #
    # helper: idle / overlap                                             #
    # ------------------------------------------------------------------ #
    def _is_node_in_slot(self, n, t) -> int:
        """Constant 1 / 0 because slots are fixed."""
        return 1 if self.slot_of[n] == t else 0

    def _add_idle_overlap_constraints(self):
        eps = 1e-3
        M = len(self.workload.node_list) + eps

        resources = {}  # Core ↦ {nodes}, Link ↦ {nodes}
        # Cores
        for n in self.comp_nodes:
            resources.setdefault(n.chosen_resource_allocation, set()).add(n)
        # Links (potentially)
        for n in self.transfer_nodes:
            for p in n.possible_resource_allocation:
                for lk in p:
                    resources.setdefault(lk, set()).add(n)

        # sums, binaries, idle-latency per resource ---------------------
        self.slot_idle_start = {}
        self.slot_idle_end = {}
        self.idle_latency = {}

        for r, nodes in resources.items():
            # inclusive / exclusive cumulative sums
            sum_incl = {}
            sum_excl = {}

            for t in range(self.max_slot + 1):
                si = self.m.addVar(vtype=GRB.INTEGER, name=f"sum_incl_{_resource_key(r)}_{t}")
                se = self.m.addVar(vtype=GRB.INTEGER, name=f"sum_excl_{_resource_key(r)}_{t}")
                sum_incl[t], sum_excl[t] = si, se

                self.m.addConstr(
                    si == gp.quicksum(self._is_node_in_slot(n, tau) for n in nodes for tau in range(t + 1))
                )
                self.m.addConstr(se == gp.quicksum(self._is_node_in_slot(n, tau) for n in nodes for tau in range(t)))

            total = gp.quicksum(self._is_node_in_slot(n, tau) for n in nodes for tau in range(self.max_slot + 1))

            # idle-start / end binaries & latency account
            idle_lat = self.m.addVar(vtype=GRB.INTEGER, name=f"idle_lat_{_resource_key(r)}")
            self.idle_latency[r] = idle_lat

            expr = gp.LinExpr()

            for t in range(self.max_slot + 1):
                id_st = self.m.addVar(vtype=GRB.BINARY, name=f"idle_st_{_resource_key(r)}_{t}")
                id_en = self.m.addVar(vtype=GRB.BINARY, name=f"idle_en_{_resource_key(r)}_{t}")
                self.slot_idle_start[(r, t)] = id_st
                self.slot_idle_end[(r, t)] = id_en

                # “start”  ↔  incl ≤ 1
                self.m.addConstr(sum_incl[t] + eps <= 1 + M * (1 - id_st))
                self.m.addConstr(sum_incl[t] >= 1 - M * id_st)

                # “end”    ↔  excl ≥ total-1
                self.m.addConstr(sum_excl[t] >= total - 1 + eps - M * (1 - id_en))
                self.m.addConstr(sum_excl[t] <= total - 1 + M * id_en)

                # contribute latency of this slot when idle
                expr += (id_st + id_en) * self.slot_lat[t]

            self.m.addConstr(idle_lat == expr, name=f"idle_lat_def_{_resource_key(r)}")

        # global overlap = min(idle_latencies)
        self.overlap = self.m.addVar(vtype=GRB.INTEGER, name="overlap")
        for r, v in self.idle_latency.items():
            self.m.addConstr(self.overlap <= v, name=f"overlap_le_{_resource_key(r)}")

    # ------------------------------------------------------------------ #
    # objective                                                          #
    # ------------------------------------------------------------------ #
    def _add_latency_objective(self):
        """
        total_latency =  iter * Σ slot_lat  –  (iter-1) * overlap
        """
        total_slot = gp.quicksum(self.slot_lat.values())
        self.total_latency = self.m.addVar(vtype=GRB.INTEGER, name="total_lat")
        self.m.addConstr(self.total_latency == self.iter * total_slot - (self.iter - 1) * self.overlap)
        self.m.setObjective(self.total_latency, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # solve & report                                                     #
    # ------------------------------------------------------------------ #
    def solve(self):
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError("No feasible solution")

        chosen_path = {
            n: next(p for p in n.possible_resource_allocation if self.pvar[(n, tuple(p))].X > 0.5)
            for n in self.transfer_nodes
        }
        # schedule is fixed – just return it
        return self.slot_of, chosen_path
