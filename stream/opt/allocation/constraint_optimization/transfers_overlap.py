import gurobipy as gp
from gurobipy import GRB

from stream.workload.steady_state_workload import SteadyStateWorkload


class TransferAllocator:
    def __init__(self, workload: SteadyStateWorkload, max_timeslot: int = 100, iterations: int = 1):
        self.workload = workload
        self.max_timeslot = max_timeslot
        self.iterations = iterations

        self.comp_nodes = workload.computation_nodes
        self.tensor_nodes = workload.tensor_nodes
        self.transfer_nodes = workload.transfer_nodes

        self.m = gp.Model("transfer_allocation")
        self.m.setParam("OutputFlag", 1)  # Enable output for debugging

        self.timeslot_vars = {}
        self.transfer_path_vars = {}
        self.transfer_paths = {}
        self.node_time_indicator = {}
        self.link_active = {}
        self.node_latency = {}  # (node, resource) -> latency
        self.slot_latency_vars = {}
        self.resource_idle_start = {}  # (resource, t) -> binary
        self.resource_idle_end = {}  # (resource, t) -> binary
        self.resource_idle_latencies = {}
        self.overlap_var = None
        self.total_latency = None

    def solve(self):
        self._initialize_node_latencies()
        self._add_timeslot_variables()
        self._add_node_timeslot_indicators()
        self._add_transfer_path_variables()
        self._add_transfer_path_constraints()
        self._add_topological_constraints()
        self._add_link_active_variables()
        self._add_link_contention_constraints()
        self._add_slot_latency_variables()
        self._add_idle_slot_indicators()
        self._add_overlap_variable()
        self._add_latency_objective()

        self.m.optimize()

        if self.m.Status == GRB.OPTIMAL:
            print(f"Optimal latency-aware schedule found with latency {self.total_latency.X}")
            schedule = {node: int(self.timeslot_vars[node].X) for node in self.workload.node_list}
            routing = {
                node: next(path for path, var in self.transfer_path_vars[node].items() if var.X > 0.5)
                for node in self.transfer_nodes
            }
            return schedule, routing
        else:
            raise RuntimeError("No optimal solution found.")

    def _initialize_node_latencies(self):
        self.node_latency = {}
        for node in self.comp_nodes:
            assert node.runtime is not None, f"Node {node.node_name} has no runtime defined."
            self.node_latency[(node, node.chosen_resource_allocation)] = 10

        for node in self.tensor_nodes:
            self.node_latency[(node, node.chosen_resource_allocation)] = 0

        for node in self.transfer_nodes:
            for path in node.possible_resource_allocation:
                min_bandwidth = min(link.bandwidth for link in path)
                latency = node.tensor.size / min_bandwidth
                self.node_latency[(node, tuple(path))] = latency

    def _add_timeslot_variables(self):
        self.timeslot_vars = {
            node: self.m.addVar(vtype=GRB.INTEGER, lb=0, ub=self.max_timeslot, name=f"t_{node.node_name}")
            for node in self.workload.node_list
        }

    def _add_node_timeslot_indicators(self):
        for node in self.workload.node_list:
            for t in range(self.max_timeslot + 1):
                v = self.m.addVar(vtype=GRB.BINARY, name=f"active_{node.node_name}_{t}")
                self.node_time_indicator[(node, t)] = v

            # Enforce that only one is active
            self.m.addConstr(
                gp.quicksum(self.node_time_indicator[node, t] for t in range(self.max_timeslot + 1)) == 1,
                name=f"unique_slot_{node.node_name}",
            )

            # Tie to timeslot_vars[node]
            self.m.addConstr(
                self.timeslot_vars[node]
                == gp.quicksum(t * self.node_time_indicator[node, t] for t in range(self.max_timeslot + 1)),
                name=f"slot_binding_{node.node_name}",
            )

    def _add_transfer_path_variables(self):
        self.transfer_paths = {
            node: [tuple(path) for path in node.possible_resource_allocation] for node in self.transfer_nodes
        }
        self.transfer_path_vars = {
            node: {path: self.m.addVar(vtype=GRB.BINARY, name=f"p_{node.node_name}_{str(path)}") for path in paths}
            for node, paths in self.transfer_paths.items()
        }

    def _add_transfer_path_constraints(self):
        for node, path_vars in self.transfer_path_vars.items():
            self.m.addConstr(gp.quicksum(path_vars.values()) == 1, name=f"one_path_{node.node_name}")

    def _add_topological_constraints(self):
        for u, v in self.workload.get_edges():
            self.m.addConstr(
                self.timeslot_vars[v] >= self.timeslot_vars[u] + 1, name=f"order_{u.node_name}_{v.node_name}"
            )

    def _add_link_active_variables(self):
        for node in self.transfer_nodes:
            for path in self.transfer_paths[node]:
                for link in path:
                    for t in range(self.max_timeslot + 1):
                        var = self.m.addVar(vtype=GRB.BINARY, name=f"link_active_{node.node_name}_{link}_{t}")
                        self.link_active[(node, link, t)] = var
                        self.m.addConstr(var <= self.transfer_path_vars[node][path])
                        self.m.addConstr(var <= self.node_time_indicator[node, t])
                        self.m.addConstr(
                            var >= self.transfer_path_vars[node][path] + self.node_time_indicator[node, t] - 1
                        )

    def _add_link_contention_constraints(self):
        link_time_usage = {}
        for (node, link, t), var in self.link_active.items():
            link_time_usage.setdefault((link, t), []).append(var)

        for (link, t), usage_vars in link_time_usage.items():
            self.m.addConstr(gp.quicksum(usage_vars) <= 1, name=f"link_usage_{link}_{t}")

    def _add_slot_latency_variables(self):
        for t in range(self.max_timeslot + 1):
            var = self.m.addVar(vtype=GRB.INTEGER, name=f"slot_latency_{t}")
            self.slot_latency_vars[t] = var
            for node in self.workload.node_list:
                if node in self.transfer_nodes:
                    for path in self.transfer_paths[node]:
                        latency = self.node_latency[(node, path)]
                        self.m.addConstr(
                            var >= latency * self.node_time_indicator[node, t] * self.transfer_path_vars[node][path],
                            name=f"latency_bound_{node.node_name}_{t}_{str(path)}",
                        )
                else:
                    res = node.chosen_resource_allocation
                    latency = self.node_latency[(node, res)]
                    self.m.addConstr(
                        var >= latency * self._is_node_in_slot(node, t), name=f"latency_bound_{node.node_name}_{t}"
                    )

    def _is_node_in_slot(self, node, t):
        return self.node_time_indicator[(node, t)]

    def _add_idle_slot_indicators(self):
        eps = 0.001
        M = len(self.workload.node_list) + eps

        resource_to_nodes = {}

        for node in self.comp_nodes:
            res = node.chosen_resource_allocation
            resource_to_nodes.setdefault(res, set()).add(node)

        for node, link, _ in self.link_active:
            resource_to_nodes.setdefault(link, set()).add(node)

        self.sum_inclusive = {}
        self.sum_exclusive = {}
        self.total_assignments = {}

        for res, nodes in resource_to_nodes.items():
            for t in range(self.max_timeslot + 1):
                # Sum of assignments up to and including t
                inc = self.m.addVar(vtype=GRB.INTEGER, name=f"sum_inclusive_{res}_{t}")
                self.sum_inclusive[(res, t)] = inc
                self.m.addConstr(
                    inc == gp.quicksum(self._is_node_in_slot(n, tau) for n in nodes for tau in range(t + 1))
                )

                # Sum of assignments strictly before t
                exc = self.m.addVar(vtype=GRB.INTEGER, name=f"sum_exclusive_{res}_{t}")
                self.sum_exclusive[(res, t)] = exc
                self.m.addConstr(exc == gp.quicksum(self._is_node_in_slot(n, tau) for n in nodes for tau in range(t)))

            # Total assignments on this resource
            total = self.m.addVar(vtype=GRB.INTEGER, name=f"total_assigned_{res}")
            self.total_assignments[res] = total
            self.m.addConstr(
                total
                == gp.quicksum(self._is_node_in_slot(n, tau) for n in nodes for tau in range(self.max_timeslot + 1))
            )

            for t in range(self.max_timeslot + 1):
                idle_start = self.m.addVar(vtype=GRB.BINARY, name=f"idle_start_{res}_{t}")
                idle_end = self.m.addVar(vtype=GRB.BINARY, name=f"idle_end_{res}_{t}")
                self.resource_idle_start[(res, t)] = idle_start
                self.resource_idle_end[(res, t)] = idle_end

                # Start = before anything has run yet
                self.m.addConstr(self.sum_inclusive[(res, t)] + eps <= 1 + M * (1 - idle_start))
                self.m.addConstr(self.sum_inclusive[(res, t)] >= 1 - M * idle_start)

                # End = everything already ran before this slot
                self.m.addConstr(
                    self.sum_exclusive[(res, t)] >= self.total_assignments[res] - 1 + eps - M * (1 - idle_end)
                )
                self.m.addConstr(self.sum_exclusive[(res, t)] <= self.total_assignments[res] - 1 + M * idle_end)

    def _add_overlap_variable(self):
        # Aggregate idle latency for each resource
        for r in set(k[0] for k in self.resource_idle_start):
            idle_latency = self.m.addVar(vtype=GRB.INTEGER, name=f"idle_latency_{r}")
            expr = gp.quicksum(
                self.resource_idle_start[(r, t)] * self.slot_latency_vars[t]
                + self.resource_idle_end[(r, t)] * self.slot_latency_vars[t]
                for t in range(self.max_timeslot + 1)
            )
            self.m.addConstr(idle_latency == expr, name=f"idle_latency_def_{r}")
            self.resource_idle_latencies[r] = idle_latency

        # Overlap is minimum across all resource idle latencies
        self.overlap_var = self.m.addVar(vtype=GRB.INTEGER, name="overlap")
        for r, idle_var in self.resource_idle_latencies.items():
            self.m.addConstr(self.overlap_var <= idle_var, name=f"overlap_le_resource_{r}")

    def _add_latency_objective(self):
        self.total_latency = self.m.addVar(vtype=GRB.INTEGER, name="total_latency")
        total_slot_latency = gp.quicksum(self.slot_latency_vars.values())
        assert self.overlap_var is not None, "Overlap variable must be defined."
        self.m.addConstr(
            self.total_latency == self.iterations * total_slot_latency - (self.iterations - 1) * self.overlap_var,
            name="total_latency_def",
        )
        self.m.setObjective(self.total_latency, GRB.MINIMIZE)
