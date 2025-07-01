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

        self.m = gp.Model("transfer_schedule")
        self.m.setParam("OutputFlag", 1)

        self.timeslot_vars = {}
        self.transfer_path_vars = {}
        self.transfer_paths = {}
        self.transfer_time_indicator = {}
        self.transfer_active = {}
        self.makespan = None

    def solve(self):
        self._add_timeslot_variables()
        self._add_transfer_path_variables()
        self._add_transfer_path_constraints()
        self._add_topological_constraints()
        self._add_transfer_timeslot_indicators()
        self._add_transfer_active_variables()
        self._add_link_contention_constraints()
        self._add_makespan_objective()

        self.m.optimize()

        if self.m.Status == GRB.OPTIMAL:
            assert self.makespan is not None, "Makespan variable should be defined."
            print(f"Optimal schedule found with makespan {self.makespan.X}")
            schedule = {node: int(self.timeslot_vars[node].X) for node in self.workload.node_list}
            routing = {
                node: next(path for path, var in self.transfer_path_vars[node].items() if var.X > 0.5)
                for node in self.transfer_nodes
            }
            return schedule, routing
        else:
            raise RuntimeError("No optimal solution found.")

    def _add_timeslot_variables(self):
        self.timeslot_vars = {
            node: self.m.addVar(vtype=GRB.INTEGER, lb=0, ub=self.max_timeslot, name=f"t_{node.node_name}")
            for node in self.workload.node_list
        }

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

    def _add_transfer_timeslot_indicators(self):
        for node in self.transfer_nodes:
            for t in range(self.max_timeslot + 1):
                var = self.m.addVar(vtype=GRB.BINARY, name=f"time_{node.node_name}_{t}")
                self.transfer_time_indicator[(node, t)] = var
            self.m.addConstr(
                gp.quicksum(self.transfer_time_indicator[node, t] for t in range(self.max_timeslot + 1)) == 1
            )
            self.m.addConstr(
                self.timeslot_vars[node]
                == gp.quicksum(t * self.transfer_time_indicator[node, t] for t in range(self.max_timeslot + 1))
            )

    def _add_transfer_active_variables(self):
        for node in self.transfer_nodes:
            for path in self.transfer_paths[node]:
                for t in range(self.max_timeslot + 1):
                    var = self.m.addVar(vtype=GRB.BINARY, name=f"active_{node.node_name}_{str(path)}_{t}")
                    self.transfer_active[(node, path, t)] = var
                    self.m.addConstr(var <= self.transfer_path_vars[node][path])
                    self.m.addConstr(var <= self.transfer_time_indicator[node, t])
                    self.m.addConstr(
                        var >= self.transfer_path_vars[node][path] + self.transfer_time_indicator[node, t] - 1
                    )

    def _add_link_contention_constraints(self):
        link_usage = {}  # (link, slot) -> [(node, path)]
        for node in self.transfer_nodes:
            for path in self.transfer_paths[node]:
                for link in path:
                    for t in range(self.max_timeslot + 1):
                        link_usage.setdefault((link, t), []).append((node, path))

        for (link, t), node_path_list in link_usage.items():
            self.m.addConstr(
                gp.quicksum(self.transfer_active[(node, path, t)] for node, path in node_path_list) <= 1,
                name=f"link_usage_{link}_{t}",
            )

    def _add_makespan_objective(self):
        self.makespan = self.m.addVar(vtype=GRB.INTEGER, name="makespan")
        for node in self.workload.node_list:
            self.m.addConstr(self.makespan >= self.timeslot_vars[node], name=f"makespan_ge_t_{node.node_name}")
        self.m.setObjective(self.makespan, GRB.MINIMIZE)
