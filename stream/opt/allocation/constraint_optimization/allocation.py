import sys
from typing import TypeAlias

import gurobipy as gp
from gurobipy import GRB
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

ALLOCATION_T: TypeAlias = list[tuple[int, str, tuple[int, int]]]


def get_optimal_allocations(
    workload: ComputationNodeWorkload,
    accelerator: Accelerator,
    cost_lut: CostModelEvaluationLUT,
    iterations: int,
    gap: float = 0.5,
    time_limit: int = 600,
    latency_attr: str = "latency_total1",
) -> ALLOCATION_T:
    core_ids = sorted((core.id for core in accelerator.cores.node_list if core.id != accelerator.offchip_core_id))
    core_capacities = get_core_capacities(accelerator, MemoryOperand("I2"), core_ids)

    nodes = sorted(workload.node_list)
    ids = convert_ids(nodes)

    latencies, possible_allocation_splits = get_latencies(
        nodes,
        core_ids,
        accelerator,
        cost_lut,
        impossible_lat=0,
        ids=ids,
        latency_attr=latency_attr,
    )
    energies = get_energies(nodes, core_ids, accelerator, cost_lut, impossible_energy=0, ids=ids)
    output_operand = LayerOperand("O")
    dependencies = {
        (ids[p], ids[c]): p.operand_size_bit[output_operand] for p, c in workload.edges() if p in nodes and c in nodes
    }

    layer_ids = sorted(set(n.id for n in nodes))
    groups: dict[int, list] = {layer_id: [] for layer_id in layer_ids}

    for node in nodes:
        groups[node.id].append(ids[node])

    weights = {}

    for group in groups.values():
        assert len(group) > 0, "Empty group given"
        for i, node_id in enumerate(group):
            node = next(k for k, v in ids.items() if v == node_id)
            constant_i2_ops = [
                op for op in node.constant_operands if node.memory_operand_links[op] == MemoryOperand("I2")
            ]
            nb_weights = sum((node.operand_size_bit[op] for op in constant_i2_ops))
            if i == 0:
                weights[node_id] = nb_weights
            else:
                assert nb_weights == weights[group[0]], f"Grouped nodes {group} don't have same amount of weights"
                weights[node_id] = 0

    if len(nodes) == 1:
        core_capacities = {core_name: 10e10 for core_name in [f"Core {i}" for i in core_ids]}

    allocation = constraint_allocation_optimization(
        latencies,
        energies,
        weights,
        dependencies,
        core_capacities,
        groups,
        possible_allocation_splits,
        iterations,
        gap=gap,
        time_limit=time_limit,
    )
    allocation = invert_ids_list(allocation, len(nodes))

    return allocation


def constraint_allocation_optimization(
    latencies: dict[tuple[int, str, int], int],
    energies: dict[tuple[int, str], float],
    weights_per_id: dict[int, int],
    dependencies: dict[tuple[int, int], int],
    core_capacities: dict[str, float],
    groups: dict[int, list[int]],
    possible_allocation_splits: dict[int, dict[str, dict[int, int]]],
    N: int = 1,
    gap: float = 0.5,
    time_limit: int = 600,
) -> list[tuple[int, int, int]]:
    """Get the optimal node-core allocation using constraint optimization.
    The timeline is divided into a number of slots. Each node will be assigned to one slot.

    Args:
        latencies: Latency for each node in form {id: latency}
        weights_per_id: Weights (in bits) for each node in form {id: weights}
        dependencies: Dependencies between nodes in form {(producer_id, consumer_id): tensor_size}
        core_capacities: Weight capacity (in bits) of each core in form {core: capacity}
        allocations: TODO: Add fixed allocation constraints
        N: The number of iterations of the steady state
    """
    node_core_k_ids, lats = gp.multidict(latencies)
    node_ids = sorted(set([node_core_id[0] for node_core_id in node_core_k_ids]))
    _, weights = gp.multidict(weights_per_id)
    cores, capacities = gp.multidict(core_capacities)
    slots = list(range(len(node_ids)))
    k_split_list = gp.tuplelist(range(1, len(cores) + 1))
    m = gp.Model("scheduling")
    # Disable prints
    m.setParam("outputFlag", 0)
    m.setParam("LogToConsole", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("Threads", 1)
    # Number of K splits for each node through one-hot vector
    k_split_vec = m.addVars(node_ids, k_split_list, vtype=GRB.BINARY, name="k_split_vec")
    m.addConstrs((k_split_vec.sum(node_id, "*") == 1 for node_id in node_ids))
    # Total number of K splits is the number of cores a node is allocated to
    k_splits = m.addVars(node_ids, vtype=GRB.INTEGER, ub=len(cores), name="k_splits")
    # FIX THE NUMBER OF K SPLITS TO 1 FOR COMPARISON WITH GENETIC ALGORITHM
    # k_splits = {node_id: 1 for node_id in node_ids}
    m.addConstrs(
        (k_splits[node_id] == gp.quicksum(k_split_vec[node_id, k] * k for k in k_split_list) for node_id in node_ids)
    )
    # Core assignments: to which cores is a node allocated (no notion of time slot yet)
    # This is linked to the assignments with slot below where 'assignments' is defined
    core_assignments = m.addVars(cores, node_ids, vtype=GRB.BINARY, name="core_assignments")
    m.addConstrs((core_assignments.sum("*", node_id) == k_splits[node_id] for node_id in node_ids))
    # Enfoce only the possible allocation - split combinations are allowed

    for node_id in node_ids:
        m.addConstrs(
            (
                core_assignments[core, node_id]
                <= gp.quicksum(
                    possible_allocation_splits[node_id][core][k] * k_split_vec[node_id, k] for k in k_split_list
                )
                for core in cores
            )
        )
    # Latency of each K split on each core is determined through the given latency for entire node
    lat_per_id_per_core = m.addVars(node_ids, cores, vtype=GRB.INTEGER, name="lat_per_core_per_id")
    for node_id in node_ids:
        m.addConstrs(
            (
                lat_per_id_per_core[node_id, core]
                == gp.quicksum(k_split_vec[node_id, k] * lats[node_id, core, k] for k in k_split_list)
                for core in cores
            )
        )
    slot_assignments = m.addVars(slots, node_ids, vtype=GRB.BINARY, name="node_assignments")
    m.addConstrs((slot_assignments.sum("*", node_id) == 1 for node_id in node_ids))
    assignments = m.addVars(cores, slots, node_ids, vtype=GRB.BINARY, name="assignments")
    m.addConstrs(
        (
            core_assignments[core, node_id] == assignments.sum(core, "*", node_id)
            for core in cores
            for node_id in node_ids
        )
    )
    # Each node should be assigned to one core and one slot
    m.addConstrs(
        (
            slot_assignments[slot, node_id] * assignments.sum("*", slot, node_id)
            == slot_assignments[slot, node_id] * k_splits[node_id]
            for node_id in node_ids
            for slot in slots
        )
    )
    m.addConstrs((assignments.sum(core, slot, "*") <= 1 for core in cores for slot in slots))
    m.addConstrs((assignments.sum("*", "*", node_id) == k_splits[node_id] for node_id in node_ids))
    # Groups should have the same core allocation
    for _, group in groups.items():
        for pair in zip(group, group[1:]):
            node_i, node_j = pair
            m.addConstrs((assignments.sum(core, "*", node_i) == assignments.sum(core, "*", node_j) for core in cores))
    # # Force impossible allocations to zero:
    # for node_id, impossible_cores in impossible_allocations.items():
    #     m.addConstrs((assignments.sum(impossible_core, "*", node_id) == 0 for impossible_core in impossible_cores))
    # Dependency constraints
    slot_per_id = m.addVars(node_ids, vtype=GRB.INTEGER, name="slot_per_id")
    m.addConstrs(
        (
            slot_per_id[node_id] == gp.quicksum(slot * slot_assignments[slot, node_id] for slot in slots)
            for node_id in node_ids
        )
    )
    for dependency in dependencies:
        p_id, c_id = dependency
        m.addConstr(slot_per_id[c_id] >= slot_per_id[p_id] + 1)
    # Calculate the number of weights per K split
    weights_per_split = m.addVars(node_ids, vtype=GRB.INTEGER, name="weights_per_split")
    m.addConstrs((weights_per_split[node_id] * k_splits[node_id] >= weights[node_id] for node_id in node_ids))
    # Calculate the number of weights on each core
    weights_per_core = m.addVars(cores, vtype=GRB.INTEGER, name="weights_per_core")
    m.addConstrs(
        (
            weights_per_core[core]
            >= gp.quicksum(weights_per_split[id] * assignments.sum(core, "*", id) for id in node_ids)
            for core in cores
        )
    )
    # Limit the number of weights on each core
    m.addConstrs((weights_per_core[core] <= capacities[core] for core in cores))
    # Calculate the latency of each slot depending on assigned id in the slot
    lat_per_core_per_slot = m.addVars(cores, slots, vtype=GRB.INTEGER, name="lat_per_core_per_slot")
    m.addConstrs(
        (
            lat_per_core_per_slot[core, slot]
            == gp.quicksum(lat_per_id_per_core[id, core] * assignments[core, slot, id] for id in node_ids)
            for core in cores
            for slot in slots
        )
    )

    lat_per_slot = m.addVars(slots, vtype=GRB.INTEGER, name="lat_per_slot")
    m.addConstrs(
        lat_per_slot[slot] == gp.max_(lat_per_core_per_slot[core, slot] for core in cores)  # type: ignore
        for slot in slots
    )

    lat = m.addVar(vtype=GRB.INTEGER, name="lat")
    m.addConstr(lat == gp.quicksum(lat_per_slot))

    # In order to get the period of subsequent steady state iterations,
    # we need to compute the amount of overlap possible between iterations
    # for example, with x, y, z representing allocations of iteration 0, 1, 2:
    # slot  0 1 2 3 4 5
    # c0   |x|-|x|y|-|y|z|...
    # c1   |-|x|-|x|y|-|y|...
    # there is overlap between iteration 1 (y) on core 0.
    # We introduce some helper variables to compute the start and end:
    # Inclusive sum of the assignments on each core
    sum_inclusive = m.addVars(cores, slots, vtype=GRB.INTEGER, name="sum_inclusive")
    m.addConstrs(
        sum_inclusive[core, slot] == gp.quicksum(assignments.sum(core, t, "*") for t in range(slot + 1))
        for core in cores
        for slot in slots
    )
    # Exclusive sum of the assignments on each core
    sum_exclusive = m.addVars(cores, slots, vtype=GRB.INTEGER, name="sum_exclusive")
    m.addConstrs(
        sum_exclusive[core, slot] == gp.quicksum(assignments.sum(core, t, "*") for t in range(slot))
        for core in cores
        for slot in slots
    )
    # Idle slots at the start
    eps = 0.0001
    M = len(node_ids) + eps
    idle_start = m.addVars(cores, slots, vtype=GRB.BINARY, name="idle_start")
    m.addConstrs(
        1 >= sum_inclusive[core, slot] + eps - M * (1 - idle_start[core, slot]) for core in cores for slot in slots
    )
    m.addConstrs(1 <= sum_inclusive[core, slot] + M * idle_start[core, slot] for core in cores for slot in slots)
    # Number of nodes assigned to core
    assignments_sum = m.addVars(cores, vtype=GRB.INTEGER, name="assignments_sum")
    m.addConstrs(assignments_sum[core] == assignments.sum(core, "*", "*") for core in cores)
    # Idle slots at the end
    idle_end = m.addVars(cores, slots, vtype=GRB.BINARY, name="idle_end")
    m.addConstrs(
        sum_exclusive[core, slot] >= assignments_sum[core] - 1 + eps - M * (1 - idle_end[core, slot])
        for core in cores
        for slot in slots
    )
    m.addConstrs(
        sum_exclusive[core, slot] <= assignments_sum[core] - 1 + M * idle_end[core, slot]
        for core in cores
        for slot in slots
    )

    # Total idle time: sum of start and end idle times convert to latency of each slot
    idle_total = m.addVars(cores, vtype=GRB.INTEGER, name="idle_total")
    m.addConstrs(
        idle_total[core]
        == gp.quicksum((idle_start[core, slot] + idle_end[core, slot]) * lat_per_slot[slot] for slot in slots)
        for core in cores
    )
    # Minimum idle time across all cores = overlap possible between iterations
    idle_min = m.addVar(vtype=GRB.INTEGER, name="idle_min")
    m.addConstr(idle_min == gp.min_(idle_total))

    total_lat = m.addVar(vtype=GRB.INTEGER, name="total_lat")
    m.addConstr(total_lat == N * lat - (N - 1) * idle_min)

    ################################## ENERGY CONSTRAINTS ##################################
    # node_ids_ens, ens = gp.multidict(energies)
    # assert node_ids_ens == node_core_k_ids
    # en_per_id_per_core = m.addVars(
    #     node_ids, cores, vtype=GRB.INTEGER, name="en_per_core_per_id"
    # )
    # # Max number of K splits of each node determined through possible allocations
    # for node_id, impossible_cores in impossible_allocations.items():
    #     possible_cores = sorted(set(cores) - set(impossible_cores))
    #     m.addConstrs(
    #         en_per_id_per_core[node_id, core] * k_splits[node_id] >= ens[node_id, core]
    #         for core in possible_cores
    #     )
    # BELOW WAS UNNOTED
    # en_per_node = m.addVars(node_ids, vtype=GRB.INTEGER, name="en_per_node")
    # m.addConstrs(
    #     en_per_node[node_id] >= gp.quicksum(assignments.sum(core, "*", node_id) * ens[node_id, core] for core in cores)
    #     for node_id in node_ids
    # )
    # en_per_iteration = m.addVar(vtype=GRB.INTEGER, name="en_per_iteration")
    # m.addConstr(en_per_iteration == gp.quicksum(en_per_node))
    # energy = m.addVar(vtype=GRB.INTEGER, name="energy")
    # m.addConstr(energy == N * en_per_iteration)

    # # EDP
    # edp = m.addVar(vtype=GRB.INTEGER, name="edp")
    # m.addConstr(edp == energy * total_lat)
    ########################################################################################

    # m.setObjective(total_lat, GRB.MINIMIZE)
    m.setObjective(total_lat, GRB.MINIMIZE)

    # Keep multiple solutions within a gap
    # m.setParam(GRB.Param.PoolSolutions, 1024)
    m.setParam(GRB.Param.PoolGap, gap)
    # m.setParam("NonConvex", 2)

    m.optimize()

    # Only save the best solution for now
    allocations = []
    best_latencies = []
    nSolutions = m.SolCount
    if nSolutions == 0:
        from pprint import pprint

        print("energies")
        pprint(energies)
        print("possible_allocation_splits")
        pprint(possible_allocation_splits)
        m.computeIIS()
        m.write("iismodel.ilp")
        raise ValueError("No solutions for constraint optimization. Check iismodel.ilp")
    for e in range(nSolutions):
        m.setParam(GRB.Param.SolutionNumber, e)
        best_latencies.append(m.ObjVal)
        allocation = []
        for (core, slot, node_id), i in assignments.items():
            if round(i.X) == 1:
                allocation.append((slot, core, node_id))
        allocation = sorted(allocation)
        allocations.append(allocation)

    status = m.Status
    if status == GRB.UNBOUNDED:
        print("The model cannot be solved because it is unbounded")
        sys.exit(-1)
    if status == GRB.OPTIMAL:
        print(f"The optimal objective is {m.ObjVal}")
    elif status == GRB.TIME_LIMIT:
        print(f"The optimization was terminated after {time_limit} seconds.")
    else:
        print(f"Optimization was stopped with status {status}")
        sys.exit(-1)
    return allocations[0]
