# transfer_and_tensor_allocator.py
import math
from collections import defaultdict
from math import ceil, prod
from typing import TypeAlias

import gurobipy as gp
from gurobipy import GRB, quicksum

from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.mapping.mapping import Mapping, Resource
from stream.opt.allocation.constraint_optimization.context import (
    TransferAndTensorContext,
    build_transfer_context,
)
from stream.opt.allocation.constraint_optimization.timeslot_allocation import (
    _resource_key,
)
from stream.workload.steady_state.iteration_space import ComputeTileReuse, MemTileReuse, SteadyStateIterationSpace
from stream.workload.steady_state.node import Node
from stream.workload.steady_state.workload import SteadyStateWorkload
from stream.workload.workload import ComputationNode, HasIterationSpace, InEdge, OutEdge, Tensor, TransferNode, Workload

PathKey: TypeAlias = tuple[TransferNode, tuple[CommunicationLink, ...]]
TensorAlloc: TypeAlias = dict[Tensor, Core]
TransferAlloc: TypeAlias = dict[TransferNode, tuple[CommunicationLink]]
MemoryAlloc: TypeAlias = dict[TransferNode, Core]


class TransferAndTensorAllocator:
    """
    MILP that decides

    1. **where** every *movable* Tensor lives,
    2. **which path** each Transfer uses,

    while all slots/resources coming from existing timeslots remain unchanged.
    """

    VAR_THRESHOLD = 0.5  # threshold to decide if a binary variable is chosen

    # ------------------------------------------------------------ #
    # ctor / public API                                            #
    # ------------------------------------------------------------ #
    def __init__(
        self,
        workload: Workload,
        timeslots: dict[Node, int],
        accelerator: Accelerator,
        iterations: int,
        ssis: dict[HasIterationSpace, SteadyStateIterationSpace],
        multiplicities: dict[ComputationNode, int],
        mapping: Mapping,
        cost_lut: CoreCostLUT,
        *,
        big_m: int | None = None,
        gurobi_verbosity: int = 1,
        nb_cols_to_use: int = 4,
        context: TransferAndTensorContext | None = None,
    ):
        self.workload = workload
        self.slot_of = timeslots
        self.accelerator = accelerator
        self.context = context or build_transfer_context(accelerator, nb_cols_to_use=nb_cols_to_use)
        self.offchip_core_id = self.context.offchip_core_id
        self.iterations = iterations
        self.ssis = ssis
        self.multiplicities = multiplicities
        self.mapping = mapping
        self.cost_lut = cost_lut

        self.max_slot = max(timeslots.values()) if timeslots else 0
        self.big_m = big_m or len(workload.nodes()) + 5
        self.force_io_transfers_on_mem_tile = self.context.force_io_transfers_on_mem_tile

        # ------------------- categorise nodes -------------------- #
        self.ssc_nodes: list[ComputationNode] = workload.get_computation_nodes()
        self.transfer_nodes: list[TransferNode] = workload.get_transfer_nodes()

        # Possible tensor allocations
        self.tensor_fixed: list[Tensor] = []
        self.tensor_var: list[Tensor] = []  # will remain empty for now
        self.possible_tensor_allocations: dict[Tensor, list[Resource]] = {}
        for in_edge in workload.get_in_edges():
            self.possible_tensor_allocations[in_edge.outputs[0]] = [self.accelerator.get_core(self.offchip_core_id)]
            self.tensor_fixed.append(in_edge.outputs[0])
        for node in workload.get_computation_nodes():
            for t in node.tensors:
                if t not in self.possible_tensor_allocations:
                    self.possible_tensor_allocations[t] = self.mapping.get(node).resource_allocation
                    self.tensor_fixed.append(t)
        for out_edge in workload.get_out_edges():
            tensor = out_edge.inputs[0]
            self.possible_tensor_allocations[tensor] = [self.accelerator.get_core(self.offchip_core_id)]
            self.tensor_fixed.append(tensor)

        # Possible transfer allocations
        self.possible_transfer_allocations: dict[TransferNode, list[Resource]] = {}
        for node in workload.get_transfer_nodes():
            self.possible_transfer_allocations[node] = self.mapping.get(node).resource_allocation

        # ------------------------------------------------------------------------------
        # memory cores that may act as on‑chip caches (exclude the DRAM/off‑chip id)
        # ------------------------------------------------------------------------------
        self.force_double_buffering = self.context.force_double_buffering
        self.mem_cores = list(self.context.mem_cores)

        # --------------- optimisation model ---------------------- #
        self.model = gp.Model("transfer_tensor_alloc")
        self.model.setParam("OutputFlag", gurobi_verbosity)

        # decision vars
        self.x_tensor: dict[tuple[Tensor, Core], gp.Var] = {}
        self.y_path: dict[PathKey, gp.Var] = {}

        # helpers
        self.link_set: set[CommunicationLink] = set()
        self.links_in_path: dict[PathKey, list[CommunicationLink]] = {}

        # latency vars
        self.slot_latency: dict[int, gp.Var] = {}
        self.overlap = None
        self.total_latency = None

        # transfer fire helpers init
        # dict((transfer, steady state iteration space index): (fires_across_ss_iterations, extra_mem_bytes))
        self._ensure_same_ssis_for_all_transfers()
        self.reuse_levels: dict[tuple[TransferNode, int], tuple[int, int]] = {}
        self.tiles_needed_levels: dict[tuple[TransferNode, int], int] = {}
        self.transfer_nodes_to_optimize_firings_for: list[TransferNode] = []
        self._init_transfer_fire_helpers()

        self._build_model()

    # ------------------------------------------------------------ #
    # internal helpers                                             #
    # ------------------------------------------------------------ #
    # ---------- memory factor ( NO ping-pong for now ) ---------- #
    # TODO: Update this for memory-only cores
    @staticmethod
    def _mem_factor(t: Tensor, core: Core) -> int:
        # if getattr(core, "type", None) == "COMPUTE":
        return 1  # NO ping–pong
        # else:
        #     full_fac = getattr(t, "slices_per_full", 1)
        #     return int(full_fac)

    # ---------- latency of a transfer along a path -------------- #
    @staticmethod
    def _transfer_latency(tr: TransferNode, path: tuple[CommunicationLink, ...]) -> int:
        if not path:
            return 0
        min_bw = min(link.bandwidth for link in path)
        assert len(tr.inputs) == 1, "Only single-input transfers are supported for latency calculation."
        tensor = tr.inputs[0]
        return ceil(tensor.size_bits() / min_bw)

    # ---------- init transfer fire helpers ----------------------- #
    def _ensure_same_ssis_for_all_transfers(self) -> None:
        """
        Ensure that all transfers have the same steady-state iteration space dimensions and sizes (SSIS).
        """
        first_ssis = self.ssis[self.transfer_nodes[0]]
        # first_transfer_ssis_dims = first_transfer_ssis.get_temporal_dimensions()
        first_transfer_ssis_sizes = first_ssis.get_temporal_sizes()
        first_transfer_ssis_total_size = prod(first_transfer_ssis_sizes)
        for tr in self.transfer_nodes:
            transfer_ssis = self.ssis[tr]
            # transfer_ssis_dims = transfer_ssis.get_temporal_dimensions()
            transfer_ssis_sizes = transfer_ssis.get_temporal_sizes()
            transfer_ssis_total_size = prod(transfer_ssis_sizes)
            # if not (
            #     transfer_ssis_dims == first_transfer_ssis_dims and transfer_ssis_sizes == first_transfer_ssis_sizes
            # ):
            if transfer_ssis_total_size != first_transfer_ssis_total_size:
                raise ValueError(
                    # f"Transfer {tr.node_name} has different SSIS dims and sizes than the {self.transfer_nodes[0]}: "
                    # f"{transfer_ssis_dims}, {transfer_ssis_sizes} != "
                    # f"{first_transfer_ssis_dims}, {first_transfer_ssis_sizes}"
                    f"Transfer {tr.name} has different SSIS total size than the {self.transfer_nodes[0].name}: "
                    f"{transfer_ssis_total_size} != {first_transfer_ssis_total_size}"
                )

    def _init_transfer_fire_helpers(self) -> None:
        for tr in self.transfer_nodes:  # only the movable tensors
            ssis = self.ssis[tr].get_temporal_variables()  # e.g. [Nk, Nk-1, …, N0]
            sizes = [iter_var.size for iter_var in ssis]
            relevancies = [iter_var.relevant for iter_var in ssis]
            reuses = [iter_var.compute_tile_reuse for iter_var in ssis]

            # Check that all compute reuses are NOT_SET, else continue
            if any(r != ComputeTileReuse.NOT_SET for r in reuses):
                continue
            self.transfer_nodes_to_optimize_firings_for.append(tr)

            # level = -1  →  "transfer every steady state iteration"
            fires = math.prod(sizes)
            size_factor = 1
            self.reuse_levels[(tr, -1)] = (fires, size_factor)
            tiles_needed = 1
            self.tiles_needed_levels[(tr, -1)] = tiles_needed
            # level = i  →  keep while loops 0..i stay in cache
            for i, (Nl, relevancy) in enumerate(zip(sizes, relevancies, strict=True)):  # i = 0 … K-1
                size_factor *= Nl if relevancy else 1  # enlarge tile size factor only if relevant
                tiles_needed *= Nl if relevancy else 1
                fires //= Nl  # fewer transfers
                self.reuse_levels[(tr, i)] = (fires, size_factor)
                self.tiles_needed_levels[(tr, i)] = tiles_needed

    # ------------------------------------------------------------------------------
    # only transfers whose src OR dst tensor is CONSTANT are eligible for MemC
    # ------------------------------------------------------------------------------
    def _is_const_i(self, tr: TransferNode) -> bool:
        src = tr.inputs[0]
        return isinstance(src, InEdge)

    def _is_const_o(self, tr: TransferNode) -> bool:
        dst = tr.outputs[0]
        return isinstance(dst, OutEdge)

    def _is_const_io(self, tr: TransferNode) -> bool:
        return self._is_const_i(tr) or self._is_const_o(tr)

    # ------------------------------------------------------------
    # bandwidth of the FIRST link on a DRAM → mem‑core path
    # ------------------------------------------------------------
    def _first_link_bw_from_dram(
        self,
        tr: TransferNode,
        mc: Core,
    ) -> int:
        """
        Return the maximum bandwidth of the first link among all paths that
        start at the off-chip core and end at the given memory core *mc*.

        Raises:
            ValueError - if no such path exists in tr.possible_resource_allocation.
        """
        best_bw: int | None = None
        for path in tr.possible_resource_allocation:  # list[list[Link]]
            if not path:  # empty == same core
                continue
            for link in path:
                sender = link.sender
                receiver = link.receiver
                if (  # for constant input transfers
                    isinstance(sender, Core)
                    and isinstance(receiver, Core)
                    and sender.id == self.offchip_core_id
                    and receiver.id == mc.id
                ) or (  # for constant output transfers
                    isinstance(sender, Core)
                    and isinstance(receiver, Core)
                    and sender.id == mc.id
                    and receiver.id == self.offchip_core_id
                ):
                    best_bw = link.bandwidth if best_bw is None else max(best_bw, link.bandwidth)
        if best_bw is None:
            raise ValueError(
                f"No DRAM→{mc.name} path found for transfer {tr.node_name}. "
                f"Check that transfer is for constant input/output tensors."
            )
        return best_bw

    # ------------------------------------------------------------ #
    # model construction                                           #
    # ------------------------------------------------------------ #
    def _build_model(self):
        self._create_vars()
        self._create_constraints()
        self._overlap_and_objective()

    # ...................... VARIABLES ................... #
    def _create_vars(self):
        self.__create_tensor_placement_vars()
        self.__create_transfer_path_vars()
        self.__create_compute_core_reuse_vars()
        self.__create_mem_core_reuse_vars()
        self.__create_transfer_mem_core_vars()
        self.__create_slot_latency_vars()

    def _paths_for_transfer(self, tr: TransferNode) -> list[PathKey]:
        return [p for p in self.y_path if p[0] is tr]

    @staticmethod
    def _path_src_core(path: tuple[CommunicationLink, ...]) -> Core | None:
        return path[0].sender if path else None

    @staticmethod
    def _path_dst_core(path: tuple[CommunicationLink, ...]) -> Core | None:
        return path[-1].receiver if path else None

    def __create_slot_latency_vars(self):
        for s in range(self.max_slot + 1):
            self.slot_latency[s] = self.model.addVar(vtype=GRB.INTEGER, name=f"L_{s}")

    def __create_compute_core_reuse_vars(self):
        self.z_stopC: dict[tuple[TransferNode, int], gp.Var] = {}
        for tr in self.transfer_nodes:
            sizes = self.ssis[tr].get_temporal_sizes()
            for stop in range(-1, len(sizes)):  # -1 .. K-1
                v = self.model.addVar(vtype=GRB.BINARY, name=f"zStopC_{tr.name}_L{stop}")
                self.z_stopC[(tr, stop)] = v
            # Choose exactly one stop-level
            self.model.addConstr(
                quicksum(self.z_stopC[(tr, s)] for s in range(-1, len(sizes))) == 1,
                name=f"zStopC_Choose_One_{tr.name}",
            )
            if tr not in self.transfer_nodes_to_optimize_firings_for:
                # Get the stop value by looking at the reuses of the ssis
                reuses = tr.steady_state_iteration_space.get_temporal_compute_tile_reuses()
                # Find the index of the last 'REUSE' in the reuses
                stop = -2
                for i in range(len(reuses) - 1, -1, -1):
                    if reuses[i] == ComputeTileReuse.REUSE:
                        stop = i
                        break
                assert stop >= -1, f"Something went wrong for Transfer {tr.node_name} REUSE indexing: {reuses}"
                # Set the z_stopC variable to 1 for the chosen stop
                self.model.addConstr(
                    self.z_stopC[(tr, stop)] == 1,
                    name=f"zStopC_FixedStop_{tr.node_name}_L{stop}",
                )

    def __create_mem_core_reuse_vars(self):
        self.z_stopM: dict[tuple[TransferNode, int], gp.Var] = {}

        for tr in self.transfer_nodes:
            sizes = self.ssis[tr].get_temporal_sizes()
            for stop in range(-1, len(sizes)):
                v = self.model.addVar(vtype=GRB.BINARY, name=f"zStopM_{tr.name}_L{stop}")
                self.z_stopM[(tr, stop)] = v
            self.model.addConstr(
                quicksum(self.z_stopM[(tr, s)] for s in range(-1, len(sizes))) == 1,
                name=f"cacheChooseStopM_{tr.name}",
            )
            # If any of the reuses is already set to NO_REUSE, enforce these levels to 0
            mem_tile_reuses = self.ssis[tr].get_temporal_mem_tile_reuses()
            for i, r in enumerate(mem_tile_reuses):
                if r == MemTileReuse.NO_REUSE:
                    self.model.addConstr(
                        self.z_stopM[(tr, i)] == 0,
                        name=f"zStopM_NoReuse_{tr.name}_L{i}",
                    )

            # stopM ≥ stopC (every cached loop for Compute must also be cached in the MemC)
            for s in range(-1, len(sizes)):
                cumC = quicksum(self.z_stopC[(tr, u)] for u in range(s, len(sizes)))
                cumM = quicksum(self.z_stopM[(tr, u)] for u in range(s, len(sizes)))
                self.model.addConstr(cumC <= cumM, name=f"nest_{tr.name}_L{s}")

    def __create_transfer_mem_core_vars(self):
        self.m_store: dict[tuple[TransferNode, Core], gp.Var] = {}
        if not self.mem_cores:
            return  # no memory cores, nothing to do
        for tr in self.transfer_nodes:
            memory_allocation = self.mapping.get(tr).memory_allocation
            if not memory_allocation:
                continue
            for mc in memory_allocation:
                v = self.model.addVar(vtype=GRB.BINARY, name=f"mStore_{tr.name}_{_resource_key(mc)}")
                self.m_store[(tr, mc)] = v
            if not self.force_io_transfers_on_mem_tile:
                v_none = self.model.addVar(vtype=GRB.BINARY, name=f"mStore_{tr.name}_NONE")
                self.m_store[(tr, None)] = v_none
            self.model.addConstr(
                quicksum(self.m_store[(tr, c)] for c in memory_allocation) == 1, name=f"chooseMemCore_{tr.name}"
            )

    def __create_transfer_path_vars(self):
        for tr in self.transfer_nodes:
            for p in self.possible_transfer_allocations[tr]:  # list[list[Link]]
                p_tuple = tuple(p)
                v = self.model.addVar(vtype=GRB.BINARY, name=f"y_{tr.name}_{hash(p_tuple)}")
                self.y_path[(tr, p_tuple)] = v
                self.links_in_path[(tr, p_tuple)] = list(p_tuple)
                self.link_set.update(p_tuple)

    def __create_tensor_placement_vars(self):
        for t in self.tensor_var:
            for c in self.possible_tensor_allocations[t]:  # type: ignore
                v = self.model.addVar(vtype=GRB.BINARY, name=f"x_{t.name}_{_resource_key(c)}")
                self.x_tensor[(t, c)] = v

    # ...................... tensor placement .................... #
    def _tensor_placement_constraints(self):
        for t in self.tensor_var:
            self.model.addConstr(
                quicksum(self.x_tensor[(t, c)] for c in self.possible_tensor_allocations[t]) == 1,  # type: ignore
                name=f"place_{t.name}",
            )

    # ...................... CONSTRAINTS ................... #
    def _create_constraints(self):
        self._path_choice_constraints()
        self._tensor_placement_constraints()
        self._transfer_fire_rate_constraints()
        self._link_contention_constraints()
        self._memory_capacity_constraints()
        self._object_fifo_depth_constraints()
        self._slot_latency_constraints()

    def _transfer_fire_rate_constraints(self):
        # firesC = Mem Core to Compute Core
        # firesM = DRAM to Mem Core
        self.firesC, self.firesM = {}, {}
        for tr in self.transfer_nodes:
            fires_c = self.model.addVar(vtype=GRB.INTEGER, name=f"firesC_{tr.name}")
            fires_m = self.model.addVar(vtype=GRB.INTEGER, name=f"firesM_{tr.name}")
            self.firesC[tr], self.firesM[tr] = fires_c, fires_m

            self.model.addConstr(
                fires_c
                == quicksum(
                    self.reuse_levels[(tr, s)][0] * self.z_stopC[(tr, s)]
                    for s in range(-1, len(self.ssis[tr].get_temporal_variables()))
                ),
                name=f"firesC_def_{tr.name}",
            )
            self.model.addConstr(
                fires_m
                == quicksum(
                    self.reuse_levels[(tr, s)][0] * self.z_stopM[(tr, s)]
                    for s in range(-1, len(self.ssis[tr].get_temporal_variables()))
                ),
                name=f"firesM_def_{tr.name}",
            )

    # ...................... path choice ........................ #
    def _path_choice_constraints(self) -> None:
        """
        For every transfer, exactly one path must be selected and path choices must be coherent
        with tensor placements for both source and destination tensors.
        """
        for tr in self.transfer_nodes:
            paths = self._paths_for_transfer(tr)
            self._add_one_path_constraint(tr, paths)
            self._add_source_tensor_coherence_constraints(tr, paths)
            self._add_destination_tensor_coherence_constraints(tr, paths)
            # self._add_io_transfers_path_coherence_constraints(tr, paths)

    def _add_one_path_constraint(self, tr: TransferNode, paths: list[PathKey]) -> None:
        """Ensure exactly one path is selected for each transfer."""
        self.model.addConstr(quicksum(self.y_path[p] for p in paths) == 1, name=f"one_path_{tr.name}")

    def _add_source_tensor_coherence_constraints(self, tr: TransferNode, paths: list[PathKey]) -> None:
        """Add constraints to ensure path choice is coherent with source tensor placement."""
        predecessors = list(self.workload.predecessors(tr))
        predecessors = tr.inputs
        assert all(isinstance(n, Tensor) for n in predecessors), (
            f"Transfer {tr.name} has non-tensor predecessor(s): {predecessors}"
        )
        successors = list(self.workload.successors(tr))
        for src_tensor in predecessors:
            if src_tensor in self.tensor_var:
                for p in paths:
                    src_core = self._path_src_core(p[1])
                    if src_core is not None:
                        assert isinstance(src_core, Core), f"Expected {src_core} to be a Core, got {type(src_core)}"
                        self.model.addConstr(
                            self.y_path[p] <= self.x_tensor[(src_tensor, src_core)],
                            name=f"path_core_link_src_{tr.name}_{_resource_key(src_core)}",
                        )
                        continue
                    # empty path is only possible if the src_tensor is fixed on the same core as the dst_tensor
                    dst_tensor = successors[0] if successors else None
                    if dst_tensor is not None and dst_tensor in self.tensor_fixed:
                        dst_core = self.resource_of[dst_tensor]
                        assert isinstance(dst_core, Core), f"Expected {dst_core} to be a Core, got {type(dst_core)}"
                        self.model.addConstr(
                            self.y_path[p] <= self.x_tensor[(src_tensor, dst_core)],
                            name=f"path_core_link_src_empty_path_{tr.name}_{_resource_key(dst_core)}",
                        )

    def _add_destination_tensor_coherence_constraints(self, tr: TransferNode, paths: list[PathKey]) -> None:
        """Add constraints to ensure path choice is coherent with destination tensor placement."""
        dst_tensors = tr.outputs
        assert len(tr.inputs) == 1, "Only single-input transfers are supported for destination tensor coherence."
        src_tensor = tr.inputs[0]
        for dst_tensor in dst_tensors:
            if dst_tensor in self.tensor_var:
                assert isinstance(dst_tensor, Tensor), (
                    f"Expected {dst_tensor.name} to be a Tensor, got {type(dst_tensor)}"
                )
                for p in paths:
                    dst_core = self._path_dst_core(p[1])
                    if dst_core is not None:
                        if isinstance(dst_core, str):
                            continue  # dst_core is any core, no constraint needed
                        self.model.addConstr(
                            self.y_path[p] <= self.x_tensor[(dst_tensor, dst_core)],
                            name=f"path_core_link_dst_{tr.name}_{_resource_key(dst_core)}",
                        )
                        continue
                    # empty path is only possible if the dst_tensor is fixed on the same core as the src_tensor
                    src_tensor = src_tensor[0] if src_tensor else None
                    if src_tensor is not None and src_tensor in self.tensor_fixed:
                        src_core = self.resource_of[src_tensor]
                        assert isinstance(src_core, Core), f"Expected {src_core} to be a Core, got {type(src_core)}"
                        self.model.addConstr(
                            self.y_path[p] <= self.x_tensor[(dst_tensor, src_core)],
                            name=f"path_core_link_dst_empty_path_{tr.name}_{_resource_key(src_core)}",
                        )

    def _add_io_transfers_path_coherence_constraints(self, tr: TransferNode, paths: list[PathKey]) -> None:
        """
        For every path:
            if its FIRST link originates or ends in a memory core, enforce memory core matching:
            y_path ≤ m_store[(tr, memc)]
        """
        if not self._is_const_io(tr):
            return
        for p in paths:
            if not p[1]:
                continue  # empty path
            # Gather the mem cores involved in this path and make sure there's only one
            links = p[1]
            seen_mem_cores = set()
            for link in links:
                sender = link.sender
                receiver = link.receiver
                if sender in self.mem_cores:
                    seen_mem_cores.add(sender)
                if receiver in self.mem_cores:
                    seen_mem_cores.add(receiver)
            if len(seen_mem_cores) != 1:
                raise ValueError(f"Transfer {tr.node_name} doesn't have exactly one MemC in its path: {seen_mem_cores}")
            mem_core = seen_mem_cores.pop()
            self.model.addConstr(
                self.y_path[p] <= self.m_store[(tr, mem_core)],
                name=f"pathMemMatch_{tr.node_name}_{_resource_key(mem_core)}",
            )

    # ...................... link contention .................... #
    def _link_contention_constraints(self):
        # For every link/slot sum of all selected paths ≤ 1
        usage: dict[tuple[CommunicationLink, int], list[gp.Var]] = defaultdict(list)
        for (tr, path_t), y in self.y_path.items():
            s = self.slot_of[tr]
            for link in self.links_in_path[(tr, path_t)]:
                usage[(link, s)].append(y)

        for (link, s), vars_ in usage.items():
            self.model.addConstr(quicksum(vars_) <= 1, name=f"link_usage_{_resource_key(link)}_{s}")

    # ...................... memory capacity .................... #
    def _memory_capacity_constraints(self):
        self.core_load: dict[Core, gp.QuadExpr] = defaultdict(gp.QuadExpr)
        # add transfer tensors
        for tr in self.transfer_nodes:
            transfer_dims = self.workload.get_dims(tr)
            for t, cn in zip(tr.outputs, self.workload.successors(tr), strict=True):
                if isinstance(cn, OutEdge):
                    cn_tiling = tuple()
                else:
                    assert isinstance(cn, ComputationNode), f"Expected ComputationNode, got {type(cn)}"
                    cn_tiling = self.workload.get_unique_dims_inter_core_tiling(cn, self.mapping)
                partition_factor = prod(size for dim, size in cn_tiling if dim in transfer_dims)
                assert t in self.tensor_fixed, "Transfer tensors must be fixed (for now)."
                for c in self.possible_tensor_allocations[t]:
                    assert isinstance(c, Core), f"Expected {c} to be a Core, got {type(c)}"
                    for stop in range(-1, len(self.ssis[tr].get_temporal_variables())):
                        _, size_factor = self.reuse_levels[(tr, stop)]
                        req_size = ceil(size_factor * t.size_bits() / partition_factor)
                        self.core_load[c] += req_size * self.z_stopC[(tr, stop)]

        # add MemC load for transfers going through
        for tr in self.transfer_nodes:
            memory_allocation = self.mapping.get(tr).memory_allocation
            if not memory_allocation:
                continue
            for stop in range(-1, len(self.ssis[tr].get_temporal_variables())):
                _, size_factor = self.reuse_levels[(tr, stop)]
                assert len(tr.inputs) == 1, "Only single-input transfers are supported for memory capacity calculation."
                tensor = tr.inputs[0]
                req_size = size_factor * tensor.size_bits()  # full transfer size across all dsts
                for mc in memory_allocation:
                    self.core_load[mc] += req_size * self.m_store[(tr, mc)] * self.z_stopM[(tr, stop)]

        # add memory capacity constraints for each core
        for c, expr in self.core_load.items():
            cap = c.get_memory_capacity()  # user-provided helper
            self.model.addConstr(expr <= cap, name=f"mem_cap_{_resource_key(c)}")

    def _object_fifo_depth_constraints(self):
        """
        Ensure that the max FIFO depth of each core is respected.
        """
        self.object_fifo_depth: dict[Core, gp.LinExpr] = defaultdict(gp.LinExpr)
        for tr in self.transfer_nodes:
            resources = {c for t in tr.tensors for c in self.possible_tensor_allocations[t]}
            for c in resources:
                if c.id == self.offchip_core_id:
                    continue  # off-chip has no FIFO depth limit
                assert isinstance(c, Core), f"Expected {c} to be a Core, got {type(c)}"
                for stop in range(-1, len(self.ssis[tr].get_temporal_variables())):
                    tiles_needed = self.tiles_needed_levels[(tr, stop)]
                    tiles_needed += 1 if self.force_double_buffering else 0
                    self.object_fifo_depth[c] += tiles_needed * self.z_stopC[(tr, stop)]
            # Go through the possible memory cores too
            memory_allocation = self.mapping.get(tr).memory_allocation
            for mc in memory_allocation:
                if mc.id == self.offchip_core_id:
                    continue  # off-chip has no FIFO depth limit
                assert isinstance(mc, Core), f"Expected {mc} to be a Core, got {type(mc)}"
                for stop in range(-1, len(self.ssis[tr].get_temporal_variables())):
                    tiles_needed = self.tiles_needed_levels[(tr, stop)]
                    tiles_needed += 1 if self.force_double_buffering else 0
                    self.object_fifo_depth[mc] += tiles_needed * self.m_store[(tr, mc)] * self.z_stopM[(tr, stop)]
        self.context.add_object_fifo_constraints(self.model, self.object_fifo_depth)

    # ...................... slot latency ........................ #
    def _slot_latency_constraints(self):
        # constant SSC contributions
        for n in self.ssc_nodes:
            s = self.slot_of[n]
            runtimes = [self.cost_lut.get_cost(n, c).latency_total for c in self.cost_lut.get_cores(n)]
            runtime = ceil(max(runtimes)) if runtimes else 0
            self.model.addConstr(self.slot_latency[s] >= runtime, name=f"ssc_lat_{n.name}")

        # transfer (path-dependent) contributions
        for (tr, p), y in self.y_path.items():
            s = self.slot_of[tr]
            lat = self._transfer_latency(tr, p)
            self.model.addConstr(self.slot_latency[s] >= lat * y, name=f"tr_lat_{tr.name}_{hash(p)}")

    # ...................... overlap + objective ................. #
    def _overlap_and_objective(self) -> None:
        """
        ▸ idle_start[res,s]  == 1  ⇔  slot *s* is *before* the first activity on *res*
        ▸ idle_end  [res,s]  == 1  ⇔  slot *s* is *after*  the last  activity on *res*
        Only these slots contribute to the idle-latency that can overlap
        successive steady-state iterations.
        """
        max_s = self.max_slot
        big_m = self.big_m

        self._init_idle_indicators(max_s, big_m)
        self._create_idle_latency_vars(max_s)
        self._define_overlap_var()
        self._add_transfer_costs()
        self._add_dma_usage_constraints()
        self._set_total_latency_and_objective()

    # --------------------- overlap helpers --------------------- #
    def _first_last_busy_slot(self, res: Resource) -> tuple[int | None, int | None]:
        busy = sorted(self.slot_of[n] for n, r in self.resource_of.items() if r is res)
        return (busy[0], busy[-1]) if busy else (None, None)

    def _init_idle_indicators(self, max_s: int, big_m: int) -> None:
        self.idleS: dict[tuple[Resource, int], gp.Var | int] = {}
        self.idleE: dict[tuple[Resource, int], gp.Var | int] = {}
        # self._init_fixed_idle_indicators(max_s)
        self._init_link_idle_indicators(max_s, big_m)

    # def _init_fixed_idle_indicators(self, max_s: int) -> None:
    #     for res in self.tsa.resources:
    #         if res is None or (isinstance(res, Core) and res.id == self.offchip_core_id):
    #             continue
    #         first_busy, last_busy = self._first_last_busy_slot(res)
    #         if first_busy is None or last_busy is None:
    #             continue
    #         for s in range(max_s + 1):
    #             self.idleS[(res, s)] = 1 if s < first_busy else 0
    #             self.idleE[(res, s)] = 1 if s > last_busy else 0

    def _init_link_idle_indicators(self, max_s: int, big_m: int) -> None:
        self.link_used: dict[CommunicationLink, gp.Var] = {}
        self.prefixs: dict[CommunicationLink, list[gp.Var]] = {}
        self.suffixs: dict[CommunicationLink, list[gp.Var]] = {}
        for link in self.link_set:
            active_s: dict[int, gp.LinExpr] = {}
            for s in range(max_s + 1):
                active_s[s] = quicksum(
                    self.y_path[(tr, p)]
                    for (tr, p) in self.y_path
                    if link in self.links_in_path[(tr, p)] and self.slot_of[tr] == s
                )
            lu = self.model.addVar(vtype=GRB.BINARY, name=f"linkUsed_{_resource_key(link)}")
            self.link_used[link] = lu
            sum_active = quicksum(active_s.values())
            self.model.addConstr(sum_active >= lu, name=f"link_used_def_{_resource_key(link)}")
            self.model.addConstr(sum_active <= big_m * lu, name=f"link_used_def2_{_resource_key(link)}")

            prefix = [
                self.model.addVar(vtype=GRB.INTEGER, name=f"pre_{_resource_key(link)}_{s}") for s in range(max_s + 1)
            ]
            suffix = [
                self.model.addVar(vtype=GRB.INTEGER, name=f"suf_{_resource_key(link)}_{s}") for s in range(max_s + 1)
            ]
            self.prefixs[link] = prefix
            self.suffixs[link] = suffix
            self.model.addConstr(prefix[0] == active_s[0])
            self.model.addConstr(suffix[-1] == active_s[max_s])
            for s in range(1, max_s + 1):
                self.model.addConstr(prefix[s] == prefix[s - 1] + active_s[s])
                self.model.addConstr(suffix[max_s - s] == suffix[max_s - s + 1] + active_s[max_s - s])

            for s in range(max_s + 1):
                is_ = self.model.addVar(vtype=GRB.BINARY, name=f"idleS_{_resource_key(link)}_{s}")
                ie_ = self.model.addVar(vtype=GRB.BINARY, name=f"idleE_{_resource_key(link)}_{s}")
                self.idleS[(link, s)] = is_
                self.idleE[(link, s)] = ie_

                self.model.addConstr(prefix[s] <= big_m * (1 - is_))
                self.model.addConstr(prefix[s] >= lu - big_m * is_)
                self.model.addConstr(suffix[s] <= big_m * (1 - ie_))
                self.model.addConstr(suffix[s] >= lu - big_m * ie_)
                self.model.addConstr(is_ >= 1 - lu)
                self.model.addConstr(ie_ <= lu)

    def _create_idle_latency_vars(self, max_s: int) -> None:
        self.idle_lat: dict[Resource, gp.Var] = {}
        for res in {r for r, _ in self.idleS} | {r for r, _ in self.idleE}:
            expr = quicksum(
                self.idleS.get((res, s), 0) * self.slot_latency[s] + self.idleE.get((res, s), 0) * self.slot_latency[s]
                for s in range(max_s + 1)
            )
            v = self.model.addVar(vtype=GRB.INTEGER, name=f"idleLat_{_resource_key(res)}")
            self.model.addConstr(v == expr)
            self.idle_lat[res] = v

    def _define_overlap_var(self) -> None:
        overlap = self.model.addVar(vtype=GRB.INTEGER, name="overlap")
        self.overlap = overlap
        for v in self.idle_lat.values():
            self.model.addConstr(overlap <= v)

    def _add_transfer_costs(self) -> None:
        self.total_transfer_cost = quicksum(
            self._transfer_latency(tr, p) * self.y_path[(tr, p)] * self.firesC[tr] for (tr, p) in self.y_path
        )
        lat_dram_mem = {
            (tr, mc): ceil(tr.inputs[0].size_bits() / self._first_link_bw_from_dram(tr, mc))
            for tr in self.transfer_nodes
            if self._is_const_io(tr)
            for mc in self.mapping.get(tr).memory_allocation
        }
        self.total_transfer_cost += quicksum(
            lat_dram_mem[(tr, mc)] * self.firesM[tr] * self.m_store[(tr, mc)]
            for tr in self.transfer_nodes
            if self._is_const_io(tr)
            for mc in self.mapping.get(tr).memory_allocation
        )

    def _add_dma_usage_constraints(self) -> None:
        self.mem_core_usage_s2mm: dict[Core, gp.Var] = {}
        self.mem_core_usage_mm2s: dict[Core, gp.Var] = {}
        self.shim_core_usage_s2mm: dict[Core, gp.Var] = {}
        self.shim_core_usage_mm2s: dict[Core, gp.Var] = {}
        for mc in self.mem_cores:
            mem_core_usage_s2mm = self.model.addVar(vtype=GRB.INTEGER, name=f"memCoreUsageS2MM_{_resource_key(mc)}")
            self.model.addConstr(
                mem_core_usage_s2mm
                == quicksum(self.m_store[(tr, mc)] for tr in self.transfer_nodes if self._is_const_io(tr)),
                name=f"memCoreUsageS2MMConstr_{_resource_key(mc)}",
            )
            self.mem_core_usage_s2mm[mc] = mem_core_usage_s2mm

            mem_core_usage_mm2s = self.model.addVar(vtype=GRB.INTEGER, name=f"memCoreUsageMM2S_{_resource_key(mc)}")
            self.model.addConstr(
                mem_core_usage_mm2s
                == quicksum(self.m_store[(tr, mc)] for tr in self.transfer_nodes if self._is_const_io(tr)),
                name=f"memCoreUsageMM2SConstr_{_resource_key(mc)}",
            )
            self.mem_core_usage_mm2s[mc] = mem_core_usage_mm2s

            shim_core_usage_s2mm = self.model.addVar(vtype=GRB.INTEGER, name=f"shimCoreUsageS2MM_{_resource_key(mc)}")
            self.model.addConstr(
                shim_core_usage_s2mm
                == quicksum(self.m_store[(tr, mc)] for tr in self.transfer_nodes if self._is_const_o(tr)),
                name=f"shimCoreUsageS2MMConstr_{_resource_key(mc)}",
            )
            self.shim_core_usage_s2mm[mc] = shim_core_usage_s2mm

            shim_core_usage_mm2s = self.model.addVar(vtype=GRB.INTEGER, name=f"shimCoreUsageMM2S_{_resource_key(mc)}")
            self.model.addConstr(
                shim_core_usage_mm2s
                == quicksum(self.m_store[(tr, mc)] for tr in self.transfer_nodes if self._is_const_i(tr)),
                name=f"shimCoreUsageMM2SConstr_{_resource_key(mc)}",
            )
            self.shim_core_usage_mm2s[mc] = shim_core_usage_mm2s

        self.max_mem_core_usage, self.max_shim_core_usage = self.context.add_dma_usage_constraints(
            self.model,
            self.mem_core_usage_s2mm,
            self.mem_core_usage_mm2s,
            self.shim_core_usage_s2mm,
            self.shim_core_usage_mm2s,
        )

    def _set_total_latency_and_objective(self) -> None:
        self.total_lat = self.model.addVar(vtype=GRB.INTEGER, name="total_latency")
        self.total_latency = self.total_lat
        assert self.overlap is not None, "Overlap variable must be initialized before objective."
        self.model.addConstr(
            self.total_lat
            == self.iterations * quicksum(self.slot_latency.values()) - (self.iterations - 1) * self.overlap
        )
        obj_func = self.total_lat + self.total_transfer_cost + self.max_mem_core_usage
        self.model.setObjective(obj_func, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # public solve()                                                     #
    # ------------------------------------------------------------------ #
    def solve(self, *, tee: bool = True) -> tuple[TensorAlloc, TransferAlloc, MemoryAlloc, int, int, int]:  # noqa: PLR0912
        self.model.setParam("OutputFlag", 1 if tee else 0)
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            self.model.computeIIS()
            self.model.write("model.ilp")
            raise RuntimeError("Gurobi did not find an optimal solution. IIS written to model.ilp")

        # ---------- sanity checks -------------------
        self._check_io_transfers_firing_levels()

        # ---------- read back decisions --------------------------------
        tensor_alloc = self.get_tensor_allocations()
        routing = self.get_transfer_routing()
        chosen_memory_cores = self.get_chosen_memory_cores()
        self.update_transfer_reuse_levels()

        assert self.total_latency is not None, "Total latency variable was not created."
        total_latency = int(self.total_latency.X)
        overlap = int(self.overlap.X)
        latency_per_iteration = sum(slot_lat.X for slot_lat in self.slot_latency.values())
        return tensor_alloc, routing, chosen_memory_cores, total_latency, overlap, latency_per_iteration

    def update_transfer_reuse_levels(  # noqa: PLR0912
        self,
    ) -> None:
        compute_tile_reuse_levels: dict[TransferNode, int] = {}
        mem_tile_reuse_levels: dict[TransferNode, int] = {}
        for tr in self.transfer_nodes:
            for stop in range(-1, len(self.ssis[tr].get_temporal_variables())):
                if self.z_stopC[(tr, stop)].X > self.VAR_THRESHOLD:
                    compute_tile_reuse_levels[tr] = stop
                if self.z_stopM[(tr, stop)].X > self.VAR_THRESHOLD:
                    mem_tile_reuse_levels[tr] = stop
        # Update the steady state iteration space for transfers
        for tr in self.transfer_nodes:
            stop = compute_tile_reuse_levels[tr]
            # Set all iteration variables to REUSE flag
            for i, iter_var in enumerate(self.ssis[tr].get_temporal_variables()):
                if i <= stop:
                    flag = ComputeTileReuse.REUSE
                else:
                    flag = ComputeTileReuse.NO_REUSE
                iter_var.compute_tile_reuse = flag
            for i, iter_var in enumerate(self.ssis[tr].get_temporal_variables()):
                if i <= mem_tile_reuse_levels[tr]:
                    flag = MemTileReuse.REUSE
                else:
                    flag = MemTileReuse.NO_REUSE
                iter_var.mem_tile_reuse |= flag  # append the memory reuse flag
        # # Print the updated reuse levels for debugging
        # for tr in self.transfer_nodes:
        #     compute_stop = compute_tile_reuse_levels.get(tr, -1)
        #     mem_stop = mem_tile_reuse_levels.get(tr, -1)
        #     print(f"Transfer {tr.node_name}: Compute Stop = {compute_stop}, Mem Stop = {mem_stop}")
        #     for i, iter_var in enumerate(tr.steady_state_iteration_space.get_temporal_variables()):
        #         print(f"  Iter {i}: Reuse = {iter_var.reuse.name}, Size = {iter_var.size}")

    def get_transfer_routing(
        self,
    ) -> dict[TransferNode, tuple[CommunicationLink]]:
        routing: dict[TransferNode, tuple[CommunicationLink]] = {}
        for tr in self.transfer_nodes:
            chosen = [
                p for p in self.possible_transfer_allocations[tr] if self.y_path[(tr, tuple(p))].X > self.VAR_THRESHOLD
            ]
            if len(chosen) != 1:
                raise ValueError(f"{tr.name}: expected exactly one path, got {chosen}")
            path = chosen[0]
            routing[tr] = path
        return routing

    def get_chosen_memory_cores(
        self,
    ) -> dict[TransferNode, Core]:
        # --- which MemC was selected
        chosen_memory_cores: dict[TransferNode, Core] = {}
        for tr in self.transfer_nodes:
            for mc in self.mem_cores:
                if self.m_store.get((tr, mc), None) and self.m_store[(tr, mc)].X > self.VAR_THRESHOLD:
                    chosen_memory_cores[tr] = mc
                    break
        return chosen_memory_cores

    def update_transfer_memory_core_allocation(
        self,
    ) -> None:
        chosen_memory_cores = self.get_chosen_memory_cores()
        for tr, core in chosen_memory_cores.items():
            assert isinstance(core, Core), f"Expected {core} to be a Core, got {type(core)}"
            tr.chosen_memory_core = core

    def get_tensor_allocations(
        self,
    ) -> dict[Tensor, Core]:
        tensor_alloc: dict[Tensor, Core] = {}
        for t in self.tensor_var:
            chosen = [c for c in self.possible_tensor_allocations[t] if self.x_tensor[(t, c)].X > self.VAR_THRESHOLD]
            if len(chosen) != 1:
                raise ValueError(f"{t.node_name}: expected exactly one core, got {chosen}")
            core = chosen[0]
            t.chosen_resource_allocation = core
            tensor_alloc[t] = core
        return tensor_alloc

    def get_updated_steady_state_workload(self) -> SteadyStateWorkload:
        """Return a new SteadyStateWorkload with updated resource allocations.
        This is necessary as there's a bug in networkx that doesn't update the edges when node attributes change.
        """
        ssw_upd = SteadyStateWorkload()
        for edge in self.workload.edges(data=True):
            src, dst, data = edge
            assert src in self.workload.node_list, f"Source {src} not found in ssw nodes."
            assert dst in self.workload.node_list, f"Destination {dst} not found in ssw nodes."
            src = self.workload.node_list[self.workload.node_list.index(src)]
            dst = self.workload.node_list[self.workload.node_list.index(dst)]
            ssw_upd.add_edge(src, dst, **data)
        return ssw_upd

    def _check_io_transfers_firing_levels(self) -> None:
        # firesC should always be greater or equal to firesM
        for tr in self.transfer_nodes:
            assert self.firesC[tr].X >= self.firesM[tr].X - 1e-6
        # chosen stopM ≥ stopC
        for tr in self.transfer_nodes:
            stop_max = len(self.ssis[tr].get_temporal_variables())
            stopC = next(s for s in range(-1, stop_max) if self.z_stopC[(tr, s)].X > self.VAR_THRESHOLD)
            stopM = next(s for s in range(-1, stop_max) if self.z_stopM[(tr, s)].X > self.VAR_THRESHOLD)
            assert stopM >= stopC

    def _eval_and_print_linexpr(self, expr, sol=None):
        """
        Evaluate and pretty-print a Gurobi linear expression using size()/getVar(i)/getCoeff(i),
        with each term on its own line.

        Parameters
        ----------
        expr : gurobipy.LinExpr
            The linear expression to evaluate.
        sol : dict, optional
            Mapping from variable -> value. If None, use var.X (solution value).

        Returns
        -------
        float
            Evaluated numeric value of the expression.
        """
        val = expr.getConstant()

        print("Expression terms:")
        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)
            x = sol[var] if sol is not None else var.X
            term_val = coeff * x
            val += term_val
            print(f"  {coeff} * {var.VarName} (value={x:.4f}) -> {term_val:.4f}")

        if expr.getConstant() != 0:
            print(f"  constant {expr.getConstant()}")

        print(f"Total = {val:.4f}")
        return val
