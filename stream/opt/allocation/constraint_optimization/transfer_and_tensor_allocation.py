# transfer_and_tensor_allocator.py
import math
from collections import defaultdict
from math import ceil
from typing import Any

import gurobipy as gp
from gurobipy import GRB, quicksum

from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.opt.allocation.constraint_optimization.timeslot_allocation import (
    Resource,
    TimeSlotAllocation,
    _resource_key,
)
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.iteration_space import ComputeTileReuse, MemTileReuse
from stream.workload.steady_state.node import SteadyStateNode
from stream.workload.steady_state.tensor import SteadyStateTensor, TensorFlag
from stream.workload.steady_state.transfer import SteadyStateTransfer
from stream.workload.steady_state.workload import SteadyStateWorkload


class TransferAndTensorAllocator:
    """
    MILP that decides

    1. **where** every *movable* SteadyStateTensor lives,
    2. **which path** each SteadyStateTransfer uses,

    while all slots/resources coming from an existing TimeSlotAllocation
    remain unchanged.
    """

    VAR_THRESHOLD = 0.5  # threshold to decide if a binary variable is chosen

    # ------------------------------------------------------------ #
    # ctor / public API                                            #
    # ------------------------------------------------------------ #
    def __init__(
        self,
        ssw: SteadyStateWorkload,
        tsa: TimeSlotAllocation,
        accelerator: Accelerator,
        *,
        iterations: int = 1,
        big_m: int | None = None,
        gurobi_verbosity: int = 1,
        nb_cols_to_use: int = 4,
    ):
        self.ssw = ssw
        self.tsa = tsa
        self.accelerator = accelerator
        self.offchip_core_id = self.accelerator.offchip_core_id
        self.iterations = iterations
        self.max_slot = tsa.slot_max
        self.big_m = big_m or len(ssw.nodes()) + 5
        self.force_io_transfers_on_mem_tile = True

        # ------------------- categorise nodes -------------------- #
        self.ssc_nodes: list[SteadyStateComputation] = ssw.computation_nodes
        self.transfer_nodes: list[SteadyStateTransfer] = ssw.transfer_nodes

        # Fixed-vs-movable tensors
        self.tensor_fixed: list[SteadyStateTensor] = []
        self.tensor_var: list[SteadyStateTensor] = []
        for t in ssw.tensor_nodes:
            if len(t.possible_resource_allocation or []) <= 1 or t.chosen_resource_allocation is not None:
                self.tensor_fixed.append(t)
            else:
                self.tensor_var.append(t)
        assert not self.tensor_var, "Variable tensors are not supported since transfer firing changes."

        # quick look-ups from TSA
        self.slot_of: dict[Any, int] = {n: tsa.get_timeslot_of_node(n) for n in tsa.nodes}
        self.resource_of: dict[Any, Resource] = {n: next(iter(tsa.get_resources_for_node(n))) for n in tsa.nodes}

        # ------------------------------------------------------------------------------
        # memory cores that may act as on‑chip caches (exclude the DRAM/off‑chip id)
        # ------------------------------------------------------------------------------
        self.MAX_NB_COLS_TO_USE = nb_cols_to_use
        self.mem_cores: list[Core] = [
            c
            for c in self.accelerator.core_list
            if isinstance(c, Core)
            and c.id != self.offchip_core_id
            and c.type == "memory"
            and c.col_id is not None
            and c.col_id < self.MAX_NB_COLS_TO_USE
        ]

        # --------------- optimisation model ---------------------- #
        self.model = gp.Model("transfer_tensor_alloc")
        self.model.setParam("OutputFlag", gurobi_verbosity)

        # decision vars
        self.x_tensor: dict[tuple[SteadyStateTensor, Core], gp.Var] = {}
        self.y_path: dict[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]], gp.Var] = {}

        # helpers
        self.link_set: set[CommunicationLink] = set()
        self.links_in_path: dict[
            tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]],
            list[CommunicationLink],
        ] = {}

        # latency vars
        self.slot_latency: dict[int, gp.Var] = {}
        self.overlap = None
        self.total_latency = None

        # transfer fire helpers init
        # dict((transfer, steady state iteration space index): (fires_across_ss_iterations, extra_mem_bytes))
        self._ensure_same_ssis_for_all_transfers()
        self.reuse_levels: dict[tuple[SteadyStateTransfer, int], tuple[int, int]] = {}
        self.tiles_needed_levels: dict[tuple[SteadyStateTransfer, int], int] = {}
        self.transfer_nodes_to_optimize_firings_for: list[SteadyStateTransfer] = []
        self._init_transfer_fire_helpers()

        self._build_model()

    # ------------------------------------------------------------ #
    # internal helpers                                             #
    # ------------------------------------------------------------ #
    # ---------- memory factor ( NO ping-pong for now ) ---------- #
    # TODO: Update this for memory-only cores
    @staticmethod
    def _mem_factor(t: SteadyStateTensor, core: Core) -> int:
        # if getattr(core, "type", None) == "COMPUTE":
        return 1  # NO ping–pong
        # else:
        #     full_fac = getattr(t, "slices_per_full", 1)
        #     return int(full_fac)

    # ---------- latency of a transfer along a path -------------- #
    @staticmethod
    def _transfer_latency(tr: SteadyStateTransfer, path: tuple[CommunicationLink, ...]) -> int:
        if not path:
            return 0
        min_bw = min(link.bandwidth for link in path)
        return ceil(tr.size / min_bw)

    # ---------- init transfer fire helpers ----------------------- #
    def _ensure_same_ssis_for_all_transfers(self) -> None:
        """
        Ensure that all transfers have the same steady-state iteration space dimensions and sizes (SSIS).
        """
        first_transfer_ssis = self.transfer_nodes[0].steady_state_iteration_space
        first_transfer_ssis_dims = first_transfer_ssis.get_temporal_dimensions()
        first_transfer_ssis_sizes = first_transfer_ssis.get_temporal_sizes()
        for tr in self.transfer_nodes:
            transfer_ssis = tr.steady_state_iteration_space
            transfer_ssis_dims = transfer_ssis.get_temporal_dimensions()
            transfer_ssis_sizes = transfer_ssis.get_temporal_sizes()
            if not (
                transfer_ssis_dims == first_transfer_ssis_dims and transfer_ssis_sizes == first_transfer_ssis_sizes
            ):
                raise ValueError(
                    f"Transfer {tr.node_name} has different SSIS dims and sizes than the {self.transfer_nodes[0]}: "
                    f"{transfer_ssis_dims}, {transfer_ssis_sizes} != "
                    f"{first_transfer_ssis_dims}, {first_transfer_ssis_sizes}"
                )

    def _init_transfer_fire_helpers(self) -> None:
        for tr in self.transfer_nodes:  # only the movable tensors
            ssis = tr.steady_state_iteration_space.get_temporal_variables()  # e.g. [Nk, Nk-1, …, N0]
            sizes = [iter_var.size for iter_var in ssis]
            relevancies = [iter_var.relevant for iter_var in ssis]
            reuses = [iter_var.compute_tile_reuse for iter_var in ssis]

            # Check that all compute reuses are NOT_SET, else continue
            if any(r != ComputeTileReuse.NOT_SET for r in reuses):
                continue
            self.transfer_nodes_to_optimize_firings_for.append(tr)

            # level = -1  →  "transfer every steady state iteration"
            fires = math.prod(sizes)
            mem_needed = tr.size
            self.reuse_levels[(tr, -1)] = (fires, mem_needed)
            tiles_needed = 1
            self.tiles_needed_levels[(tr, -1)] = tiles_needed
            # level = i  →  keep while loops 0..i stay in cache
            for i, (Nl, relevancy) in enumerate(zip(sizes, relevancies, strict=True)):  # i = 0 … K-1
                mem_needed *= Nl if relevancy else 1  # enlarge tile size factor only if relevant
                tiles_needed *= Nl if relevancy else 1
                fires //= Nl  # fewer transfers
                self.reuse_levels[(tr, i)] = (fires, mem_needed)
                self.tiles_needed_levels[(tr, i)] = tiles_needed

    # ------------------------------------------------------------------------------
    # only transfers whose src OR dst tensor is CONSTANT are eligible for MemC
    # ------------------------------------------------------------------------------
    def _is_const_io(self, tr: SteadyStateTransfer) -> bool:
        src_t = next(iter(self.ssw.predecessors(tr)), None)
        dst_t = next(iter(self.ssw.successors(tr)), None)
        return src_t and TensorFlag.CONSTANT in src_t.tensor_flag or dst_t and TensorFlag.CONSTANT in dst_t.tensor_flag

    def _any_path_through_mc(self, tr: SteadyStateTransfer, mc: Core) -> bool:
        for path in tr.possible_resource_allocation:
            if any(mc.id == link.receiver.id for link in path):
                return True
        return False

    # ------------------------------------------------------------
    # bandwidth of the FIRST link on a DRAM → mem‑core path
    # ------------------------------------------------------------
    def _first_link_bw_from_dram(
        self,
        tr: SteadyStateTransfer,
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

    def __create_slot_latency_vars(self):
        for s in range(self.max_slot + 1):
            self.slot_latency[s] = self.model.addVar(vtype=GRB.INTEGER, name=f"L_{s}")

    def __create_compute_core_reuse_vars(self):
        self.z_stopC: dict[tuple[SteadyStateTransfer, int], gp.Var] = {}
        for tr in self.transfer_nodes:
            sizes = tr.steady_state_iteration_space.get_temporal_sizes()
            for stop in range(-1, len(sizes)):  # -1 .. K-1
                v = self.model.addVar(vtype=GRB.BINARY, name=f"zStopC_{tr.node_name}_L{stop}")
                self.z_stopC[(tr, stop)] = v
            # Choose exactly one stop-level
            self.model.addConstr(
                quicksum(self.z_stopC[(tr, s)] for s in range(-1, len(sizes))) == 1,
                name=f"zStopC_Choose_One_{tr.node_name}",
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
        self.z_stopM: dict[tuple[SteadyStateTransfer, int], gp.Var] = {}

        for tr in self.transfer_nodes:
            sizes = tr.steady_state_iteration_space.get_temporal_sizes()
            for stop in range(-1, len(sizes)):
                v = self.model.addVar(vtype=GRB.BINARY, name=f"zStopM_{tr.node_name}_L{stop}")
                self.z_stopM[(tr, stop)] = v
            self.model.addConstr(
                quicksum(self.z_stopM[(tr, s)] for s in range(-1, len(sizes))) == 1,
                name=f"cacheChooseStopM_{tr.node_name}",
            )
            # If any of the reuses is already set to NO_REUSE, enforce these levels to 0
            mem_tile_reuses = tr.steady_state_iteration_space.get_temporal_mem_tile_reuses()
            for i, r in enumerate(mem_tile_reuses):
                if r == MemTileReuse.NO_REUSE:
                    self.model.addConstr(
                        self.z_stopM[(tr, i)] == 0,
                        name=f"zStopM_NoReuse_{tr.node_name}_L{i}",
                    )

            # stopM ≥ stopC (every cached loop for Compute must also be cached in the MemC)
            for s in range(-1, len(sizes)):
                cumC = quicksum(self.z_stopC[(tr, u)] for u in range(s, len(sizes)))
                cumM = quicksum(self.z_stopM[(tr, u)] for u in range(s, len(sizes)))
                self.model.addConstr(cumC <= cumM, name=f"nest_{tr.node_name}_L{s}")

    def __create_transfer_mem_core_vars(self):
        self.m_store: dict[tuple[SteadyStateTransfer, Core], gp.Var] = {}
        if not self.mem_cores:
            return  # no memory cores, nothing to do
        for tr in self.transfer_nodes:
            if not self._is_const_io(tr):
                continue
            possible_mem_cores = []
            for mc in self.mem_cores:
                v = self.model.addVar(vtype=GRB.BINARY, name=f"mStore_{tr.node_name}_{_resource_key(mc)}")
                self.m_store[(tr, mc)] = v
                possible_mem_cores.append(mc)
            if not self.force_io_transfers_on_mem_tile:
                v_none = self.model.addVar(vtype=GRB.BINARY, name=f"mStore_{tr.node_name}_NONE")
                self.m_store[(tr, None)] = v_none
                possible_mem_cores.append(None)
            self.model.addConstr(
                quicksum(self.m_store[(tr, c)] for c in possible_mem_cores) == 1, name=f"chooseMemCore_{tr.node_name}"
            )

    def __create_transfer_path_vars(self):
        for tr in self.transfer_nodes:
            for p in tr.possible_resource_allocation:  # list[list[Link]]
                p_tuple = tuple(p)
                v = self.model.addVar(vtype=GRB.BINARY, name=f"y_{tr.node_name}_{hash(p_tuple)}")
                self.y_path[(tr, p_tuple)] = v
                self.links_in_path[(tr, p_tuple)] = list(p_tuple)
                self.link_set.update(p_tuple)

    def __create_tensor_placement_vars(self):
        for t in self.tensor_var:
            for c in t.possible_resource_allocation:  # type: ignore
                v = self.model.addVar(vtype=GRB.BINARY, name=f"x_{t.node_name}_{_resource_key(c)}")
                self.x_tensor[(t, c)] = v

    # ...................... tensor placement .................... #
    def _tensor_placement_constraints(self):
        for t in self.tensor_var:
            self.model.addConstr(
                quicksum(self.x_tensor[(t, c)] for c in t.possible_resource_allocation) == 1,  # type: ignore
                name=f"place_{t.node_name}",
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
            fires_c = self.model.addVar(vtype=GRB.INTEGER, name=f"firesC_{tr.node_name}")
            fires_m = self.model.addVar(vtype=GRB.INTEGER, name=f"firesM_{tr.node_name}")
            self.firesC[tr], self.firesM[tr] = fires_c, fires_m

            self.model.addConstr(
                fires_c
                == quicksum(
                    self.reuse_levels[(tr, s)][0] * self.z_stopC[(tr, s)]
                    for s in range(-1, len(tr.steady_state_iteration_space.get_temporal_variables()))
                ),
                name=f"firesC_def_{tr.node_name}",
            )
            self.model.addConstr(
                fires_m
                == quicksum(
                    self.reuse_levels[(tr, s)][0] * self.z_stopM[(tr, s)]
                    for s in range(-1, len(tr.steady_state_iteration_space.get_temporal_variables()))
                ),
                name=f"firesM_def_{tr.node_name}",
            )

    # ...................... path choice ........................ #
    def _path_choice_constraints(self) -> None:
        """
        For every transfer, exactly one path must be selected and path choices must be coherent
        with tensor placements for both source and destination tensors.
        """
        for tr in self.transfer_nodes:
            paths = [p for p in self.y_path if p[0] is tr]
            self._add_one_path_constraint(tr, paths)
            self._add_source_tensor_coherence_constraints(tr, paths)
            self._add_destination_tensor_coherence_constraints(tr, paths)
            self._add_io_transfers_path_coherence_constraints(tr, paths)

    def _add_one_path_constraint(
        self, tr: SteadyStateTransfer, paths: list[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]]]
    ) -> None:
        """Ensure exactly one path is selected for each transfer."""
        self.model.addConstr(quicksum(self.y_path[p] for p in paths) == 1, name=f"one_path_{tr.node_name}")

    def _add_source_tensor_coherence_constraints(
        self, tr: SteadyStateTransfer, paths: list[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]]]
    ) -> None:
        """Add constraints to ensure path choice is coherent with source tensor placement."""
        predecessors = list(self.ssw.predecessors(tr))
        assert all(isinstance(n, SteadyStateTensor) for n in predecessors), (
            f"Transfer {tr.node_name} has non-tensor predecessor(s): {predecessors}"
        )
        successors = list(self.ssw.successors(tr))
        for src_tensor in predecessors:
            if src_tensor in self.tensor_var:
                assert isinstance(src_tensor, SteadyStateTensor), (
                    f"Expected {src_tensor.node_name} to be a SteadyStateTensor, got {type(src_tensor)}"
                )
                for p in paths:
                    if p[1]:
                        src_core = p[1][0].sender  # first link’s source core
                        assert isinstance(src_core, Core), f"Expected {src_core} to be a Core, got {type(src_core)}"
                        self.model.addConstr(
                            self.y_path[p] <= self.x_tensor[(src_tensor, src_core)],
                            name=f"path_core_link_src_{tr.node_name}_{_resource_key(src_core)}",
                        )
                    else:
                        # empty path is only possible if the src_tensor is fixed on the same core as the dst_tensor
                        dst_tensor = successors[0] if successors else None
                        if dst_tensor is not None and dst_tensor in self.tensor_fixed:
                            dst_core = self.resource_of[dst_tensor]
                            assert isinstance(dst_core, Core), f"Expected {dst_core} to be a Core, got {type(dst_core)}"
                            self.model.addConstr(
                                self.y_path[p] <= self.x_tensor[(src_tensor, dst_core)],
                                name=f"path_core_link_src_empty_path_{tr.node_name}_{_resource_key(dst_core)}",
                            )

    def _add_destination_tensor_coherence_constraints(
        self, tr: SteadyStateTransfer, paths: list[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]]]
    ) -> None:
        """Add constraints to ensure path choice is coherent with destination tensor placement."""
        successors = list(self.ssw.successors(tr))
        assert all(isinstance(n, SteadyStateTensor) for n in successors), (
            f"Transfer {tr.node_name} has non-tensor successor(s): {successors}"
        )
        predecessors = list(self.ssw.predecessors(tr))
        for dst_tensor in successors:
            if dst_tensor in self.tensor_var:
                assert isinstance(dst_tensor, SteadyStateTensor), (
                    f"Expected {dst_tensor.node_name} to be a SteadyStateTensor, got {type(dst_tensor)}"
                )
                for p in paths:
                    if p[1]:
                        dst_core = p[1][-1].receiver  # last link’s destination core
                        if isinstance(dst_core, str):
                            continue  # dst_core is any core, no constraint needed
                        self.model.addConstr(
                            self.y_path[p] <= self.x_tensor[(dst_tensor, dst_core)],
                            name=f"path_core_link_dst_{tr.node_name}_{_resource_key(dst_core)}",
                        )
                    else:
                        # empty path is only possible if the dst_tensor is fixed on the same core as the src_tensor
                        src_tensor = predecessors[0] if predecessors else None
                        if src_tensor is not None and src_tensor in self.tensor_fixed:
                            src_core = self.resource_of[src_tensor]
                            assert isinstance(src_core, Core), f"Expected {src_core} to be a Core, got {type(src_core)}"
                            self.model.addConstr(
                                self.y_path[p] <= self.x_tensor[(dst_tensor, src_core)],
                                name=f"path_core_link_dst_empty_path_{tr.node_name}_{_resource_key(src_core)}",
                            )

    def _add_io_transfers_path_coherence_constraints(
        self, tr: SteadyStateTransfer, paths: list[tuple[SteadyStateTransfer, tuple[CommunicationLink, ...]]]
    ) -> None:
        """
        For every path:
            if its FIRST link originates in a memory core, enforce memory core matching:
            y_path ≤ m_store[(tr, memc)]
        """
        for p in paths:
            if not p[1]:
                continue  # empty path
            src_core = p[1][0].sender
            if isinstance(src_core, Core) and src_core in self.mem_cores:
                self.model.addConstr(
                    self.y_path[p] <= self.m_store[(tr, src_core)],
                    name=f"pathMemMatch_{tr.node_name}_{_resource_key(src_core)}",
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
        # start with fixed tensors
        self.core_load: dict[Core, gp.QuadExpr] = defaultdict(gp.QuadExpr)
        for t in self.tensor_fixed:
            core = self.resource_of[t]
            if isinstance(core, Core):
                self.core_load[core] += self._mem_factor(t, core) * t.size  # type: ignore

        # add variable tensors
        for t in self.tensor_var:
            for c in t.possible_resource_allocation:  # type: ignore
                self.core_load[c] += self._mem_factor(t, c) * t.size * self.x_tensor[(t, c)]  # type: ignore

        # add transfer tensors
        for tr in self.transfer_nodes:
            for t in self.ssw.successors(tr):
                assert isinstance(t, SteadyStateTensor), (
                    f"Expected {t.node_name} to be a SteadyStateTensor, got {type(t)}"
                )
                assert t in self.tensor_fixed, "Post transfer tensors must be fixed (for now)."
                c = self.resource_of[t]
                assert isinstance(c, Core), f"Expected {c} to be a Core, got {type(c)}"
                for stop in range(-1, len(tr.steady_state_iteration_space.get_temporal_variables())):
                    _, size_factor = self.reuse_levels[(tr, stop)]
                    self.core_load[c] += size_factor * self.z_stopC[(tr, stop)]

        # add MemC load for transfers going through
        for tr in self.transfer_nodes:
            if not self._is_const_io(tr):
                continue
            for stop in range(-1, len(tr.steady_state_iteration_space.get_temporal_variables())):
                _, mem_need = self.reuse_levels[(tr, stop)]
                for mc in self.mem_cores:
                    self.core_load[mc] += mem_need * self.m_store[(tr, mc)] * self.z_stopM[(tr, stop)]

        # add memory capacity constraints for each core
        for c, expr in self.core_load.items():
            cap = c.get_memory_capacity()  # user-provided helper
            self.model.addConstr(expr <= cap, name=f"mem_cap_{_resource_key(c)}")

    def _object_fifo_depth_constraints(self):
        """
        Ensure that the FIFO depth of each object is respected.
        This is a placeholder for future implementation.
        """
        self.object_fifo_depth: dict[Core, gp.LinExpr] = defaultdict(gp.LinExpr)
        for tr in self.transfer_nodes:
            preds_and_succs = list(self.ssw.predecessors(tr)) + list(self.ssw.successors(tr))
            resources = {self.resource_of[t] for t in preds_and_succs if isinstance(t, SteadyStateTensor)}
            for c in resources:
                assert isinstance(c, Core), f"Expected {c} to be a Core, got {type(c)}"
                for stop in range(-1, len(tr.steady_state_iteration_space.get_temporal_variables())):
                    tiles_needed = self.tiles_needed_levels[(tr, stop)]
                    self.object_fifo_depth[c] += tiles_needed * self.z_stopC[(tr, stop)]
        for c, expr in self.object_fifo_depth.items():
            max_fifo_depth = c.get_max_object_fifo_depth()
            self.model.addConstr(expr <= max_fifo_depth, name=f"obj_fifo_depth_{_resource_key(c)}")

    # ...................... slot latency ........................ #
    def _slot_latency_constraints(self):
        # constant SSC contributions
        for n in self.ssc_nodes:
            s = self.slot_of[n]
            assert n.runtime is not None, f"Node {n.node_name} has no runtime defined."
            self.model.addConstr(self.slot_latency[s] >= n.runtime, name=f"ssc_lat_{n.node_name}")

        # transfer (path-dependent) contributions
        for (tr, p), y in self.y_path.items():
            s = self.slot_of[tr]
            lat = self._transfer_latency(tr, p)
            self.model.addConstr(self.slot_latency[s] >= lat * y, name=f"tr_lat_{tr.node_name}_{hash(p)}")

    # ...................... overlap + objective ................. #
    def _overlap_and_objective(self) -> None:  # noqa: PLR0915
        """
        ▸ idle_start[res,s]  == 1  ⇔  slot *s* is *before* the first activity on *res*
        ▸ idle_end  [res,s]  == 1  ⇔  slot *s* is *after*  the last  activity on *res*
        Only these slots contribute to the idle-latency that can overlap
        successive steady-state iterations.
        """
        # ------------------------------------------------------------------
        # helpers
        # ------------------------------------------------------------------
        max_s = self.max_slot
        big_m = self.big_m

        def _first_last_busy(res: Resource) -> tuple[int | None, int | None]:
            """Return first / last busy slot of a *fixed* resource."""
            busy = sorted(self.slot_of[n] for n, r in self.resource_of.items() if r is res)
            return (busy[0], busy[-1]) if busy else (None, None)

        # ------------------------------------------------------------------
        # 1) idle-indicator   idleS / idleE
        # ------------------------------------------------------------------
        self.idleS: dict[tuple[Resource, int], gp.Var | int] = {}
        self.idleE: dict[tuple[Resource, int], gp.Var | int] = {}

        # ............................................ fixed cores .........
        for res in self.tsa.resources:
            if res is None or (isinstance(res, Core) and res.id == self.offchip_core_id):
                continue  # not part of overlap

            first_busy, last_busy = _first_last_busy(res)
            if first_busy is None or last_busy is None:
                continue

            for s in range(max_s + 1):
                self.idleS[(res, s)] = 1 if s < first_busy else 0
                self.idleE[(res, s)] = 1 if s > last_busy else 0

        # ............................................ links (path-dep) ....
        # we need:
        #   link_used      = 1  ⇔  link carries traffic in *any* slot
        #   prefix_sum[s]  = Σ_{τ≤s}  active_{τ}
        #   suffix_sum[s]  = Σ_{τ≥s}  active_{τ}
        # idleS = 1  ⇔  prefix_sum[s]==0        (no activity up to *s*)
        # idleE = 1  ⇔  suffix_sum[s]==0        (no activity after *s*)

        self.link_used: dict[CommunicationLink, gp.Var] = {}
        self.prefixs: dict[CommunicationLink, list[gp.Var]] = {}
        self.suffixs: dict[CommunicationLink, list[gp.Var]] = {}

        for link in self.link_set:
            # ---------- active_{link,s} expression ------------------------
            active_s: dict[int, gp.LinExpr] = {}
            for s in range(max_s + 1):
                active_s[s] = quicksum(
                    self.y_path[(tr, p)]
                    for (tr, p) in self.y_path
                    if link in self.links_in_path[(tr, p)] and self.slot_of[tr] == s
                )

            # ---------- link_used binary ----------------------------------
            lu = self.model.addVar(vtype=GRB.BINARY, name=f"linkUsed_{_resource_key(link)}")
            self.link_used[link] = lu
            sum_active = quicksum(active_s.values())
            # (1) if lu == 1  ⇒  link carries traffic
            self.model.addConstr(sum_active >= lu, name=f"link_used_def_{_resource_key(link)}")
            # (2) if link carries traffic  ⇒  lu == 1
            self.model.addConstr(sum_active <= big_m * lu, name=f"link_used_def2_{_resource_key(link)}")

            # ---------- cumulative sums -----------------------------------
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

            # ---------- idleS / idleE binaries ----------------------------
            for s in range(max_s + 1):
                is_ = self.model.addVar(vtype=GRB.BINARY, name=f"idleS_{_resource_key(link)}_{s}")
                ie_ = self.model.addVar(vtype=GRB.BINARY, name=f"idleE_{_resource_key(link)}_{s}")
                self.idleS[(link, s)] = is_
                self.idleE[(link, s)] = ie_

                # prefix_sum == 0  →  is_ = 1
                self.model.addConstr(prefix[s] <= big_m * (1 - is_))
                self.model.addConstr(prefix[s] >= lu - big_m * is_)

                # suffix_sum == 0  →  ie_ = 1
                self.model.addConstr(suffix[s] <= big_m * (1 - ie_))
                self.model.addConstr(suffix[s] >= lu - big_m * ie_)

                # If the link is never used (lu==0) → force is_=1 and ie_=0 for correct idle_lat calculation
                self.model.addConstr(is_ >= 1 - lu)
                self.model.addConstr(ie_ <= lu)

        # ------------------------------------------------------------------
        # 2) idle-latency per resource
        # ------------------------------------------------------------------
        self.idle_lat: dict[Resource, gp.Var] = {}
        for res in {r for r, _ in self.idleS} | {r for r, _ in self.idleE}:
            expr = quicksum(
                self.idleS.get((res, s), 0) * self.slot_latency[s] + self.idleE.get((res, s), 0) * self.slot_latency[s]
                for s in range(max_s + 1)
            )
            v = self.model.addVar(vtype=GRB.INTEGER, name=f"idleLat_{_resource_key(res)}")
            self.model.addConstr(v == expr)
            self.idle_lat[res] = v

        # ------------------------------------------------------------------
        # 3) overlap  =  min idle_lat[r]   (only over resources that are used)
        # ------------------------------------------------------------------
        overlap = self.model.addVar(vtype=GRB.INTEGER, name="overlap")
        self.overlap = overlap
        for v in self.idle_lat.values():
            self.model.addConstr(overlap <= v)

        # ------------------------------------------------------------------
        # EXTRA) transfer fires (across all transfers and steady-state iterations)
        # ------------------------------------------------------------------
        self.total_transfer_cost = quicksum(
            self._transfer_latency(tr, p) * self.y_path[(tr, p)] * self.firesC[tr] for (tr, p) in self.y_path
        )

        # cost of DRAM → MemC   (single hop, latency = size / bw of that link)
        lat_dram_mem = {
            (tr, mc): ceil(tr.size / self._first_link_bw_from_dram(tr, mc))
            for tr in self.transfer_nodes
            if self._is_const_io(tr)
            for mc in self.mem_cores
            if self._any_path_through_mc(tr, mc)
        }
        self.total_transfer_cost += quicksum(
            lat_dram_mem[(tr, mc)] * self.firesM[tr] * self.m_store[(tr, mc)]
            for tr in self.transfer_nodes
            if self._is_const_io(tr)
            for mc in self.mem_cores
            if self._any_path_through_mc(tr, mc)
        )

        # ------------------------------------------------------------------
        # EXTRA) max mem core usage across all memory cores (for constant IO transfers)
        # ------------------------------------------------------------------
        self.mem_core_usage_per_core: dict[Core, gp.Var] = {}
        for mc in self.mem_cores:
            usage = self.model.addVar(vtype=GRB.INTEGER, name=f"memCoreUsage_{_resource_key(mc)}")
            self.model.addConstr(
                usage == quicksum(self.m_store[(tr, mc)] for tr in self.transfer_nodes if self._is_const_io(tr)),
                name=f"memCoreUsageConstr_{_resource_key(mc)}",
            )
            self.mem_core_usage_per_core[mc] = usage

        self.min_mem_core_usage = self.model.addVar(vtype=GRB.INTEGER, name="minMemCoreUsage")
        for i, usage in enumerate(self.mem_core_usage_per_core.values()):
            self.model.addConstr(self.min_mem_core_usage <= usage, name=f"minMemCoreUsage_le_{i}")

        # ------------------------------------------------------------------
        # 4) total latency + objective
        # ------------------------------------------------------------------
        self.total_lat = self.model.addVar(vtype=GRB.INTEGER, name="total_latency")
        self.total_latency = self.total_lat
        self.model.addConstr(
            self.total_lat == self.iterations * quicksum(self.slot_latency.values()) - (self.iterations - 1) * overlap
        )
        obj_func = self.total_lat + self.total_transfer_cost - self.min_mem_core_usage
        self.model.setObjective(obj_func, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # public solve()                                                     #
    # ------------------------------------------------------------------ #
    def solve(self, *, tee: bool = True) -> tuple[TimeSlotAllocation, SteadyStateWorkload, int]:  # noqa: PLR0912
        self.model.setParam("OutputFlag", 1 if tee else 0)
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError("Gurobi did not find an optimal solution.")

        # ---------- sanity checks -------------------
        self._check_io_transfers_firing_levels()

        # ---------- read back decisions --------------------------------
        tensor_alloc = self.get_tensor_allocations()
        routing = self.get_transfer_routing()
        self.update_transfer_reuse_levels()
        self.update_transfer_memory_core_allocation()

        # ---------- rebuild TSA with new resources ---------------------
        new_allocs: list[tuple[int, Resource, SteadyStateNode]] = []
        for slot, res, node in self.tsa.allocations:
            if isinstance(node, SteadyStateTensor):
                if res is None:
                    assert node in tensor_alloc, f"Tensor {node.node_name} not found in tensor_alloc."
                    res_new = tensor_alloc[node]
                else:
                    res_new = res
                new_allocs.append((slot, res_new, node))
            elif isinstance(node, SteadyStateTransfer):
                assert node in routing, f"Transfer {node.node_name} not found in routing."
                res_new = routing[node]
                for link in res_new:
                    new_allocs.append((slot, link, node))
            elif isinstance(node, SteadyStateComputation):
                assert res is not None, f"Computation {node.node_name} has no resource assigned."
                new_allocs.append((slot, res, node))
            else:
                raise TypeError(f"Unexpected node type: {type(node)} in TSA allocations.")
        # Update the TimeSlotAllocation with the new allocations
        tsa_upd = TimeSlotAllocation(new_allocs)
        # Update the steady state workload to prevent graph inconsistencies
        ssw_upd = self.get_updated_steady_state_workload()
        assert self.total_latency is not None, "Total latency variable was not created."
        return tsa_upd, ssw_upd, int(self.total_latency.X)

    def update_transfer_reuse_levels(  # noqa: PLR0912
        self,
    ) -> None:
        compute_tile_reuse_levels: dict[SteadyStateTransfer, int] = {}
        mem_tile_reuse_levels: dict[SteadyStateTransfer, int] = {}
        for tr in self.transfer_nodes:
            for stop in range(-1, len(tr.steady_state_iteration_space.get_temporal_variables())):
                if self.z_stopC[(tr, stop)].X > self.VAR_THRESHOLD:
                    compute_tile_reuse_levels[tr] = stop
                if self.z_stopM[(tr, stop)].X > self.VAR_THRESHOLD:
                    mem_tile_reuse_levels[tr] = stop
        # Update the steady state iteration space for transfers
        for tr in self.transfer_nodes:
            stop = compute_tile_reuse_levels[tr]
            # Set all iteration variables to REUSE flag
            for i, iter_var in enumerate(tr.steady_state_iteration_space.get_temporal_variables()):
                if i <= stop:
                    flag = ComputeTileReuse.REUSE
                else:
                    flag = ComputeTileReuse.NO_REUSE
                iter_var.compute_tile_reuse = flag
            for i, iter_var in enumerate(tr.steady_state_iteration_space.get_temporal_variables()):
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
    ) -> dict[SteadyStateTransfer, tuple[CommunicationLink]]:
        routing: dict[SteadyStateTransfer, tuple[CommunicationLink]] = {}
        for tr in self.transfer_nodes:
            chosen = [p for p in tr.possible_resource_allocation if self.y_path[(tr, tuple(p))].X > self.VAR_THRESHOLD]
            if len(chosen) != 1:
                raise ValueError(f"{tr.node_name}: expected exactly one path, got {chosen}")
            path = chosen[0]
            tr.chosen_resource_allocation = path
            tr.runtime = self._transfer_latency(tr, path)
            routing[tr] = path
        return routing

    def get_chosen_memory_cores(
        self,
    ) -> dict[SteadyStateTransfer, Core]:
        # --- which MemC was selected
        chosen_memory_cores: dict[SteadyStateTransfer, Core] = {}
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
    ) -> dict[SteadyStateTensor, Core]:
        tensor_alloc: dict[SteadyStateTensor, Core] = {}
        for t in self.tensor_var:
            chosen = [c for c in t.possible_resource_allocation if self.x_tensor[(t, c)].X > self.VAR_THRESHOLD]
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
        for edge in self.ssw.edges(data=True):
            src, dst, data = edge
            assert src in self.ssw.node_list, f"Source {src} not found in ssw nodes."
            assert dst in self.ssw.node_list, f"Destination {dst} not found in ssw nodes."
            src = self.ssw.node_list[self.ssw.node_list.index(src)]
            dst = self.ssw.node_list[self.ssw.node_list.index(dst)]
            ssw_upd.add_edge(src, dst, **data)
        return ssw_upd

    def _check_io_transfers_firing_levels(self) -> None:
        # firesC should always be greater or equal to firesM
        for tr in self.transfer_nodes:
            assert self.firesC[tr].X >= self.firesM[tr].X - 1e-6
        # chosen stopM ≥ stopC
        for tr in self.transfer_nodes:
            stop_max = len(tr.steady_state_iteration_space.get_temporal_variables())
            stopC = next(s for s in range(-1, stop_max) if self.z_stopC[(tr, s)].X > self.VAR_THRESHOLD)
            stopM = next(s for s in range(-1, stop_max) if self.z_stopM[(tr, s)].X > self.VAR_THRESHOLD)
            assert stopM >= stopC
