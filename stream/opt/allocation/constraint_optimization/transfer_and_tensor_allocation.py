import logging
import math
import os
from collections import defaultdict
from math import ceil, prod
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import yaml

# GRB.Callback constants — used only by _mip_progress_callback (Gurobi-specific, per D-04)
from gurobipy import GRB

from stream.cost_model.communication_manager import MulticastPathPlan
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
from stream.opt.allocation.constraint_optimization.utils import get_active_latency
from stream.opt.solver import (
    ConstraintSelection,
    SolverBackend,
    SolverModel,
    SolverParams,
    SolverVar,
    SolverVarType,
    create_solver,
)
from stream.workload.node import HasOutputs, TransferType
from stream.workload.steady_state.iteration_space import (
    IterationVariableType,
    LoopEffect,
    Reuse,
    SteadyStateIterationSpace,
)
from stream.workload.steady_state.node import Node
from stream.workload.workload import (
    ComputationNode,
    HasIterationSpace,
    InEdge,
    OutEdge,
    Tensor,
    TransferNode,
    Workload,
)

_logger = logging.getLogger(__name__)

TensorPlacementChoice: TypeAlias = tuple[Core, ...]

TensorReuseLevels: TypeAlias = dict[Tensor, int]
TensorDepths: TypeAlias = dict[Tensor, int]
TensorAlloc: TypeAlias = dict[Tensor, TensorPlacementChoice]
TransferAlloc: TypeAlias = dict[TransferNode, MulticastPathPlan]
MemoryAlloc: TypeAlias = dict[TransferNode, TensorPlacementChoice]


class TransferAndTensorAllocator:
    """
    MILP that decides

    1. where every movable tensor lives
    2. which routing choice each transfer uses
    """

    VAR_THRESHOLD = 0.5
    DMA_COUNT_SAME_TENSOR_ON_CORE_ONCE_GLOBALLY = False
    # False: count tensor-core occupancy separately for each transfer that uses it
    # True:  count tensor-core occupancy only once across all transfers

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        workload: Workload,
        timeslots: dict[Node, int],
        accelerator: Accelerator,
        iterations: int,
        ssis: dict[HasIterationSpace | Tensor, SteadyStateIterationSpace],
        multiplicities: dict[ComputationNode, int],
        mapping: Mapping,
        cost_lut: CoreCostLUT,
        *,
        big_m: int | None = None,
        gurobi_verbosity: int = 1,
        nb_cols_to_use: int = 4,
        output_path: str = "",
        context: TransferAndTensorContext | None = None,
        backend: str = "ORTOOLS_GSCIP",
        constraint_selection: ConstraintSelection | None = None,
    ):
        self.workload = workload
        self.slot_of = timeslots
        self.accelerator = accelerator
        self.context = context or build_transfer_context(
            accelerator, nb_cols_to_use=nb_cols_to_use, force_double_buffering=True
        )
        self.offchip_core_id = self.context.offchip_core_id
        self.iterations = iterations
        self.ssis = ssis
        self.multiplicities = multiplicities
        self.mapping = mapping
        self.cost_lut = cost_lut
        self.output_path = output_path
        self.backend_str = backend
        self.constraint_selection = constraint_selection or ConstraintSelection()

        self.max_slot = max(timeslots.values()) if timeslots else 0
        self.big_m = big_m or len(workload.nodes()) + 5
        self.force_io_transfers_on_mem_tile = self.context.force_io_transfers_on_mem_tile

        # ------------------- categorise nodes -------------------- #
        self.ssc_nodes: tuple[ComputationNode, ...] = tuple(workload.get_computation_nodes())
        self.transfer_nodes: tuple[TransferNode, ...] = tuple(workload.get_transfer_nodes())

        self.force_double_buffering = self.context.force_double_buffering
        self.mem_cores = list(self.context.mem_cores)

        # ------------------- canonicalized options -------------------- #
        self.tensors: list[Tensor] = []
        self.tensor_fixed: list[Tensor] = []
        self.tensor_var: list[Tensor] = []
        self.possible_tensor_allocations: dict[Tensor, tuple[TensorPlacementChoice, ...]] = {}
        self.possible_transfer_allocations: dict[TransferNode, tuple[MulticastPathPlan, ...]] = {}

        self._init_option_sets()

        # ------------------- optimization model ---------------------- #
        self.model: SolverModel = create_solver(SolverBackend[self.backend_str], "transfer_tensor_alloc")
        self.model.set_param(SolverParams.VERBOSITY, gurobi_verbosity)
        self.model.set_param(SolverParams.LOG_TO_CONSOLE, 0)

        # primary decision vars
        self.x_tensor_choice: dict[tuple[Tensor, TensorPlacementChoice], SolverVar] = {}
        self.y_path_choice: dict[tuple[TransferNode, MulticastPathPlan], SolverVar] = {}

        # auxiliary indicators
        self.transfer_core_indicator: dict[tuple[TransferNode, Core], SolverVar] = {}
        self.tensor_core_indicator: dict[tuple[Tensor, Core], SolverVar] = {}
        self.same_core_indicator: dict[tuple[Tensor, Tensor, Core], SolverVar] = {}

        # helpers
        self.link_set: set[CommunicationLink] = set()
        self.links_in_choice: dict[tuple[TransferNode, MulticastPathPlan], set[CommunicationLink]] = {}
        self.choice_src_cores: dict[tuple[TransferNode, MulticastPathPlan], set[Core]] = {}
        self.choice_dst_cores: dict[tuple[TransferNode, MulticastPathPlan], set[Core]] = {}
        self.choice_mem_cores: dict[tuple[TransferNode, MulticastPathPlan], set[Core]] = {}
        self.choice_has_empty_path: dict[tuple[TransferNode, MulticastPathPlan], bool] = {}

        # latency vars
        self._transfer_latency_cache: dict[tuple[TransferNode, MulticastPathPlan], SolverVar] = {}
        self.slot_latency: dict[int, SolverVar] = {}
        self.overlap: SolverVar | None = None
        self.total_latency: SolverVar | None = None

        # transfer fire helpers init
        self._ensure_same_ssis_for_all_transfers()
        self.reuse_levels: dict[tuple[Tensor, int], tuple[int, int]] = {}
        self.tiles_needed_levels: dict[tuple[Tensor, int], int] = {}
        self.bds_needed_levels: dict[tuple[Tensor, int], int] = {}
        self.tensors_to_optimize_reuse_for: list[Tensor] = []
        self._init_transfer_fire_helpers()

        # track optimization progress
        self.optimization_trace: list[dict[str, float | str | None]] = []

        # counter for deduplicating variable/constraint names across all add_var calls
        # (MathOpt rejects duplicate names; Gurobi silently accepts them)
        self._name_counter: dict[str, int] = {}

        self._build_model()

    # ------------------------------------------------------------ #
    # option canonicalization                                      #
    # ------------------------------------------------------------ #
    def _init_option_sets(self) -> None:
        for node in self.workload.topological_sort():
            if not isinstance(node, HasOutputs):
                continue
            for tensor in node.outputs:
                if tensor in self.possible_tensor_allocations:
                    continue
                raw_alloc = self._retrieve_core_allocation(node)
                normalized = self._normalize_tensor_choices(raw_alloc)
                self.possible_tensor_allocations[tensor] = normalized
                self.tensors.append(tensor)
                if len(normalized) == 1:
                    self.tensor_fixed.append(tensor)
                else:
                    self.tensor_var.append(tensor)

        for tr in self.transfer_nodes:
            raw_paths = self.mapping.get(tr).resource_allocation
            self.possible_transfer_allocations[tr] = self._normalize_path_choices(raw_paths)

    def _normalize_tensor_choices(self, raw: Any) -> tuple[TensorPlacementChoice, ...]:
        """
        Canonical form:
            tuple[ tuple[Core, ...], ... ]
        """
        if raw is None:
            raise ValueError("Tensor allocation options cannot be None.")

        raw_tuple = tuple(raw)
        if not raw_tuple:
            raise ValueError("Tensor allocation options cannot be empty.")

        # Case 1: raw itself is one flat iterable of Core objects
        if all(isinstance(x, Core) for x in raw_tuple):
            return (tuple(raw_tuple),)

        # Case 2: raw is iterable of choices, each choice iterable of Core
        out: list[TensorPlacementChoice] = []
        for choice in raw_tuple:
            choice_tuple = tuple(choice)
            if not choice_tuple:
                raise ValueError("Empty tensor placement choice encountered.")
            if not all(isinstance(c, Core) for c in choice_tuple):
                raise TypeError(f"Invalid tensor placement choice: {choice_tuple}")
            out.append(tuple(choice_tuple))
        return tuple(out)

    def _normalize_path_choices(self, raw: Any) -> tuple[MulticastPathPlan, ...]:
        """
        Canonical form:
            tuple[MulticastPathPlan, ...]

        Expects raw to be an iterable of MulticastPathPlan objects.
        """
        if raw is None:
            raise ValueError("Transfer path options cannot be None.")
        raw_tuple = tuple(raw)
        if not raw_tuple:
            raise ValueError("Transfer path options cannot be empty.")
        if not all(isinstance(x, MulticastPathPlan) for x in raw_tuple):
            bad_types = {type(x) for x in raw_tuple if not isinstance(x, MulticastPathPlan)}
            raise TypeError(
                f"Unsupported routing choice structure. Expected iterable of MulticastPathPlan, "
                f"got invalid element types: {bad_types}"
            )
        return raw_tuple

    # ------------------------------------------------------------ #
    # internal helpers                                             #
    # ------------------------------------------------------------ #
    @staticmethod
    def _mem_factor(t: Tensor, core: Core) -> int:
        return 1

    @staticmethod
    def _transfer_latency_for_path(tr: TransferNode, path: MulticastPathPlan) -> int:
        if not path or not path.links_used:
            return 0
        min_bw = min(link.bandwidth for link in path.links_used)
        assert len(tr.inputs) == 1, "Only single-input transfers are supported for latency calculation."
        tensor = tr.inputs[0]
        return ceil(tensor.size_bits() / min_bw)

    def _ensure_same_ssis_for_all_transfers(self) -> None:
        first_ssis = self.ssis[self.transfer_nodes[0]]
        first_transfer_ssis_sizes = first_ssis.get_temporal_sizes()
        first_transfer_ssis_total_size = prod(first_transfer_ssis_sizes)
        for tr in self.transfer_nodes:
            transfer_ssis = self.ssis[tr]
            transfer_ssis_sizes = transfer_ssis.get_temporal_sizes()
            transfer_ssis_total_size = prod(transfer_ssis_sizes)
            if transfer_ssis_total_size != first_transfer_ssis_total_size:
                raise ValueError(
                    f"Transfer {tr.name} has different SSIS total size than the {self.transfer_nodes[0].name}: "
                    f"{transfer_ssis_total_size} != {first_transfer_ssis_total_size}"
                )

    def _init_transfer_fire_helpers(self) -> None:
        for t in self.workload.tensors:
            ssis = self.ssis[t].get_applicable_temporal_variables()
            sizes = [iter_var.size for iter_var in ssis]
            relevancies = [iter_var.relevant for iter_var in ssis]
            reuses = [iter_var.reuse for iter_var in ssis]
            if any(r != Reuse.NOT_SET for r in reuses):
                continue
            self.tensors_to_optimize_reuse_for.append(t)
            fires = math.prod(sizes)
            size_factor = 1
            tiles_needed = 1
            bds_needed = 1
            self.reuse_levels[(t, -1)] = (fires, size_factor)
            self.tiles_needed_levels[(t, -1)] = tiles_needed
            self.bds_needed_levels[(t, -1)] = bds_needed
            for i, (Nl, relevancy) in enumerate(zip(sizes, relevancies, strict=True)):
                size_factor *= Nl if relevancy else 1
                tiles_needed *= Nl if relevancy else 1
                fires //= Nl
                self.reuse_levels[(t, i)] = (fires, size_factor)
                self.tiles_needed_levels[(t, i)] = tiles_needed
                if relevancy:
                    bds_needed = 1
                else:
                    bds_needed *= Nl
                self.bds_needed_levels[(t, i)] = bds_needed
            if self.force_double_buffering:
                self.tiles_needed_levels[(t, -1)] = 2

    def _is_const_i(self, tr: TransferNode) -> bool:
        src = next(iter(self.workload.predecessors(tr)))
        return isinstance(src, InEdge)

    def _is_const_o(self, tr: TransferNode) -> bool:
        dst = next(iter(self.workload.successors(tr)))
        return isinstance(dst, OutEdge)

    def _is_const_io(self, tr: TransferNode) -> bool:
        return self._is_const_i(tr) or self._is_const_o(tr)

    def _constant_transfer_tensor(self, tr: TransferNode) -> Tensor:
        if self._is_const_i(tr):
            return tr.outputs[0]
        if self._is_const_o(tr):
            return tr.inputs[0]
        raise ValueError(f"Transfer {tr.name} is not a constant I/O transfer.")

    def _all_dma_candidate_cores(self) -> set[Core]:
        """
        All on-chip cores that may host tensors participating in transfers.
        Off-chip core is excluded from DMA accounting.
        """
        cores: set[Core] = set()
        for t in self.workload.tensors:
            if not isinstance(t, Tensor):
                continue
            for core in self._candidate_cores_for_tensor(t):
                if core.id == self.offchip_core_id:
                    continue
                cores.add(core)
        return cores

    def _unique_tensor_list(self, tensors) -> list[Tensor]:
        """
        Stable unique filtering for tensors.
        """
        seen: set[Tensor] = set()
        out: list[Tensor] = []
        for t in tensors:
            if not isinstance(t, Tensor):
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _transfer_incoming_tensors(self, tr: TransferNode) -> list[Tensor]:
        """
        Tensors whose presence on a core contributes to incoming DMA usage on that core.

        Assumption:
            incoming DMA on a core corresponds to transfer outputs allocated on that core.
        """
        return self._unique_tensor_list(tr.outputs)

    def _transfer_outgoing_tensors(self, tr: TransferNode) -> list[Tensor]:
        """
        Tensors whose presence on a core contributes to outgoing DMA usage on that core.

        Assumption:
            outgoing DMA on a core corresponds to transfer inputs allocated on that core.
        """
        return self._unique_tensor_list(tr.inputs)

    def _transfer_incoming_dma_expr(self, tr: TransferNode, core: Core):
        """
        DMA contribution of one transfer to the incoming DMA load of one core.

        If DMA_COUNT_SAME_TENSOR_ON_CORE_ONCE_GLOBALLY is False:
            each transfer contributes its own tensor occupancy.

        If DMA_COUNT_SAME_TENSOR_ON_CORE_ONCE_GLOBALLY is True:
            the aggregation across transfers is handled in _add_dma_usage_constraints(),
            so this helper is only used in the per-transfer mode.
        """
        tensors = self._transfer_incoming_tensors(tr)
        return self.model.quicksum(self._tensor_on_core_expr(t, core) for t in tensors)

    def _transfer_outgoing_dma_expr(self, tr: TransferNode, core: Core):
        """
        DMA contribution of one transfer to the outgoing DMA load of one core.
        """
        tensors = self._transfer_outgoing_tensors(tr)
        return self.model.quicksum(self._tensor_on_core_expr(t, core) for t in tensors)

    def _global_incoming_dma_expr(self, core: Core):
        """
        Global incoming DMA usage on one core, counting each tensor at most once across all transfers.
        """
        tensors: set[Tensor] = set()
        for tr in self.transfer_nodes:
            tensors.update(self._transfer_incoming_tensors(tr))
        return self.model.quicksum(self._tensor_on_core_expr(t, core) for t in tensors)

    def _global_outgoing_dma_expr(self, core: Core):
        """
        Global outgoing DMA usage on one core, counting each tensor at most once across all transfers.
        """
        tensors: set[Tensor] = set()
        for tr in self.transfer_nodes:
            tensors.update(self._transfer_outgoing_tensors(tr))
        return self.model.quicksum(self._tensor_on_core_expr(t, core) for t in tensors)

    # ------------------------------------------------------------ #
    # canonical choice metadata                                    #
    # ------------------------------------------------------------ #
    def _index_choice_metadata(self) -> None:
        for tr in self.transfer_nodes:
            for choice in self.possible_transfer_allocations[tr]:
                key = (tr, choice)
                self.links_in_choice[key] = self._links_of_choice(choice)
                self.link_set.update(self.links_in_choice[key])
                self.choice_src_cores[key] = self._src_cores_of_choice(choice)
                self.choice_dst_cores[key] = self._dst_cores_of_choice(choice)
                self.choice_mem_cores[key] = self._mem_cores_of_choice(choice)
                self.choice_has_empty_path[key] = len(choice.links_used) == 0

    @staticmethod
    def _links_of_choice(choice: MulticastPathPlan) -> set[CommunicationLink]:
        return set(choice.links_used)

    @staticmethod
    def _src_cores_of_choice(choice: MulticastPathPlan) -> set[Core]:
        return set(choice.sources)

    @staticmethod
    def _dst_cores_of_choice(choice: MulticastPathPlan) -> set[Core]:
        return set(choice.targets)

    def _mem_cores_of_choice(self, choice: MulticastPathPlan) -> set[Core]:
        out: set[Core] = set()
        for link in choice.links_used:
            if link.sender in self.mem_cores:
                out.add(link.sender)
            if link.receiver in self.mem_cores:
                out.add(link.receiver)
        return out

    # ------------------------------------------------------------ #
    # derived linear expressions / indicators                      #
    # ------------------------------------------------------------ #
    def _tensor_choices(self, t: Tensor) -> tuple[TensorPlacementChoice, ...]:
        return self.possible_tensor_allocations[t]

    def _path_choices(self, tr: TransferNode) -> tuple[MulticastPathPlan, ...]:
        return self.possible_transfer_allocations[tr]

    def _fixed_tensor_choice(self, t: Tensor) -> TensorPlacementChoice:
        choices = self._tensor_choices(t)
        assert len(choices) == 1, f"Tensor {t.name} is not fixed."
        return choices[0]

    def _candidate_cores_for_tensor(self, t: Tensor) -> set[Core]:
        return {core for choice in self._tensor_choices(t) for core in choice}

    def _tensor_on_core_expr(self, t: Tensor, core: Core):
        if t in self.tensor_fixed:
            return int(core in self._fixed_tensor_choice(t))

        return self.model.quicksum(
            self.x_tensor_choice[(t, choice)]._raw for choice in self._tensor_choices(t) if core in choice
        )

    def _transfer_uses_core_var(self, tr: TransferNode, core: Core) -> SolverVar:
        key = (tr, core)
        if key in self.transfer_core_indicator:
            return self.transfer_core_indicator[key]

        u = self.model.add_var(vtype=SolverVarType.BINARY, name=f"u_{tr.name}_{_resource_key(core)}")
        self.transfer_core_indicator[key] = u

        occ_exprs = [self._tensor_on_core_expr(t, core) for t in tr.tensors if isinstance(t, Tensor)]
        if not occ_exprs:
            self.model.add_constr(u == 0, name=f"u_zero_{tr.name}_{_resource_key(core)}")
            return u

        for i, occ in enumerate(occ_exprs):
            self.model.add_constr(u >= occ, name=f"u_lb_{tr.name}_{_resource_key(core)}_{i}")
        self.model.add_constr(
            u <= self.model.quicksum(occ_exprs),
            name=f"u_ub_{tr.name}_{_resource_key(core)}",
        )
        return u

    def _tensor_uses_core_var(self, t: Tensor, core: Core) -> SolverVar:
        key = (t, core)
        if key in self.tensor_core_indicator:
            return self.tensor_core_indicator[key]

        u = self.model.add_var(vtype=SolverVarType.BINARY, name=f"u_{t.name}_{_resource_key(core)}")
        self.tensor_core_indicator[key] = u

        occ = self._tensor_on_core_expr(t, core)
        self.model.add_constr(
            u == occ,
            name=f"u_eq_{t.name}_{_resource_key(core)}",
        )
        return u

    def _same_core_var(self, src_tensor: Tensor, dst_tensor: Tensor, core: Core) -> SolverVar:
        key = (src_tensor, dst_tensor, core)
        if key in self.same_core_indicator:
            return self.same_core_indicator[key]

        v = self.model.add_var(
            vtype=SolverVarType.BINARY,
            name=f"same_{src_tensor.name}_{dst_tensor.name}_{_resource_key(core)}",
        )
        self.same_core_indicator[key] = v

        src_occ = self._tensor_on_core_expr(src_tensor, core)
        dst_occ = self._tensor_on_core_expr(dst_tensor, core)

        self.model.add_constr(
            v <= src_occ, name=f"same_src_ub_{src_tensor.name}_{dst_tensor.name}_{_resource_key(core)}"
        )
        self.model.add_constr(
            v <= dst_occ, name=f"same_dst_ub_{src_tensor.name}_{dst_tensor.name}_{_resource_key(core)}"
        )
        self.model.add_constr(
            v >= src_occ + dst_occ - 1,
            name=f"same_lb_{src_tensor.name}_{dst_tensor.name}_{_resource_key(core)}",
        )
        return v

    def _all_candidate_cores_for_transfer(self, tr: TransferNode) -> set[Core]:
        out: set[Core] = set()
        for t in tr.tensors:
            if isinstance(t, Tensor):
                out.update(self._candidate_cores_for_tensor(t))
        return out

    # ------------------------------------------------------------ #
    # model construction                                           #
    # ------------------------------------------------------------ #
    def _build_model(self):
        self._create_vars()
        self._index_choice_metadata()
        self._create_constraints()
        self._overlap_and_objective()

    # ...................... VARIABLES ................... #
    def _create_vars(self):
        self.__create_tensor_placement_vars()
        self.__create_transfer_path_vars()
        self.__create_reuse_vars()
        self.__create_slot_latency_vars()

    def __create_slot_latency_vars(self):
        for s in range(self.max_slot + 1):
            self.slot_latency[s] = self.model.add_var(vtype=SolverVarType.INTEGER, name=f"L_{s}")

    def __create_reuse_vars(self):
        self.z_stop: dict[tuple[Tensor, int], SolverVar] = {}
        for t in self.workload.tensors:
            sizes = self.ssis[t].get_applicable_temporal_sizes()
            for stop in range(-1, len(sizes)):
                v = self.model.add_var(vtype=SolverVarType.BINARY, name=f"zStop_{t.name}_L{stop}")
                self.z_stop[(t, stop)] = v
            self.model.add_constr(
                self.model.quicksum(self.z_stop[(t, s)]._raw for s in range(-1, len(sizes))) == 1,
                name=f"zStop_Choose_One_{t.name}",
            )
            if t not in self.tensors_to_optimize_reuse_for:
                reuses = self.ssis[t].get_temporal_reuses()
                stop = -2
                for i in range(len(reuses) - 1, -1, -1):
                    if reuses[i] == Reuse.REUSE:
                        stop = i
                        break
                assert stop >= -1, f"Something went wrong for {t.name} REUSE indexing: {reuses}"
                self.model.add_constr(
                    self.z_stop[(t, stop)] == 1,
                    name=f"zStop_FixedStop_{t.name}_L{stop}",
                )

    def __create_transfer_path_vars(self):
        for tr in self.transfer_nodes:
            for i, choice in enumerate(self.possible_transfer_allocations[tr]):
                v = self.model.add_var(vtype=SolverVarType.BINARY, name=f"y_{tr.name}_choice_{i}")
                self.y_path_choice[(tr, choice)] = v

    def __create_tensor_placement_vars(self):
        for t in self.tensor_var:
            for choice in self.possible_tensor_allocations[t]:
                choice_name = "__".join(_resource_key(c) for c in choice)
                v = self.model.add_var(vtype=SolverVarType.BINARY, name=f"x_{t.name}_{choice_name}")
                self.x_tensor_choice[(t, choice)] = v

    # ...................... tensor placement .................... #
    def _tensor_placement_constraints(self):
        for t in self.tensor_var:
            self.model.add_constr(
                self.model.quicksum(
                    self.x_tensor_choice[(t, choice)]._raw for choice in self.possible_tensor_allocations[t]
                )
                == 1,
                name=f"place_{t.name}",
            )

    # ...................... CONSTRAINTS ................... #
    def _create_constraints(self):
        self._tensor_placement_constraints()
        self._path_choice_constraints()
        self._transfer_fire_rate_constraints()
        self._reuse_factor_rate_constraints()
        self._link_contention_constraints()
        if self.constraint_selection.memory_capacity:
            self._memory_capacity_constraints()
        else:
            _logger.warning("ConstraintSelection: skipping memory_capacity constraints")
        if self.constraint_selection.object_fifo_depth:
            self._object_fifo_depth_constraints()
        else:
            _logger.warning("ConstraintSelection: skipping object_fifo_depth constraints")
        if self.constraint_selection.buffer_descriptors:
            self._buffer_descriptor_constraints()
        else:
            _logger.warning("ConstraintSelection: skipping buffer_descriptors constraints")
        self._slot_latency_constraints()
        self._force_nonconstant_reuse_levels()
        self._force_final_output_reuse_levels()
        self._ensure_memory_and_compute_reuse_compatibility()
        self._force_reuse_includes_spatial()

    def _transfer_fire_rate_constraints(self):
        self.fires: dict[TransferNode, SolverVar] = {}
        for tr in self.transfer_nodes:
            assert len(tr.inputs) == 1, (
                f"Only single-input transfers are supported for fire rate constraints, "
                f"but {tr.name} has inputs {tr.inputs}."
            )
            t = tr.outputs[0]
            fires = self.model.add_var(vtype=SolverVarType.INTEGER, name=f"fires_{tr.name}")
            self.fires[tr] = fires

            self.model.add_constr(
                fires
                == self.model.quicksum(
                    self.reuse_levels[(t, s)][0] * self.z_stop[(t, s)]._raw
                    for s in range(-1, len(self.ssis[t].get_applicable_temporal_variables()))
                ),
                name=f"fires_def_{tr.name}",
            )

    def _reuse_factor_rate_constraints(self):
        self.reuse_factors: dict[TransferNode, SolverVar] = {}
        for tr in self.transfer_nodes:
            assert len(tr.inputs) == 1, (
                f"Only single-input transfers are supported for fire rate constraints, "
                f"but {tr.name} has inputs {tr.inputs}."
            )
            t = tr.outputs[0]
            reuse_factor = self.model.add_var(vtype=SolverVarType.INTEGER, name=f"reuse_factor_{tr.name}")
            self.reuse_factors[tr] = reuse_factor

            self.model.add_constr(
                reuse_factor
                == self.model.quicksum(
                    self.reuse_levels[(t, s)][1] * self.z_stop[(t, s)]._raw
                    for s in range(-1, len(self.ssis[t].get_applicable_temporal_variables()))
                ),
                name=f"reuse_factor_def_{tr.name}",
            )

    # ...................... path choice ........................ #
    def _path_choice_constraints(self) -> None:
        for tr in self.transfer_nodes:
            choices = self._path_choices(tr)
            self._add_one_path_constraint(tr, choices)
            self._add_source_tensor_coherence_constraints(tr, choices)
            self._add_destination_tensor_coherence_constraints(tr, choices)
            self._add_empty_path_coherence_constraints(tr, choices)

    def _add_one_path_constraint(self, tr: TransferNode, choices: tuple[MulticastPathPlan, ...]) -> None:
        self.model.add_constr(
            self.model.quicksum(self.y_path_choice[(tr, choice)]._raw for choice in choices) == 1,
            name=f"one_path_{tr.name}",
        )

    def _add_source_tensor_coherence_constraints(
        self, tr: TransferNode, choices: tuple[MulticastPathPlan, ...]
    ) -> None:
        src_tensors = tr.inputs
        assert all(isinstance(t, Tensor) for t in src_tensors), (
            f"Transfer {tr.name} has non-tensor input(s): {src_tensors}"
        )

        for src_tensor in src_tensors:
            for i, choice in enumerate(choices):
                y = self.y_path_choice[(tr, choice)]
                for src_core in self.choice_src_cores[(tr, choice)]:
                    self.model.add_constr(
                        y <= self._tensor_on_core_expr(src_tensor, src_core),
                        name=f"path_src_match_{tr.name}_{src_tensor.name}_{_resource_key(src_core)}_choice_{i}",
                    )

    def _add_destination_tensor_coherence_constraints(
        self, tr: TransferNode, choices: tuple[MulticastPathPlan, ...]
    ) -> None:
        dst_tensors = tr.outputs
        for dst_tensor in dst_tensors:
            assert isinstance(dst_tensor, Tensor), f"Expected {dst_tensor} to be a Tensor."
            for i, choice in enumerate(choices):
                y = self.y_path_choice[(tr, choice)]
                for dst_core in self.choice_dst_cores[(tr, choice)]:
                    self.model.add_constr(
                        y <= self._tensor_on_core_expr(dst_tensor, dst_core),
                        name=f"path_dst_match_{tr.name}_{dst_tensor.name}_{_resource_key(dst_core)}_choice_{i}",
                    )

    def _add_empty_path_coherence_constraints(self, tr: TransferNode, choices: tuple[MulticastPathPlan, ...]) -> None:
        """
        If a routing choice contains only empty paths, enforce that source and destination
        tensors can be colocated on at least one common candidate core.
        """
        if len(tr.inputs) != 1 or len(tr.outputs) == 0:
            return

        src_tensor = tr.inputs[0]
        assert isinstance(src_tensor, Tensor)

        for i, choice in enumerate(choices):
            if not self.choice_has_empty_path[(tr, choice)]:
                continue
            # Handle only all-empty choices here. Mixed empty/non-empty choices are still
            # covered by src/dst coherence on the non-empty paths.
            if len(choice.links_used) != 0:
                raise ValueError("Something went wrong in empty path determination")
            y = self.y_path_choice[(tr, choice)]
            for dst_tensor in tr.outputs:
                assert isinstance(dst_tensor, Tensor)
                common_cores = self._candidate_cores_for_tensor(src_tensor) & self._candidate_cores_for_tensor(
                    dst_tensor
                )
                if not common_cores:
                    self.model.add_constr(y == 0, name=f"empty_path_infeasible_{tr.name}_{dst_tensor.name}_choice_{i}")
                    continue
                coloc_terms = [self._same_core_var(src_tensor, dst_tensor, core) for core in common_cores]
                self.model.add_constr(
                    y <= self.model.quicksum(t._raw for t in coloc_terms),
                    name=f"empty_path_match_{tr.name}_{dst_tensor.name}_choice_{i}",
                )

    # ...................... link contention .................... #
    def _link_contention_constraints(self):
        usage: dict[tuple[CommunicationLink, int], list[SolverVar]] = defaultdict(list)
        for (tr, choice), y in self.y_path_choice.items():
            s = self.slot_of[tr]
            for link in self.links_in_choice[(tr, choice)]:
                usage[(link, s)].append(y)

        for (link, s), vars_ in usage.items():
            self.model.add_constr(
                self.model.quicksum(v._raw for v in vars_) <= 1, name=f"link_usage_{_resource_key(link)}_{s}"
            )

    # ...................... memory capacity .................... #
    def _memory_capacity_constraints(self):
        self.core_load: dict[Core, Any] = defaultdict(int)
        # Transfer output tensors on their chosen compute/memory cores
        for node in self.workload.get_iteration_space_nodes():
            for t in node.outputs:
                tensor_size = self.workload.get_tensor_single_core(t, node, self.mapping).size_bits()
                candidate_cores = self._candidate_cores_for_tensor(t)
                for c in candidate_cores:
                    assert isinstance(c, Core)
                    u = self._tensor_uses_core_var(t, c)
                    for stop in range(-1, len(self.ssis[t].get_applicable_temporal_variables())):
                        _, size_factor = self.reuse_levels[(t, stop)]
                        req_size = ceil(size_factor * tensor_size)
                        uz = self._add_binary_product(
                            a=u,
                            b=self.z_stop[(t, stop)],
                            base_name=f"memload_{t.name}_{_resource_key(c)}_L{stop}",
                        )
                        self.core_load[c] = self.core_load[c] + req_size * uz._raw

        for c, expr in self.core_load.items():
            cap = c.get_memory_capacity()
            self.model.add_constr(expr <= cap, name=f"mem_cap_{_resource_key(c)}")

    def _object_fifo_depth_constraints(self):
        self.object_fifo_depth: dict[Core, Any] = defaultdict(int)
        for tr in self.transfer_nodes:
            # TODO: Confirm assumption that OF linking causes only single object fifo depth increase
            for t in tr.outputs:
                resources = self._candidate_cores_for_tensor(t)
                for c in resources:
                    if c.id == self.offchip_core_id:
                        continue
                    assert isinstance(c, Core)
                    u = self._tensor_uses_core_var(t, c)
                    for stop in range(-1, len(self.ssis[t].get_applicable_temporal_variables())):
                        tiles_needed = self.tiles_needed_levels[(t, stop)]

                        uz = self._add_binary_product(
                            a=u,
                            b=self.z_stop[(t, stop)],
                            base_name=f"objfifo_{t.name}_{_resource_key(c)}_L{stop}",
                        )
                        self.object_fifo_depth[c] = self.object_fifo_depth[c] + tiles_needed * uz._raw
        self.context.add_object_fifo_constraints(self.model, self.object_fifo_depth)

    def _buffer_descriptor_constraints(self):
        """
        For compute tiles: use tiles_needed_levels to determine how many buffer descriptors are needed (relevant).
        For memory tiles: use bds_needed_levels to determine how many buffer descriptors are needed (irrelevant/repeat).
        """
        self.bd_depth: dict[Core, Any] = defaultdict(int)
        for tr in self.transfer_nodes:
            for t in tr.tensors:
                resources = self._candidate_cores_for_tensor(t)
                for c in resources:
                    if c.id == self.offchip_core_id:
                        continue
                    u = self._tensor_uses_core_var(t, c)
                    assert isinstance(c, Core)
                    if c.type == "compute":
                        for stop in range(-1, len(self.ssis[t].get_applicable_temporal_variables())):
                            bds_needed = self.tiles_needed_levels[(t, stop)]
                            uz = self._add_binary_product(
                                a=u,
                                b=self.z_stop[(t, stop)],
                                base_name=f"bddepth_{t.name}_{_resource_key(c)}_L{stop}",
                            )
                            self.bd_depth[c] = self.bd_depth[c] + bds_needed * uz._raw
                    else:
                        # If the core is a memory core, we add bd usage only if the eq. tensor on compute
                        # is not being reused (zStop[t, stop] == 0 at that reuse level)
                        # This means we create a new 'active' helper variable for the eq. tensor
                        # TODO: Shouldn't just be exactly that compute tensor reuse level
                        if t in tr.outputs:
                            assert len(tr.inputs) == 1
                            compute_tensor = tr.inputs[0]
                        elif t in tr.inputs:
                            # TODO: Check that for multiple outputs the reuse levels are equivalent,
                            # otherswise we may need to create separate active variables for each output tensor.
                            compute_tensor = tr.outputs[0]
                        else:
                            raise NotImplementedError("Expected tensor to be either input or output of the transfer.")
                        for stop in range(-1, len(self.ssis[t].get_applicable_temporal_variables())):
                            src_tensor_reuse = self.z_stop[(compute_tensor, stop)]
                            gate_var = self.model.add_var(
                                vtype=SolverVarType.BINARY,
                                name=f"active_{compute_tensor.name}_{_resource_key(c)}_L{stop}",
                            )
                            self.model.add_constr(
                                gate_var == 1 - src_tensor_reuse,
                                name=f"active_gate_{compute_tensor.name}_{_resource_key(c)}_L{stop}",
                            )
                            uz = self._add_binary_product(
                                a=u,
                                b=self.z_stop[(t, stop)],
                                base_name=f"bddepth_{t.name}_{_resource_key(c)}_L{stop}",
                            )
                            uzgate = self._add_binary_product(
                                a=uz,
                                b=gate_var,
                                base_name=f"bddepth_active_{t.name}_{_resource_key(c)}_L{stop}",
                            )
                            bds_needed = self.bds_needed_levels[(t, stop)]
                            self.bd_depth[c] = self.bd_depth[c] + bds_needed * uzgate._raw
        self.context.add_buffer_descriptor_constraints(self.model, self.bd_depth)

    def _ensure_memory_and_compute_reuse_compatibility(self):
        """
        Ensure that for COMPUTE_TO_MEM and MEM_TO_COMPUTE transfers the input and output
        reuse levels are equal.
        """
        for tr in self.transfer_nodes:
            inputs = tr.inputs
            outputs = tr.outputs
            relevancies = self.ssis[tr].get_applicable_temporal_relevancies()
            if tr.transfer_type in (TransferType.COMPUTE_TO_MEM,):
                assert len(outputs) == 1, "Expected exactly one output tensor for COMPUTE_TO_MEM transfer."
                output_tensor = outputs[0]
                for input_tensor in inputs:
                    for s in range(-1, len(relevancies)):
                        self.model.add_constr(
                            self.z_stop[(output_tensor, s)] == self.z_stop[(input_tensor, s)],
                            name=f"reuse_eq_input_{tr.name}_L{s}",
                        )
            elif tr.transfer_type in (TransferType.MEM_TO_COMPUTE,):
                assert len(inputs) == 1, "Expected exactly one input tensor for MEM_TO_COMPUTE transfer."
                input_tensor = inputs[0]
                for output_tensor in outputs:
                    for s in range(-1, len(relevancies)):
                        self.model.add_constr(
                            self.z_stop[(input_tensor, s)] == self.z_stop[(output_tensor, s)],
                            name=f"reuse_eq_output_{tr.name}_L{s}",
                        )

    def _force_reuse_includes_spatial(self):
        """
        Force reuse to cover any applicable temporal loop that sits inside (or at)
        the outermost spatial variable of a tensor. Temporal loops outside the
        outermost spatial do not need to be buffered for spatial coverage.
        """
        for t in self.tensors_to_optimize_reuse_for:
            variables = self.ssis[t].variables
            applicable_temporal = self.ssis[t].get_applicable_temporal_variables()
            outermost_spatial_pos = -1
            for pos, var in enumerate(variables):
                if var.type == IterationVariableType.SPATIAL:
                    outermost_spatial_pos = pos
            if outermost_spatial_pos < 0:
                continue
            min_reuse_level = -1
            for i, tv in enumerate(applicable_temporal):
                pos = next(p for p, v in enumerate(variables) if v is tv)
                if pos <= outermost_spatial_pos:
                    min_reuse_level = i
                else:
                    break
            if min_reuse_level < 0:
                continue
            self.model.add_constr(
                self.model.quicksum(self.z_stop[(t, s)]._raw for s in range(min_reuse_level, len(applicable_temporal)))
                >= 1,
                name=f"force_reuse_past_spatial_{t.name}",
            )

    # ...................... slot latency ........................ #
    def _slot_latency_constraints(self):
        for n in self.ssc_nodes:
            s = self.slot_of[n]
            latencies = [self.cost_lut.get_cost(n, c).latency_total for c in self.cost_lut.get_cores(n)]
            runtime = ceil(max(latencies)) if latencies else 0
            active_latency = get_active_latency(n, runtime, self.ssis)
            self.model.add_constr(self.slot_latency[s] >= active_latency, name=f"ssc_lat_{n.name}")

        for (tr, choice), y in self.y_path_choice.items():
            s = self.slot_of[tr]
            active_latency = self._active_transfer_latency(tr, choice, y)
            self.model.add_constr(self.slot_latency[s] >= active_latency, name=f"tr_lat_{tr.name}_{hash(choice)}")

    def _force_nonconstant_reuse_levels(self):
        """
        Forces the reuse level at the destination of a compute to compute transfer.
        It forces buffering up until the top irrelevant loop.
        """
        for tr in self.transfer_nodes:
            if tr.transfer_type not in (TransferType.COMPUTE_TO_COMPUTE):
                continue
            relevancies = self.ssis[tr].get_applicable_temporal_relevancies()
            last_irrelevant = -1
            for t in tr.outputs:
                for i, r in enumerate(relevancies):
                    if r is False:
                        last_irrelevant = i
                if last_irrelevant >= 0:
                    self.model.add_constr(
                        self.model.quicksum(self.z_stop[(t, s)]._raw for s in range(last_irrelevant, len(relevancies)))
                        >= 1,
                        name=f"force_intermediate_reuse_{tr.name}",
                    )

    def _force_final_output_reuse_levels(self):
        """
        Forces the reuse level of the outputs of the final compute node(s) by looking at COMPUTE_TO_MEM transfer inputs.
        It forces buffering up until the top irrelevant loop.
        """
        for tr in self.transfer_nodes:
            if tr.transfer_type not in (TransferType.COMPUTE_TO_MEM,):
                continue
            for t in tr.inputs:
                relevancies = self.ssis[t].get_applicable_temporal_relevancies()
                last_irrelevant = -1
                for i, r in enumerate(relevancies):
                    if r is False:
                        last_irrelevant = i
                if last_irrelevant >= 0:
                    self.model.add_constr(
                        self.model.quicksum(self.z_stop[(t, s)]._raw for s in range(last_irrelevant, len(relevancies)))
                        >= 1,
                        name=f"force_output_reuse_{tr.name}",
                    )

    # ...................... overlap + objective ................. #
    def _overlap_and_objective(self) -> None:
        max_s = self.max_slot
        big_m = self.big_m

        self._init_idle_indicators(max_s, big_m)
        self._create_idle_latency_vars(max_s)
        self._define_overlap_var()
        if self.constraint_selection.dma_channels:
            self._add_dma_usage_constraints()
        else:
            _logger.warning("ConstraintSelection: skipping dma_channels constraints")
        self._set_total_latency_and_objective()

    def _init_idle_indicators(self, max_s: int, big_m: int) -> None:
        self.idleS: dict[tuple[Resource, int], SolverVar] = {}
        self.idleE: dict[tuple[Resource, int], SolverVar] = {}
        self._init_link_idle_indicators(max_s, big_m)
        self._init_core_idle_indicators(max_s, big_m)

    def _init_link_idle_indicators(self, max_s: int, big_m: int) -> None:
        self.link_used: dict[CommunicationLink, SolverVar] = {}
        self.prefixs: dict[CommunicationLink, list[SolverVar]] = {}
        self.suffixs: dict[CommunicationLink, list[SolverVar]] = {}
        for link in self.link_set:
            active_s: dict[int, Any] = {}
            for s in range(max_s + 1):
                active_s[s] = self.model.quicksum(
                    self.y_path_choice[(tr, choice)]._raw
                    for (tr, choice) in self.y_path_choice
                    if link in self.links_in_choice[(tr, choice)] and self.slot_of[tr] == s
                )
            lu = self.model.add_var(vtype=SolverVarType.BINARY, name=f"linkUsed_{_resource_key(link)}")
            self.link_used[link] = lu
            sum_active = self.model.quicksum(active_s.values())
            self.model.add_constr(sum_active >= lu, name=f"link_used_def_{_resource_key(link)}")
            self.model.add_constr(sum_active <= big_m * lu, name=f"link_used_def2_{_resource_key(link)}")

            prefix = [
                self.model.add_var(vtype=SolverVarType.INTEGER, name=f"pre_{_resource_key(link)}_{s}")
                for s in range(max_s + 1)
            ]
            suffix = [
                self.model.add_var(vtype=SolverVarType.INTEGER, name=f"suf_{_resource_key(link)}_{s}")
                for s in range(max_s + 1)
            ]
            self.prefixs[link] = prefix
            self.suffixs[link] = suffix
            self.model.add_constr(prefix[0] == active_s[0])
            self.model.add_constr(suffix[-1] == active_s[max_s])
            for s in range(1, max_s + 1):
                self.model.add_constr(prefix[s] == prefix[s - 1] + active_s[s])
                self.model.add_constr(suffix[max_s - s] == suffix[max_s - s + 1] + active_s[max_s - s])

            for s in range(max_s + 1):
                is_ = self.model.add_var(vtype=SolverVarType.BINARY, name=f"idleS_{_resource_key(link)}_{s}")
                ie_ = self.model.add_var(vtype=SolverVarType.BINARY, name=f"idleE_{_resource_key(link)}_{s}")
                self.idleS[(link, s)] = is_
                self.idleE[(link, s)] = ie_

                self.model.add_constr(prefix[s] <= big_m * (1 - is_))
                self.model.add_constr(prefix[s] >= lu - big_m * is_)
                self.model.add_constr(suffix[s] <= big_m * (1 - ie_))
                self.model.add_constr(suffix[s] >= lu - big_m * ie_)
                self.model.add_constr(is_ >= 1 - lu)
                self.model.add_constr(ie_ <= lu)

    def _init_core_idle_indicators(self, max_s: int, big_m: int) -> None:
        # Collect the set of all compute cores from the (fixed) mapping
        core_set: set[Core] = set()
        for node in self.ssc_nodes:
            for group in self.mapping.get(node).resource_allocation:
                core_set.update(group)

        # Build core → set of slots in which it is active
        core_active_slots: dict[Core, set[int]] = defaultdict(set)
        for node in self.ssc_nodes:
            s = self.slot_of[node]
            for group in self.mapping.get(node).resource_allocation:
                for core in group:
                    core_active_slots[core].add(s)

        for core in core_set:
            active_slots = core_active_slots[core]

            # active_s[s] is a constant 0/1 for compute cores (mapping is fixed)
            active_s: dict[int, int] = {s: (1 if s in active_slots else 0) for s in range(max_s + 1)}

            # lu: core is used (always 1 since we only iterate cores from the mapping)
            lu = self.model.add_var(vtype=SolverVarType.BINARY, name=f"coreUsed_{_resource_key(core)}")
            self.model.add_constr(lu == 1, name=f"core_used_def_{_resource_key(core)}")

            prefix = [
                self.model.add_var(vtype=SolverVarType.INTEGER, name=f"pre_{_resource_key(core)}_{s}")
                for s in range(max_s + 1)
            ]
            suffix = [
                self.model.add_var(vtype=SolverVarType.INTEGER, name=f"suf_{_resource_key(core)}_{s}")
                for s in range(max_s + 1)
            ]
            self.model.add_constr(prefix[0] == active_s[0])
            self.model.add_constr(suffix[-1] == active_s[max_s])
            for s in range(1, max_s + 1):
                self.model.add_constr(prefix[s] == prefix[s - 1] + active_s[s])
                self.model.add_constr(suffix[max_s - s] == suffix[max_s - s + 1] + active_s[max_s - s])

            for s in range(max_s + 1):
                is_ = self.model.add_var(vtype=SolverVarType.BINARY, name=f"idleS_{_resource_key(core)}_{s}")
                ie_ = self.model.add_var(vtype=SolverVarType.BINARY, name=f"idleE_{_resource_key(core)}_{s}")
                self.idleS[(core, s)] = is_
                self.idleE[(core, s)] = ie_

                self.model.add_constr(prefix[s] <= big_m * (1 - is_))
                self.model.add_constr(prefix[s] >= lu - big_m * is_)
                self.model.add_constr(suffix[s] <= big_m * (1 - ie_))
                self.model.add_constr(suffix[s] >= lu - big_m * ie_)
                self.model.add_constr(is_ >= 1 - lu)
                self.model.add_constr(ie_ <= lu)

    def _create_idle_latency_vars(self, max_s: int) -> None:
        self.idle_lat: dict[Resource, SolverVar] = {}

        # Safe upper bound for slot latency
        slot_latency_ub = 0
        for n in self.ssc_nodes:
            runtimes = [self.cost_lut.get_cost(n, c).latency_total for c in self.cost_lut.get_cores(n)]
            runtime = ceil(max(runtimes)) if runtimes else 0
            slot_latency_ub = max(slot_latency_ub, runtime)

        for tr in self.transfer_nodes:
            for choice in self._path_choices(tr):
                lat = ceil(self._transfer_latency_for_path(tr, choice))
                slot_latency_ub = max(slot_latency_ub, lat)

        for res in {r for r, _ in self.idleS} | {r for r, _ in self.idleE}:
            terms = []
            for s in range(max_s + 1):
                idle_s = self.idleS.get((res, s), None)
                idle_e = self.idleE.get((res, s), None)

                if idle_s is not None:
                    prod_s = self._add_binary_scaled_continuous(
                        binary_var=idle_s,
                        continuous_var=self.slot_latency[s],
                        continuous_ub=slot_latency_ub,
                        base_name=f"idleS_lat_{_resource_key(res)}_{s}",
                    )
                    terms.append(prod_s)

                if idle_e is not None:
                    prod_e = self._add_binary_scaled_continuous(
                        binary_var=idle_e,
                        continuous_var=self.slot_latency[s],
                        continuous_ub=slot_latency_ub,
                        base_name=f"idleE_lat_{_resource_key(res)}_{s}",
                    )
                    terms.append(prod_e)

            v = self.model.add_var(vtype=SolverVarType.INTEGER, name=f"idleLat_{_resource_key(res)}")
            self.model.add_constr(
                v == self.model.quicksum(t._raw for t in terms), name=f"idleLat_def_{_resource_key(res)}"
            )
            self.idle_lat[res] = v

    def _define_overlap_var(self) -> None:
        overlap = self.model.add_var(vtype=SolverVarType.INTEGER, name="overlap")
        self.overlap = overlap
        for v in self.idle_lat.values():
            self.model.add_constr(overlap <= v)

    def _transfer_dma_usage_expr(self, tr: TransferNode, core: Core):
        return self.model.quicksum(self._tensor_on_core_expr(t, core) for t in tr.tensors if isinstance(t, Tensor))

    def _add_dma_usage_constraints(self) -> None:
        """
        Directional DMA accounting for all on-chip cores.

        Incoming DMA on a core:
            based on transfer outputs allocated on that core.

        Outgoing DMA on a core:
            based on transfer inputs allocated on that core.

        Two counting modes are supported:
            - per-transfer counting
            - global per-tensor counting
        """
        self.core_dma_in: dict[Core, SolverVar] = {}
        self.core_dma_out: dict[Core, SolverVar] = {}

        dma_cores = self._all_dma_candidate_cores()

        for core in dma_cores:
            v_in = self.model.add_var(vtype=SolverVarType.INTEGER, name=f"coreDmaIn_{_resource_key(core)}")
            v_out = self.model.add_var(vtype=SolverVarType.INTEGER, name=f"coreDmaOut_{_resource_key(core)}")

            if self.DMA_COUNT_SAME_TENSOR_ON_CORE_ONCE_GLOBALLY:
                in_expr = self._global_incoming_dma_expr(core)
                out_expr = self._global_outgoing_dma_expr(core)
            else:
                in_expr = self.model.quicksum(self._transfer_incoming_dma_expr(tr, core) for tr in self.transfer_nodes)
                out_expr = self.model.quicksum(self._transfer_outgoing_dma_expr(tr, core) for tr in self.transfer_nodes)

            self.model.add_constr(
                v_in == in_expr,
                name=f"coreDmaInConstr_{_resource_key(core)}",
            )
            self.model.add_constr(
                v_out == out_expr,
                name=f"coreDmaOutConstr_{_resource_key(core)}",
            )

            self.core_dma_in[core] = v_in
            self.core_dma_out[core] = v_out

        self.max_core_dma_in = self.model.add_var(vtype=SolverVarType.INTEGER, name="maxCoreDmaIn")
        self.max_core_dma_out = self.model.add_var(vtype=SolverVarType.INTEGER, name="maxCoreDmaOut")

        for core in dma_cores:
            self.model.add_constr(
                self.max_core_dma_in >= self.core_dma_in[core],
                name=f"maxCoreDmaIn_lb_{_resource_key(core)}",
            )
            self.model.add_constr(
                self.max_core_dma_out >= self.core_dma_out[core],
                name=f"maxCoreDmaOut_lb_{_resource_key(core)}",
            )

        # Optional hard architectural constraints through the context
        self.context.add_dma_usage_constraints(
            self.model,
            self.core_dma_in,
            self.core_dma_out,
        )

    def _set_total_latency_and_objective(self) -> None:
        self.total_lat = self.model.add_var(vtype=SolverVarType.INTEGER, name="total_latency")
        self.total_latency = self.total_lat
        assert self.overlap is not None, "Overlap variable must be initialized before objective."
        self.model.add_constr(
            self.total_lat
            == self.iterations * self.model.quicksum(v._raw for v in self.slot_latency.values())
            - (self.iterations - 1) * self.overlap
        )
        if self.constraint_selection.dma_channels:
            obj_func = self.total_lat._raw + self.max_core_dma_in._raw + self.max_core_dma_out._raw
        else:
            obj_func = self.total_lat._raw
        self.model.set_objective(obj_func, sense="minimize")

    # ------------------------------------------------------------------ #
    # public solve()                                                     #
    # ------------------------------------------------------------------ #
    def solve(
        self, *, tee: bool = True
    ) -> tuple[TensorReuseLevels, TensorDepths, TensorAlloc, TransferAlloc, MemoryAlloc, int, int, int]:
        self.model.set_param(SolverParams.VERBOSITY, 1 if tee else 0)
        self.model.optimize(self._mip_progress_callback)
        if self.model.get_status() != "OPTIMAL":
            self.model.compute_iis()

            ilp_path = os.path.join(self.output_path, "model.ilp")
            self.model.write(ilp_path)
            raise RuntimeError(f"Gurobi did not find an optimal solution. IIS written to {ilp_path}")

        tensor_alloc = self.get_tensor_allocations()
        routing = self.get_transfer_routing()
        chosen_memory_cores = self.get_chosen_memory_cores()
        tensor_reuse_levels = self.get_tensor_reuse_levels()
        tensor_depths = self.get_tensor_depths()

        # Visualize the optimization progress
        self.plot_optimization_progress(
            show=False, save_path=os.path.join(self.output_path, "optimization_progress.png")
        )
        self.save_optimization_trace(os.path.join(self.output_path, "optimization_trace.yaml"))
        self.save_optimization_metrics(save_path=os.path.join(self.output_path, "optimization_metrics.yaml"))
        self.save_slot_latency_breakdown(save_path=os.path.join(self.output_path, "slot_latency_breakdown.yaml"))

        assert self.total_latency is not None, "Total latency variable was not created."
        total_latency = int(self.total_latency.X)
        overlap = int(self.overlap.X)
        latency_per_iteration = sum(slot_lat.X for slot_lat in self.slot_latency.values())
        return (
            tensor_reuse_levels,
            tensor_depths,
            tensor_alloc,
            routing,
            chosen_memory_cores,
            total_latency,
            overlap,
            int(latency_per_iteration),
        )

    def get_tensor_reuse_levels(
        self,
    ) -> TensorReuseLevels:
        reuse_levels: TensorReuseLevels = {}
        for t in self.workload.tensors:
            for stop in range(-1, len(self.ssis[t].get_applicable_temporal_variables())):
                if self.z_stop[(t, stop)].X > self.VAR_THRESHOLD:
                    reuse_levels[t] = stop
        return reuse_levels

    def get_tensor_depths(
        self,
    ) -> TensorDepths:
        tiles_needed: TensorDepths = {}
        for t in self.workload.tensors:
            for stop in range(-1, len(self.ssis[t].get_applicable_temporal_variables())):
                if self.z_stop[(t, stop)].X > self.VAR_THRESHOLD:
                    tiles_needed[t] = self.tiles_needed_levels[(t, stop)]
        return tiles_needed

    def get_transfer_routing(self) -> TransferAlloc:
        routing: TransferAlloc = {}
        for tr in self.transfer_nodes:
            chosen = [
                choice
                for choice in self.possible_transfer_allocations[tr]
                if self.y_path_choice[(tr, choice)].X > self.VAR_THRESHOLD
            ]
            if len(chosen) != 1:
                raise ValueError(f"{tr.name}: expected exactly one routing choice, got {chosen}")
            routing[tr] = chosen[0]
        return routing

    def get_chosen_memory_cores(self) -> MemoryAlloc:
        chosen_memory_cores: MemoryAlloc = {}
        tensor_alloc = self.get_tensor_allocations()
        for tr in self.transfer_nodes:
            if not self._is_const_io(tr):
                continue
            tensor = self._constant_transfer_tensor(tr)

            if tensor in tensor_alloc:
                chosen_memory_cores[tr] = tensor_alloc[tensor]
            else:
                chosen_memory_cores[tr] = self._fixed_tensor_choice(tensor)
        return chosen_memory_cores

    def update_transfer_memory_core_allocation(self) -> None:
        chosen_memory_cores = self.get_chosen_memory_cores()
        for tr, cores in chosen_memory_cores.items():
            self.mapping.update_memory_allocation_for_node(tr, (cores,))

    def get_tensor_allocations(self) -> TensorAlloc:
        tensor_alloc: TensorAlloc = {}
        for t in self.tensor_fixed:
            tensor_alloc[t] = self._fixed_tensor_choice(t)
        for t in self.tensor_var:
            chosen = [
                choice
                for choice in self.possible_tensor_allocations[t]
                if self.x_tensor_choice[(t, choice)].X > self.VAR_THRESHOLD
            ]
            if len(chosen) != 1:
                raise ValueError(f"{t.node_name}: expected exactly one placement choice, got {chosen}")
            tensor_alloc[t] = chosen[0]
        return tensor_alloc

    def _check_io_transfers_firing_levels(self) -> None:
        for tr in self.transfer_nodes:
            for t in tr.tensors:
                stop_max = len(self.ssis[t].get_applicable_temporal_variables())
                stop = next(s for s in range(-1, stop_max) if self.z_stop[(t, s)].X > self.VAR_THRESHOLD)
                assert stop >= 0

    def _retrieve_core_allocation(self, node: Node) -> tuple[tuple[Core, ...], ...]:
        if isinstance(node, InEdge):
            assert self.accelerator.offchip_core_id is not None
            return ((self.accelerator.get_core(self.accelerator.offchip_core_id),),)
        if isinstance(node, OutEdge):
            assert self.accelerator.offchip_core_id is not None
            return ((self.accelerator.get_core(self.accelerator.offchip_core_id),),)
        if isinstance(node, TransferNode):
            return self.mapping.get(node).memory_allocation
        return self.mapping.get(node).resource_allocation

    def _safe_name(self, name: str) -> str:
        """Return a sanitized, globally-unique variable/constraint name.

        Replaces whitespace and colons (which cause issues in some backends)
        then appends a numeric suffix when the same base name would be reused
        within a single model build.  This ensures MathOpt's uniqueness
        requirement is always satisfied even when the same tensor/core/stop
        triple appears across multiple loop iterations.
        """
        sanitized = str(name).replace(" ", "_").replace(":", "_")
        count = self._name_counter.get(sanitized, 0)
        self._name_counter[sanitized] = count + 1
        if count == 0:
            return sanitized
        return f"{sanitized}_{count}"

    def _add_const_over_linexpr(
        self,
        *,
        numerator: float,
        denominator_expr: Any,
        base_name: str,
        denominator_lb: float,
        result_lb: float = 0.0,
        denominator_ub: float | None = None,
        result_ub: float | None = None,
    ) -> tuple[SolverVar, SolverVar]:
        """Encode res = numerator / den using the backend's non-linear constraint.

        Requires a backend with supports_nonlinear=True (e.g. GurobiBackend).
        Returns (res, den).
        """
        assert denominator_lb > 0.0, "denominator_lb must be strictly positive"

        n = self._safe_name(base_name)

        den = self.model.add_var(
            vtype=SolverVarType.CONTINUOUS,
            lb=denominator_lb,
            ub=denominator_ub if denominator_ub is not None else self.model.INFINITY,
            name=f"{n}__den",
        )
        res = self.model.add_var(
            vtype=SolverVarType.CONTINUOUS,
            lb=result_lb,
            ub=result_ub if result_ub is not None else self.model.INFINITY,
            name=f"{n}__val",
        )

        self.model.add_constr(den == denominator_expr, name=f"{n}__def_den")
        self.model.add_genconstr_nl(res, float(numerator) / den._raw, name=f"{n}__def_div")

        return res, den

    def _add_const_over_discrete_denominators(
        self,
        *,
        numerator: float,
        selectors: list[tuple[SolverVar, float]],
        base_name: str,
    ) -> SolverVar:
        """Encode result = numerator / denominator for linear-only backends.

        Uses piecewise enumeration over discrete denominator values via one-hot
        z_stop selectors: result = sum(z_k * (numerator / d_k)).
        """
        n = self._safe_name(base_name)
        min_denom = min(d for _, d in selectors)
        result_ub = numerator / min_denom

        result = self.model.add_var(vtype=SolverVarType.CONTINUOUS, lb=0.0, ub=result_ub, name=f"{n}__val")
        self.model.add_constr(
            result._raw == self.model.quicksum(z._raw * (numerator / d) for z, d in selectors)._raw,
            name=f"{n}__def_div",
        )
        return result

    def _add_binary_scaled_continuous(
        self,
        *,
        binary_var: SolverVar,
        continuous_var: SolverVar,
        continuous_ub: float,
        base_name: str,
        result_lb: float = 0.0,
    ) -> SolverVar:
        """
        Create exact linearization of:
            z = binary_var * continuous_var

        Assumes:
            binary_var in {0,1}
            result_lb <= continuous_var
            0 <= continuous_var <= continuous_ub

        Returns:
            z
        """
        assert continuous_ub >= 0.0, "continuous_ub must be nonnegative"

        n = self._safe_name(base_name)

        z = self.model.add_var(
            vtype=SolverVarType.CONTINUOUS,
            lb=result_lb,
            ub=continuous_ub,
            name=f"{n}__prod",
        )

        self.model.add_constr(z <= continuous_var, name=f"{n}__prod_ub1")
        self.model.add_constr(z <= continuous_ub * binary_var, name=f"{n}__prod_ub2")
        self.model.add_constr(
            z >= continuous_var - continuous_ub * (1 - binary_var),
            name=f"{n}__prod_lb1",
        )
        self.model.add_constr(z >= 0.0, name=f"{n}__prod_lb2")

        return z

    def _add_binary_times_const_over_linexpr(
        self,
        *,
        binary_var: SolverVar,
        numerator: float,
        denominator_expr: Any,
        denominator_lb: float,
        base_name: str,
        denominator_ub: float | None = None,
        selectors: list[tuple[SolverVar, float]] | None = None,
    ) -> SolverVar:
        assert denominator_lb > 0.0

        result_ub = float(numerator) / denominator_lb

        if self.model.supports_nonlinear:
            ratio_var, _ = self._add_const_over_linexpr(
                numerator=numerator,
                denominator_expr=denominator_expr,
                base_name=base_name,
                denominator_lb=denominator_lb,
                denominator_ub=denominator_ub,
                result_lb=0.0,
                result_ub=result_ub,
            )
        else:
            assert selectors is not None, "selectors required for linear-only backends"
            ratio_var = self._add_const_over_discrete_denominators(
                numerator=numerator,
                selectors=selectors,
                base_name=base_name,
            )

        return self._add_binary_scaled_continuous(
            binary_var=binary_var,
            continuous_var=ratio_var,
            continuous_ub=result_ub,
            base_name=f"{base_name}__gated",
        )

    def _add_binary_product(
        self,
        *,
        a: SolverVar,
        b: SolverVar,
        base_name: str,
    ) -> SolverVar:
        n = self._safe_name(base_name)
        w = self.model.add_var(vtype=SolverVarType.BINARY, name=f"{n}__and")
        self.model.add_constr(w <= a, name=f"{n}__ub1")
        self.model.add_constr(w <= b, name=f"{n}__ub2")
        self.model.add_constr(w >= a + b - 1, name=f"{n}__lb")
        return w

    def _active_transfer_latency(
        self,
        tr: TransferNode,
        choice: MulticastPathPlan,
        y: SolverVar,
    ) -> SolverVar:
        if (tr, choice) in self._transfer_latency_cache:
            active_latency_absent_loops_and_reuse_factor = self._transfer_latency_cache[(tr, choice)]
        else:
            latency_constant = float(self._transfer_latency_for_path(tr, choice))
            active_latency_absent_loops = get_active_latency(tr, latency_constant, self.ssis)

            reuse_factor_expr = self.reuse_factors[tr]._raw

            t = tr.outputs[0]
            applicable_temporal = self.ssis[t].get_applicable_temporal_variables()
            selectors = [
                (self.z_stop[(t, s)], float(self.reuse_levels[(t, s)][1])) for s in range(-1, len(applicable_temporal))
            ]

            active_latency_absent_loops_and_reuse_factor = self._add_binary_times_const_over_linexpr(
                binary_var=y,
                numerator=active_latency_absent_loops,
                denominator_expr=reuse_factor_expr,
                denominator_lb=1.0,
                base_name=f"transfer_latency_{tr}",
                selectors=selectors,
            )
            self._transfer_latency_cache[(tr, choice)] = active_latency_absent_loops_and_reuse_factor

        return active_latency_absent_loops_and_reuse_factor

    def _active_compute_latency(
        self,
        n: ComputationNode,
        runtime_constant: float,
    ) -> int:
        # Get the temporal steady state fraction of 'ABSENT' loops
        ssis_t = self.ssis.get(n).get_temporal_variables()
        total_product = prod([ssis_var.size for ssis_var in ssis_t])
        product_without_absent = prod([ssis_var.size for ssis_var in ssis_t if ssis_var.effect != LoopEffect.ABSENT])
        fraction = product_without_absent / total_product if total_product > 0 else 1.0
        # Scale the runtime constant by the fraction to get the effective latency
        active_latency = int(round(runtime_constant * fraction))
        return active_latency

    def _mip_progress_callback(self, model, where):
        if where not in (GRB.Callback.MIP, GRB.Callback.MIPSOL, GRB.Callback.PRESOLVE):
            return

        if where == GRB.Callback.PRESOLVE:
            point = {
                "event": "PRESOLVE",
                "time": float(model.cbGet(GRB.Callback.RUNTIME)),
                "work": float(model.cbGet(GRB.Callback.WORK)),
                "rows_removed": int(model.cbGet(GRB.Callback.PRE_ROWDEL)),
                "cols_removed": int(model.cbGet(GRB.Callback.PRE_COLDEL)),
                "bound_changes": int(model.cbGet(GRB.Callback.PRE_BNDCHG)),
                "coeff_changes": int(model.cbGet(GRB.Callback.PRE_COECHG)),
            }
            self.optimization_trace.append(point)
            return

        if where == GRB.Callback.MIP:
            best = model.cbGet(GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
            nodlft = model.cbGet(GRB.Callback.MIP_NODLFT)
            itrcnt = model.cbGet(GRB.Callback.MIP_ITRCNT)
            cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            work = model.cbGet(GRB.Callback.WORK)
            event = "MIP"
        else:
            best = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
            nodlft = None
            itrcnt = None
            cutcnt = None
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            work = model.cbGet(GRB.Callback.WORK)
            event = "MIPSOL"

        max_val = 1e90
        if not math.isfinite(best) or abs(best) >= max_val:
            best = None
        else:
            best = float(best)

        if not math.isfinite(bound) or abs(bound) >= max_val:
            bound = None
        else:
            bound = float(bound)

        gap = None
        if best is not None and bound is not None:
            gap = abs(best - bound) / max(1.0, abs(best))

        point = {
            "event": event,
            "time": float(runtime),
            "work": float(work),
            "nodecnt": float(nodecnt) if nodecnt is not None else None,
            "nodlft": float(nodlft) if nodlft is not None else None,
            "itrcnt": float(itrcnt) if itrcnt is not None else None,
            "cutcnt": int(cutcnt) if cutcnt is not None else None,
            "best_obj": best,
            "best_bound": bound,
            "gap": gap,
        }

        self.optimization_trace.append(point)

    def save_optimization_metrics(self, save_path: str) -> None:
        """
        Dump a concise YAML summary of the Gurobi run alongside the progress plot.

        Headline fields (in order of relevance for a paper):
          - search:    nodes explored, simplex/barrier iterations  -> "how much was searched"
          - solution:  objective, best bound, MIP gap              -> "what was found / how tight"
          - effort:    runtime (s), work units                     -> "how expensive it was"
          - model:    variable / constraint / nonzero counts      -> "problem size"
          - trace:     per-event progress records (reuses self.optimization_trace)
        """

        def _attr(name: str) -> Any | None:
            """Return a Gurobi model attribute or None if unavailable post-solve."""
            # Access the underlying gurobipy model for Gurobi-specific attributes.
            # GurobiBackend stores the model as ._model; fall back gracefully for other backends.
            from stream.opt.solver import GurobiBackend  # noqa: PLC0415

            raw_model = self.model._model if isinstance(self.model, GurobiBackend) else None
            if raw_model is None:
                return None
            try:
                value = getattr(raw_model, name)
            except Exception:  # noqa: BLE001  # catches AttributeError and gp.GurobiError
                return None
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        status = _attr("Status")
        status_name = {
            GRB.LOADED: "LOADED",
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.WORK_LIMIT: "WORK_LIMIT",
        }.get(status, str(status))

        # Per-event trace, normalized to a compact form (drop None values)
        trace_records: list[dict[str, Any]] = []
        for rec in self.optimization_trace:
            entry: dict[str, Any] = {"time_s": rec.get("time"), "event": rec.get("event")}
            for src, dst in (
                ("best_obj", "incumbent"),
                ("best_bound", "best_bound"),
                ("gap", "gap"),
                ("nodecnt", "nodes"),
                ("cutcnt", "cuts"),
                ("work", "work"),
            ):
                val = rec.get(src)
                if isinstance(val, int | float) and math.isfinite(val):
                    entry[dst] = val
            trace_records.append(entry)

        metrics: dict[str, Any] = {
            "status": status_name,
            "search": {
                "nodes_explored": _attr("NodeCount"),
                "simplex_iterations": _attr("IterCount"),
                "barrier_iterations": _attr("BarIterCount"),
            },
            "solution": {
                "objective": _attr("ObjVal"),
                "best_bound": _attr("ObjBound"),
                "mip_gap": _attr("MIPGap"),
            },
            "effort": {
                "runtime_s": _attr("Runtime"),
                "work_units": _attr("Work"),
            },
            "model": {
                "variables": {
                    "total": _attr("NumVars"),
                    "integer": _attr("NumIntVars"),
                    "binary": _attr("NumBinVars"),
                },
                "constraints": {
                    "linear": _attr("NumConstrs"),
                    "general": _attr("NumGenConstrs"),
                },
                "nonzeros": _attr("NumNZs"),
            },
            "trace": trace_records,
        }

        with open(save_path, "w") as fh:
            yaml.safe_dump(metrics, fh, sort_keys=False, default_flow_style=False)
        _logger.info("Optimization metrics saved to %s", save_path)

    def save_slot_latency_breakdown(self, save_path: str) -> None:  # noqa: PLR0915, PLR0912
        """Dump a debug-friendly per-slot latency breakdown next to the metrics yaml.

        For every slot lists the compute and transfer contributors with the
        intermediate values that make up its slot_latency constraint:
          - compute: LUT latency_total, SSIS fraction, active_latency
          - transfer: tensor bits, min link bandwidth, raw path cycles,
                      active_latency (absent-loop scaled), reuse_factor,
                      final contribution to slot_latency

        Best-effort: any per-node failure is silently skipped, and the whole
        method swallows top-level errors so it never blocks the pipeline.
        """

        def _scalar(v: Any) -> Any:
            if v is None or isinstance(v, bool | str):
                return v
            try:
                f = float(v)
            except (TypeError, ValueError):
                return str(v)
            if not math.isfinite(f):
                return str(v)
            if f.is_integer():
                return int(f)
            return f

        try:
            breakdown: dict[int, dict[str, Any]] = {}
            for s, lat_var in self.slot_latency.items():
                slot_val: float | None
                try:
                    slot_val = float(lat_var.X)
                except Exception:
                    slot_val = None
                breakdown[int(s)] = {
                    "slot_latency_cycles": _scalar(slot_val),
                    "compute_contributors": [],
                    "transfer_contributors": [],
                }

            # ── Compute contributors ── #
            for n in self.ssc_nodes:
                try:
                    s = int(self.slot_of[n])
                    cores = self.cost_lut.get_cores(n)
                    latencies = [self.cost_lut.get_cost(n, c).latency_total for c in cores]
                    runtime = ceil(max(latencies)) if latencies else 0
                    active = get_active_latency(n, float(runtime), self.ssis)
                    fraction: float | None = None
                    try:
                        ssis_t = self.ssis.get(n).get_temporal_variables()
                        total = prod([v.size for v in ssis_t])
                        present = prod([v.size for v in ssis_t if v.effect != LoopEffect.ABSENT])
                        fraction = (present / total) if total > 0 else 1.0
                    except Exception:
                        fraction = None
                    breakdown.setdefault(
                        s,
                        {"slot_latency_cycles": None, "compute_contributors": [], "transfer_contributors": []},
                    )
                    breakdown[s]["compute_contributors"].append(
                        {
                            "name": getattr(n, "name", str(n)),
                            "n_cores_in_lut": len(cores),
                            "lut_latency_total": _scalar(runtime),
                            "ssis_fraction": _scalar(fraction),
                            "active_latency": _scalar(active),
                        }
                    )
                except Exception:
                    continue

            # ── Transfer contributors (only the chosen path per transfer) ── #
            for (tr, choice), y in self.y_path_choice.items():
                try:
                    if float(y.X) < 0.5:  # noqa: PLR2004
                        continue
                    s = int(self.slot_of[tr])
                    raw = int(self._transfer_latency_for_path(tr, choice))
                    active_abs = int(get_active_latency(tr, float(raw), self.ssis))
                    try:
                        reuse_factor = float(self.reuse_factors[tr].getValue())
                    except Exception:
                        reuse_factor = None
                    cached_var = self._transfer_latency_cache.get((tr, choice))
                    try:
                        contribution = float(cached_var.getValue()) if cached_var is not None else None
                    except Exception:
                        contribution = None
                    tensor_bits: int | None = None
                    try:
                        if tr.inputs:
                            tensor_bits = int(tr.inputs[0].size_bits())
                    except Exception:
                        tensor_bits = None
                    min_bw: int | None = None
                    try:
                        if choice and choice.links_used:
                            min_bw = int(min(link.bandwidth for link in choice.links_used))
                    except Exception:
                        min_bw = None
                    breakdown.setdefault(
                        s,
                        {"slot_latency_cycles": None, "compute_contributors": [], "transfer_contributors": []},
                    )
                    breakdown[s]["transfer_contributors"].append(
                        {
                            "name": getattr(tr, "name", str(tr)),
                            "tensor_bits": _scalar(tensor_bits),
                            "min_link_bw": _scalar(min_bw),
                            "raw_path_cycles": _scalar(raw),
                            "active_latency_absent_loops": _scalar(active_abs),
                            "reuse_factor": _scalar(reuse_factor),
                            "contribution": _scalar(contribution),
                        }
                    )
                except Exception:
                    continue

            # ── Top-level totals ── #
            try:
                latency_per_iteration = sum(float(v.X) for v in self.slot_latency.values())
            except Exception:
                latency_per_iteration = None
            try:
                overlap_val = float(self.overlap.X) if self.overlap is not None else None
            except Exception:
                overlap_val = None
            try:
                total_latency_val = float(self.total_latency.X) if self.total_latency is not None else None
            except Exception:
                total_latency_val = None
            iter_step_val: float | None
            if latency_per_iteration is not None and overlap_val is not None:
                iter_step_val = latency_per_iteration - overlap_val
            else:
                iter_step_val = None

            summary = {
                "totals": {
                    "latency_per_iteration": _scalar(latency_per_iteration),
                    "overlap": _scalar(overlap_val),
                    "iter_step": _scalar(iter_step_val),
                    "total_latency": _scalar(total_latency_val),
                },
                "slots": [{"slot": s, **breakdown[s]} for s in sorted(breakdown)],
            }

            with open(save_path, "w") as fh:
                yaml.safe_dump(summary, fh, sort_keys=False, default_flow_style=False)
            _logger.info("Slot latency breakdown saved to %s", save_path)
        except Exception as e:
            # Never block the pipeline on this debug artifact.
            _logger.warning("save_slot_latency_breakdown failed: %s", e)

    def plot_optimization_progress(  # noqa: PLR0912, PLR0915
        self,
        *,
        save_path: str | None = None,
        show: bool = True,
        figsize: tuple[float, float] = (10.0, 6),
        show_work_subplot: bool = False,
    ) -> None:
        """
        Plot optimization progress recorded in self.optimization_trace.

        Top subplot:
        - best incumbent
        - best bound
        - relative gap (%) on a secondary y-axis

        Bottom subplot (optional):
        - solver work (if available)
        - optionally explored nodes / cuts on a secondary y-axis

        Args:
            save_path: Path to save the figure. If None, figure is not saved.
            show: Whether to display the figure.
            figsize: Figure size as (width, height).
            show_work_subplot: Whether to include the bottom work subplot.

        Expects callback records like:
            {
                "event": "MIP" or "MIPSOL",
                "time": float,
                "nodecnt": float | None,
                "best_obj": float | None,
                "best_bound": float | None,
                "gap": float | None,
                "work": float | None,
                "cutcnt": float | None,
            }
        """

        if not hasattr(self, "optimization_trace") or not self.optimization_trace:
            # Non-Gurobi backends do not populate optimization_trace via the callback.
            # Silently skip plotting rather than raising so OR-Tools solves succeed.
            return

        def _is_finite_number(x) -> bool:
            return x is not None and isinstance(x, int | float) and math.isfinite(x)

        trace = [
            rec
            for rec in self.optimization_trace
            if _is_finite_number(rec.get("time"))
            and (
                _is_finite_number(rec.get("best_obj"))
                or _is_finite_number(rec.get("best_bound"))
                or _is_finite_number(rec.get("gap"))
                or _is_finite_number(rec.get("work"))
                or _is_finite_number(rec.get("nodecnt"))
                or _is_finite_number(rec.get("cutcnt"))
            )
        ]

        if not trace:
            raise ValueError("Optimization trace exists, but it does not contain plottable finite values.")

        trace.sort(key=lambda r: (float(r["time"]), 0 if r.get("event") == "MIPSOL" else 1))

        times: list[float] = []
        incumbent: list[float] = []
        bound: list[float] = []
        gap_pct: list[float] = []

        works: list[float] = []
        nodes: list[float] = []
        cuts: list[float] = []

        last_best_obj: float | None = None
        last_best_bound: float | None = None
        last_gap: float | None = None
        last_work: float | None = None
        last_nodecnt: float | None = None
        last_cutcnt: float | None = None

        for rec in trace:
            t = float(rec["time"])

            if _is_finite_number(rec.get("best_obj")):
                last_best_obj = float(rec["best_obj"])
            if _is_finite_number(rec.get("best_bound")):
                last_best_bound = float(rec["best_bound"])
            if _is_finite_number(rec.get("gap")):
                last_gap = 100.0 * float(rec["gap"])

            if _is_finite_number(rec.get("work")):
                last_work = float(rec["work"])
            if _is_finite_number(rec.get("nodecnt")):
                last_nodecnt = float(rec["nodecnt"])
            if _is_finite_number(rec.get("cutcnt")):
                last_cutcnt = float(rec["cutcnt"])

            times.append(t)
            incumbent.append(float("nan") if last_best_obj is None else last_best_obj)
            bound.append(float("nan") if last_best_bound is None else last_best_bound)
            gap_pct.append(float("nan") if last_gap is None else last_gap)

            works.append(float("nan") if last_work is None else last_work)
            nodes.append(float("nan") if last_nodecnt is None else last_nodecnt)
            cuts.append(float("nan") if last_cutcnt is None else last_cutcnt)

        num_subplots = 2 if show_work_subplot else 1
        height_ratios = [2.0, 1.2] if show_work_subplot else [1.0]

        fig, axes = plt.subplots(
            num_subplots,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )

        if num_subplots == 1:
            ax1 = axes
        else:
            ax1, ax3 = axes

        # Top subplot: original functionality unchanged
        ax2 = ax1.twinx()

        line_inc = ax1.step(times, incumbent, where="post", label="Best incumbent")
        line_bnd = ax1.step(times, bound, where="post", label="Best bound")
        line_gap = ax2.plot(times, gap_pct, label="Gap (%)", linestyle="--")

        ax1.set_ylabel("Objective")
        ax2.set_ylabel("Gap (%)")
        ax1.set_title("Gurobi optimization progress")
        ax1.grid(True, alpha=0.3)

        handles_top = line_inc + line_bnd + line_gap
        labels_top = [h.get_label() for h in handles_top]
        ax1.legend(handles_top, labels_top, loc="best")

        # Bottom subplot: work done (optional)
        if show_work_subplot:
            ax4 = ax3.twinx()
            handles_bottom = []

            if any(not math.isnan(x) for x in works):
                line_work = ax3.step(times, works, where="post", label="Solver work")
                handles_bottom += line_work

            if any(not math.isnan(x) for x in nodes):
                line_nodes = ax4.step(times, nodes, where="post", label="Explored nodes", linestyle="--")
                handles_bottom += line_nodes

            if any(not math.isnan(x) for x in cuts):
                line_cuts = ax4.step(times, cuts, where="post", label="Cuts applied", linestyle=":")
                handles_bottom += line_cuts

            ax3.set_xlabel("Runtime (s)")
            ax3.set_ylabel("Work units")
            ax4.set_ylabel("Nodes / cuts")
            ax3.grid(True, alpha=0.3)

            if handles_bottom:
                labels_bottom = [h.get_label() for h in handles_bottom]
                ax3.legend(handles_bottom, labels_bottom, loc="best")
        else:
            ax1.set_xlabel("Runtime (s)")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            _logger.info("Optimization progress plot saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def save_optimization_trace(self, file_path: str) -> None:
        """
        Save the optimization trace to a YAML file.

        Writes a single chronological ``trace`` list. Each entry represents a
        point where something changed and has the following fields:

        - ``time_s``     – solver runtime in seconds
        - ``event``      – ``"MIPSOL"`` (new incumbent found) or ``"MIP"`` (bound update)
        - ``incumbent``  – present only on ``MIPSOL`` entries (when best_obj improved)
        - ``best_bound`` – present only when the bound changed
        - ``gap``        – relative gap at this point (when both values are available)

        Args:
            file_path: Destination path for the YAML file (e.g. "trace.yaml").
        """
        if not hasattr(self, "optimization_trace") or not self.optimization_trace:
            # Non-Gurobi backends do not populate optimization_trace via the callback.
            # Silently skip saving rather than raising so OR-Tools solves succeed.
            return

        def _fin(x) -> bool:
            return x is not None and isinstance(x, int | float) and math.isfinite(x)

        # Sort by time, MIPSOL first when times are equal (mirrors plot logic)
        sorted_trace = sorted(
            self.optimization_trace,
            key=lambda r: (float(r["time"]), 0 if r.get("event") == "MIPSOL" else 1),
        )

        entries: list[dict] = []
        last_obj: float | None = None
        last_bound: float | None = None

        for rec in sorted_trace:
            if not _fin(rec.get("time")):
                continue

            t = float(rec["time"])
            obj = float(rec["best_obj"]) if _fin(rec.get("best_obj")) else None
            bnd = float(rec["best_bound"]) if _fin(rec.get("best_bound")) else None
            gap = float(rec["gap"]) if _fin(rec.get("gap")) else None

            incumbent_improved = obj is not None and obj != last_obj
            bound_changed = bnd is not None and bnd != last_bound

            if not incumbent_improved and not bound_changed:
                continue

            entry: dict = {"time_s": t, "event": rec.get("event", "MIP")}
            if incumbent_improved:
                entry["incumbent"] = obj
                last_obj = obj
            if bound_changed:
                entry["best_bound"] = bnd
                last_bound = bnd
            if gap is not None:
                entry["gap"] = gap

            entries.append(entry)

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump({"trace": entries}, f, default_flow_style=False, sort_keys=False)

        _logger.info("Optimization trace saved to %s", file_path)
