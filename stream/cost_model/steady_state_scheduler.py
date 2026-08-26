import logging
import os
from dataclasses import replace
from itertools import combinations
from math import ceil, prod
from typing import cast

from xdsl.ir.affine import AffineMap

# if TYPE_CHECKING:
from stream.cost_model.communication_manager import MulticastPathPlan
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.datatypes import InterCoreTiling, LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.opt.allocation.constraint_optimization.context import build_transfer_context
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import (
    MemoryAlloc,
    TensorDepths,
    TensorReuseLevels,
    TransferAlloc,
    TransferAndTensorAllocator,
)
from stream.opt.solver import ConstraintSelection, SolveStats
from stream.visualization.steady_state_trace import export_steady_state_trace
from stream.workload.node import (
    ComputationNode,
    HasInputs,
    HasIterationSpace,
    HasOutputs,
    InEdge,
    Node,
    OutEdge,
    Tensor,
    TransferNode,
    TransferType,
)
from stream.workload.steady_state.computation import SteadyStateComputation
from stream.workload.steady_state.iteration_space import (
    IterationVariable,
    IterationVariableType,
    LoopEffect,
    Reuse,
    SteadyStateIterationSpace,
)
from stream.workload.utils import (
    generate_steady_state_iteration_spaces,
    get_compute_predecessors_successors,
    get_equivalent_dimension,
    get_node_with_largest_resource_allocation,
    is_mac_operator_type,
    is_reused_on_chip,
)
from stream.workload.workload import Workload

logger = logging.getLogger(__name__)

#: Nest depth of each steady-state loop kind, outermost first.
_LOOP_NEST_DEPTH: dict[str, int] = {
    "temporal": 0,
    "spatiotemporal": 1,
    "spatial": 2,
    "core_temporal": 3,
    "core_spatial": 4,
    "kernel": 5,
}

#: Core kinds that model a memory/DMA endpoint, not a compute engine -- never in a compute roofline.
_NON_COMPUTE_CORE_TYPES: frozenset[str] = frozenset({"offchip", "shim", "memory"})


def largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of ``n`` at most ``cap`` (never below 1)."""
    cap = max(1, min(cap, n))
    for candidate in range(cap, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


class SteadyStateScheduler:
    def __init__(  # noqa: PLR0913
        self,
        workload: Workload,
        accelerator: "Accelerator",
        mapping: Mapping,
        fusion_splits: dict[LayerDim, int],
        cost_lut: CoreCostLUT,
        nb_cols_to_use: int = 4,
        output_path: str = "",
        backend: str = "ORTOOLS_GSCIP",
        constraint_selection: ConstraintSelection | None = None,
        total_mac_ops: int | None = None,
    ):
        """
        Initialize the SteadyStateScheduler with the allocation and accelerator.

        Args:
            workload (ComputationNodeWorkload): The workload to be scheduled.
            total_mac_ops: Total multiply-accumulate ops of the untiled fusion group, used to report
                end-to-end MAC utilization. None disables that stat.
        """
        self.workload = workload  # Only contains nodes that are part of the current fusion stack
        self.accelerator = accelerator
        self.mapping = mapping
        self.fusion_splits = fusion_splits
        self.cost_lut = cost_lut
        self.partitioned_nodes: dict[ComputationNode, list[SteadyStateComputation]] = {}
        self.constant_tensors: dict[int, InEdge | OutEdge] = {}
        self.ssw: Workload | None = None

        # Cost model parameters
        self.latency_total = -1
        self.latency_per_iteration = -1
        self.overlap_between_iterations = -1
        self.performance_stats: dict | None = None
        self.tensor_depths: TensorDepths = {}

        self.nb_cols_to_use = nb_cols_to_use
        self.transfer_context = build_transfer_context(accelerator, nb_cols_to_use=nb_cols_to_use)
        self.backend = backend
        self.constraint_selection = constraint_selection
        self.total_mac_ops = total_mac_ops

        self.output_path = output_path
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)

        self.solve_stats: SolveStats | None = None

    def get_ir(self) -> dict:
        """Return a dictionary representation of the scheduler state for serialization/inspection.

        This captures:
        - Latency metrics (total, per-iteration, overlap)
        - Backend and constraint configuration used for the solve
        - Fusion splits applied
        - Mapping with node-to-resource allocations
        - Solve statistics (status, optimality gap, wall time)
        """
        stats = self.solve_stats
        solve_ir = (
            {
                "status": stats.status,
                "solver": stats.solver,
                "mip_gap": stats.mip_gap,
                "objective": stats.objective,
                "solve_time_s": stats.solve_time_s,
                "node_count": stats.node_count,
                "iteration_count": stats.iteration_count,
            }
            if stats is not None
            else None
        )
        cs = self.constraint_selection
        constraint_selection_ir = (
            {
                "memory_capacity": cs.memory_capacity,
                "object_fifo_depth": cs.object_fifo_depth,
                "buffer_descriptors": cs.buffer_descriptors,
                "dma_channels": cs.dma_channels,
            }
            if cs is not None
            else None
        )
        return {
            "latency": {
                "total": self.latency_total,
                "per_iteration": self.latency_per_iteration,
                "overlap_between_iterations": self.overlap_between_iterations,
            },
            "backend": self.backend,
            "solve": solve_ir,
            "constraint_selection": constraint_selection_ir,
            "fusion_splits": {str(dim): size for dim, size in self.fusion_splits.items()},
            "mapping": self.mapping.get_ir(),
            "performance": self.performance_stats,
            "steady_state": self._steady_state_ir(),
        }

    def _core_loops(self, cn: ComputationNode) -> list[dict]:
        """The loop nest inside one core (ZigZag mapping), as ``core_*`` loops; empty for a non-ZigZag core."""
        # Resolve by name -- the mapping is keyed by steady-state nodes, the cost LUT by the costed node.
        try:
            lut_node = next(n for n in self.cost_lut.get_nodes() if n.name == cn.name)
            allocation = self.mapping.get(lut_node).resource_allocation
            cores = [c for slot in (allocation or ()) for c in slot if isinstance(c, Core)]
            if not cores:
                return []
            entry = self.cost_lut.get_cost(lut_node, cores[0])
        except Exception:  # noqa: BLE001
            return []
        mapping = getattr(entry, "mapping", None)
        if mapping is None:
            return []

        loops: list[dict] = []

        def add(dim: str, size: int, kind: str) -> None:
            # No de-dup: ZigZag splits one dim over several levels, so equal-size loops are real levels.
            if int(size) > 1:
                loops.append({"dim": dim, "size": int(size), "type": kind, "node": cn.name})

        # ZigZag annotates the nest once per operand; take one operand's view (summing multiplies every dim).
        def one_operand(per_operand: dict) -> list:
            return next(iter(per_operand.values()), [])

        # Array unrollings first: these run in parallel, so they sit outside the temporal walk.
        for level in one_operand(getattr(mapping.spatial_mapping, "mapping_dict_origin", {})):
            for layer_dim, size in level:
                add(str(layer_dim), size, "core_spatial")
        for level in one_operand(getattr(mapping.temporal_mapping, "mapping_dic_stationary", {})):
            for layer_dim, size in level:
                add(str(layer_dim), size, "core_temporal")
        return loops

    def _steady_state_ir(self) -> dict | None:
        """Serialise the tiled/steady-state inspection view (operators, loop nest, transfer graph); None on failure."""
        try:
            operators = [
                {
                    "name": cn.name,
                    "op": getattr(cn, "type", "computation"),
                    "tensors": [{"name": t.name, "shape": [int(s) for s in t.shape]} for t in cn.tensors],
                }
                for cn in self.workload.get_computation_nodes()
            ]
            # The for-loop nest over the steady-state iteration space (deduped across operands, size > 1).
            loops: list[dict] = []
            seen: set = set()
            for ssis in (self.ssis or {}).values():
                for iv in ssis.variables:
                    # ABSENT: the node lacks the dim (unrolling replicates it); counting it double-counts one unrolling.
                    if iv.effect is LoopEffect.ABSENT:
                        continue
                    key = (str(iv.dimension), int(iv.size))
                    if int(iv.size) > 1 and key not in seen:
                        seen.add(key)
                        loops.append({"dim": str(iv.dimension), "size": int(iv.size), "type": iv.type.name.lower()})
            # Below the tile: expand each node's intra-core mapping per node (fused groups stay separate).
            expanded = False
            for cn in self.workload.get_computation_nodes():
                core_loops = self._core_loops(cn)
                for loop in core_loops:
                    loop["node"] = cn.name
                loops.extend(core_loops)
                expanded = expanded or bool(core_loops)
            if expanded:
                # Drop the kernel stand-in once expanded (it would double-count the intra-core work).
                loops = [loop for loop in loops if loop["type"] != "kernel"]

            def _nest_order(loop: dict) -> tuple[str, int]:
                return loop.get("node") or "", _LOOP_NEST_DEPTH.get(loop["type"], len(_LOOP_NEST_DEPTH))

            loops.sort(key=_nest_order)
            # The tiled workload graph WITH transfer nodes -- the tensor copies that reside on-chip.
            tiled_nodes: list[dict] = []
            edges: list[dict] = []
            if self.ssw is not None:
                for cn in self.ssw.get_computation_nodes():
                    tiled_nodes.append({"name": cn.name, "kind": "compute", "op": getattr(cn, "type", "computation")})
                for tn in self.ssw.get_transfer_nodes():
                    out = tn.outputs[0] if tn.outputs else None
                    transfer_type = getattr(tn, "transfer_type", None)
                    tiled_nodes.append(
                        {
                            "name": tn.name,
                            "kind": "transfer",
                            "transfer_type": getattr(transfer_type, "name", None),
                            "tensor": out.name if out is not None else None,
                            "elements": int(prod(out.shape)) if out is not None else 0,
                        }
                    )
                edges = [{"source": s.name, "target": t.name} for s, t in self.ssw.edges()]
            return {"operators": operators, "loops": loops, "tiled_graph": {"nodes": tiled_nodes, "edges": edges}}
        except Exception as exc:  # noqa: BLE001 -- inspection view must never break a solved run
            logger.warning("could not build steady-state IR: %s", exc)
            return None

    def run(self) -> Workload:
        """
        Run the steady state scheduler on the given workload.

        Returns:
            TimeSlotAllocation: The scheduled workload.
        """
        # Update the workload graph to include transfer nodes
        self.ssw = self.build_transfer_graph()
        # Update the fusion_splits based on the new workload with transfer nodes
        self.fusion_splits = self.update_fusion_splits()
        # Save the new workload with transfers
        # self.ssw.visualize(os.path.join(self.output_path, "tiled_workload_with_transfers.png"))
        # Update the mapping for the new workload graph
        self.mapping = self.update_mapping()
        # Update the cost lut for the new workload graph
        self.cost_lut = self.update_cost_lut()
        # Update the steady state iteration spaces to include transfer nodes and tensors
        self.ssis = self.generate_ssis()
        # Calculate the number of iterations based on the steady state iteration spaces
        self.iterations = self.calculate_iterations()
        # Calculate the multiplicity of each node's execution in the steady state workload
        multiplicities = self.calculate_multiplicities()
        # Get the timeslots for all nodes (resource-aware: same slot allowed iff a
        # disjoint core/link assignment exists across same-class nodes in that slot).
        timeslots = self.ssw.get_timeslots(self.mapping)
        # timeslots = self.ssw.get_timeslots_simple()  # baseline: one slot per node, no resource awareness
        # At this point, the only nodes without an allocation are the transfer nodes
        tta = TransferAndTensorAllocator(
            self.ssw,
            timeslots,
            accelerator=self.accelerator,
            iterations=self.iterations,
            ssis=self.ssis,
            multiplicities=multiplicities,
            mapping=self.mapping,
            cost_lut=self.cost_lut,
            nb_cols_to_use=self.nb_cols_to_use,
            context=self.transfer_context,
            output_path=self.output_path,
            backend=self.backend,
            constraint_selection=self.constraint_selection,
        )
        (
            tensor_reuse_levels,
            tensor_depths,
            tensor_allocations,
            transfer_allocations,
            memory_allocations,
            total_latency,
            overlap,
            latency_per_iteration,
        ) = tta.solve()
        # Capture solve statistics before tta goes out of scope (tta.model is a local variable)
        self.solve_stats = tta.model.solve_stats()
        # Capture the read-only performance summary while the solved tta is still in scope.
        try:
            self.performance_stats = tta.compute_performance_stats()
        except Exception as exc:  # observability must never break the solve
            logger.warning("Failed to compute performance stats: %s", exc)
        # total, per_iter, ov = tsa_upd.compute_latency(iterations=self.iterations, offchip_core_id=offchip_core_id)
        # assert total == total_latency_solver, (
        #     f"Calculated total latency {total} does not match total latency from solver {total_latency_solver}."
        # )
        self.latency_total, self.latency_per_iteration, self.overlap_between_iterations = (
            total_latency,
            latency_per_iteration,
            overlap,
        )
        # End-to-end MAC utilization: useful MACs vs the whole chip's peak over the full runtime
        # (so it folds in spatial fill, temporal stalls, idle cores AND transfer overhead). Purely
        # observational; never let it break the solve.
        try:
            self._augment_performance_stats_end_to_end()
        except Exception as exc:
            logger.warning("Failed to compute end-to-end MAC utilization: %s", exc)
        # Export Perfetto-compatible JSON traces of the solved schedule
        fname = ""
        trace_path = ""
        try:
            for compact, fname in [(True, "steady_state_trace_compact.json"), (False, "steady_state_trace.json")]:
                trace_path = export_steady_state_trace(
                    tta=tta,
                    iterations=self.iterations,
                    overlap=overlap,
                    latency_per_iteration=latency_per_iteration,
                    output_path=self.output_path,
                    compact=compact,
                    filename=fname,
                )
            logger.info("Steady-state schedule trace: %s", trace_path)
        except Exception as exc:  # never let a visualisation failure abort the run
            logger.warning("Failed to export steady-state trace (%s): %s", fname, exc)
        # Check that all nodes in the steady state workload have a chosen resource allocation
        # self.check_steady_state_workload_allocations(self.ssw)
        self.update_tensor_steady_state_iteration_spaces(tensor_reuse_levels)
        self.update_mapping_with_allocations(transfer_allocations, memory_allocations)
        self.ssw.visualize(os.path.join(self.output_path, "steady_state_workload_final.png"), self.mapping, self.ssis)
        # tla = TensorLifetimeAnalyzer(self.ssw)
        self.steady_state_workload = self.ssw
        return self.ssw

    def _mac_roofline_peak(self) -> tuple[int, int]:
        """``(peak_macs_per_cycle, n_cores)`` over the on-chip cores that may execute MAC work."""
        offchip_id = self.accelerator.offchip_core_id
        peak = 0
        n_cores = 0
        for core in self.accelerator.core_list:
            if core.id == offchip_id or core.type in _NON_COMPUTE_CORE_TYPES:
                continue
            op_types = getattr(core, "operator_types", None)
            if op_types is not None and not any(is_mac_operator_type(t) for t in op_types):
                continue
            units = getattr(getattr(core, "operational_array", None), "total_unit_count", 0) or 0
            if not units:
                continue
            peak += units
            n_cores += 1
        return peak, n_cores

    def _augment_performance_stats_end_to_end(self) -> None:
        """Add end-to-end MAC utilization (``total_mac_ops / (peak_macs_per_cycle * total_latency)``,
        both restricted to the matmul/conv family) to performance_stats['aggregate'] in place."""
        if not isinstance(self.performance_stats, dict):
            return
        agg = self.performance_stats.get("aggregate")
        if not isinstance(agg, dict):
            return
        peak, mac_cores = self._mac_roofline_peak()
        macs = self.total_mac_ops
        lat = self.latency_total
        util = (macs / (peak * lat)) if (macs and peak and lat and lat > 0) else None
        agg["total_mac_ops"] = macs
        agg["peak_macs_per_cycle"] = peak
        agg["mac_capable_cores"] = mac_cores
        agg["end_to_end_mac_utilization"] = util

    def update_tensor_steady_state_iteration_spaces(self, tensor_reuse_levels: TensorReuseLevels):
        for t, ssis in self.ssis.items():
            if isinstance(t, Tensor):
                assert t in tensor_reuse_levels, f"Tensor {t.name} does not have a reuse level assigned."
                reuse_level = tensor_reuse_levels[t]
                for i, iv in enumerate(ssis.get_applicable_temporal_variables()):
                    if i <= reuse_level:
                        iv.reuse = Reuse.REUSE
                    else:
                        iv.reuse = Reuse.NO_REUSE
        # Propagate spatial reuse across transfer boundaries: when one side of a
        # transfer has a SPATIAL variable that is represented as a SPATIOTEMPORAL on the
        # other side (same dimension and size), mark that spatiotemporal as REUSE so that
        # both endpoints display the same reuse boundary.
        for node in self.ssw.get_transfer_nodes():
            for src in node.inputs:
                for dst in node.outputs:
                    self._propagate_spatial_reuse(src, dst)
                    self._propagate_spatial_reuse(dst, src)
        # Mirror solved reuse from the moved tensor's SSIS (priced) onto each transfer's SSIS, by (dim, size).
        for node in self.ssw.get_transfer_nodes():
            governing = next(
                (t for t in (*node.outputs, *node.inputs) if isinstance(t, Tensor) and t in self.ssis), None
            )
            if governing is None:
                continue
            reuse_by_loop = {(v.dimension, v.size): v.reuse for v in self.ssis[governing].get_temporal_variables()}
            for iv in self.ssis[node].get_temporal_variables():
                if (iv.dimension, iv.size) in reuse_by_loop:
                    iv.reuse = reuse_by_loop[(iv.dimension, iv.size)]

    def _propagate_spatial_reuse(self, spatial_side: Tensor, temporal_side: Tensor) -> None:
        """Mark spatiotemporal variables on ``temporal_side`` as REUSE when they
        match (dimension, size) of an applicable spatial variable on ``spatial_side``."""
        if spatial_side not in self.ssis or temporal_side not in self.ssis:
            return
        spatial_keys_not_in_temporal = {
            (iv.dimension, iv.size)
            for iv in self.ssis[spatial_side].variables
            if iv.type == IterationVariableType.SPATIAL
            and iv.applicable
            and iv not in self.ssis[temporal_side].variables  # only look at temporal side vars that are not spatial
        }
        if not spatial_keys_not_in_temporal:
            return
        seen_spatial_keys = set()
        for iv in self.ssis[temporal_side].variables:
            is_spatiotemporal = iv.type in (IterationVariableType.SPATIOTEMPORAL,)
            match = (iv.dimension, iv.size) in spatial_keys_not_in_temporal
            not_seen = (iv.dimension, iv.size) not in seen_spatial_keys
            if is_spatiotemporal and match and not_seen:
                is_applicable = iv.applicable
                if is_applicable:  # Only set to reuse if it's applicable
                    iv.reuse = Reuse.REUSE
                seen_spatial_keys.add((iv.dimension, iv.size))

    def build_transfer_graph(self) -> Workload:
        new_nodes: dict[str, Node] = {node.name: node for node in self.workload.nodes}
        # Go through the tensors of the workload to find sources and destinations of the tensor
        for tensor in self.workload.tensors:
            srcs = [n for n in self.workload.nodes if isinstance(n, HasOutputs) and tensor in n.outputs]
            assert len(srcs) == 1, f"Expected exactly one source for tensor {tensor}, found {len(srcs)}"
            src = new_nodes[srcs[0].name]
            dsts = [new_nodes[n.name] for n in self.workload.nodes if isinstance(n, HasInputs) and tensor in n.inputs]
            is_constant_o_transfer = any(isinstance(dst, OutEdge) for dst in dsts)
            if is_constant_o_transfer and not isinstance(src, InEdge):
                self.add_two_transfer_nodes_for_constant_output_transfer(tensor, src, dsts, new_nodes)
            else:
                self.add_transfer_nodes(tensor, src, dsts, new_nodes)
        new_workload = Workload(new_nodes.values())
        return new_workload

    def add_transfer_nodes(self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs], new_nodes: dict[str, Node]):
        """
        Move ``tensor`` from its source to its destinations, either directly or staged on a memory tile.

        Staging splits the move in two -- source to the on-chip buffer, buffer to every destination --
        so the tile can hold the tensor across reads and re-lay it out on the way through. See
        :meth:`_stages_on_mem_tile`.
        """
        if not self._stages_on_mem_tile(tensor, src, dsts):
            transfer_type = self.determine_transfer_type(src, dsts)
            out_name = f"{tensor.name}_1"
            transfer_node, updated_tensors = self.generate_transfer_node(dsts, tensor, transfer_type, out_name)
            new_nodes[transfer_node.name] = transfer_node
            for dst, updated_tensor in zip(dsts, updated_tensors, strict=True):
                self.update_destination_node_inputs(tensor, src, new_nodes, dst, updated_tensor)
            return
        # First transfer node from source to on-chip buffer
        transfer_type_1 = self.determine_transfer_type(src, dsts, dst_type="memory")
        out_name_1 = f"{tensor.name}_1"
        transfer_node_1, updated_tensors_1 = self.generate_transfer_node([src], tensor, transfer_type_1, out_name_1)
        new_nodes[transfer_node_1.name] = transfer_node_1
        # Second transfer node from on-chip buffer to destinations
        out_name_2 = f"{tensor.name}_2"
        transfer_type_2 = self.determine_transfer_type(src, dsts, src_type="memory")
        transfer_node_2, updated_tensors_2 = self.generate_transfer_node(
            dsts, updated_tensors_1[0], transfer_type_2, out_name_2
        )
        new_nodes[transfer_node_2.name] = transfer_node_2
        for dst, updated_tensor in zip(dsts, updated_tensors_2, strict=True):
            self.update_destination_node_inputs(tensor, src, new_nodes, dst, updated_tensor)

    def _stages_on_mem_tile(self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs]) -> bool:
        """Whether a transfer is staged in a memory tile instead of landing straight on the cores.

        Two things ask for a staging buffer. An offchip input read more than once is held in the tile,
        which keeps the fan-out off the shim's two DMA channels: one channel feeds the tile, which then
        serves every core from its own. And a producer and consumer that disagree on layout need the
        tensor re-laid out between them, which is the tile's other job.
        """
        if not self._get_accelerator_memory_cores():
            return False
        if isinstance(src, InEdge):
            if self.transfer_context.force_io_transfers_on_mem_tile:
                return True
            return is_reused_on_chip(self.workload, tensor, dsts)
        produced = self._declared_layout(src, tensor)
        if produced is None:
            return False
        consumed = (self._declared_layout(dst, tensor) for dst in dsts)
        return any(layout is not None and layout != produced for layout in consumed)

    def _declared_layout(self, node: Node, tensor: Tensor):
        """The layout ``node``'s kernel declares for ``tensor``, None when it declares none.

        None is "not known", never "differs": a node without a kernel, a kernel that declares
        no layouts, and an operand past the end all leave the transfer unstaged.
        """
        kernel = self.mapping.get(node).kernel
        layouts = kernel.operand_layouts() if kernel else ()
        operands = (*getattr(node, "inputs", ()), *getattr(node, "outputs", ()))
        index = operands.index(tensor)
        return layouts[index] if index < len(layouts) else None

    def update_destination_node_inputs(self, tensor, src, new_nodes, dst, updated_tensor):
        # Find corresponding node in new_nodes as it might have already been updated
        dst_new = new_nodes[dst.name]
        # Update the dst input to the second transfer node
        assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
        input_idx = dst_new.inputs.index(tensor)
        new_inputs = dst_new.inputs[:input_idx] + (updated_tensor,) + dst_new.inputs[input_idx + 1 :]
        if isinstance(dst_new, ComputationNode):
            new_dst = replace(dst_new, inputs=new_inputs)
        elif isinstance(dst_new, OutEdge):
            new_dst = OutEdge(
                name=dst_new.name,
                inputs=new_inputs,
            )
        else:
            raise ValueError(f"Unexpected dst node type: {type(dst_new)}")
        new_nodes[dst_new.name] = new_dst
        # Update the mapping entry for this new_dst node to be the same as the original dst node
        self.mapping.set(new_dst, self.mapping.get(dst))
        # Remove the original dst node from the mapping as it has been updated with new inputs
        self.mapping.remove(dst)

    def add_two_transfer_nodes_for_constant_output_transfer(
        self, tensor: Tensor, src: HasOutputs, dsts: list[HasInputs], new_nodes: dict[str, Node]
    ):
        """
        For constant output transfers, we add two transfer nodes:
        - one from the source to the on-chip memory buffer,
        - a second one from the on-chip memory buffer to the destination.
        This is to ensure that the constant tensor is properly allocated in memory and can be reused across iterations.

        If the accelerator has no on-chip memory tiles (e.g. TPU-like hardware), falls back to a single
        direct transfer from the source to the destinations (COMPUTE_TO_MEM).
        """
        assert len(dsts) == 1, "Currently only support single destination for constant output transfer."
        dst = dsts[0]
        assert isinstance(dst, OutEdge), (
            f"Expected destination of constant transfer to be an OutEdge, found {type(dst)}"
        )
        # Fall back to a single direct transfer when no memory tiles are available
        if not self._get_accelerator_memory_cores():
            transfer_type = self.determine_transfer_type(src, dsts)
            new_tensor = self.generate_transfer_input_tensor(tensor, src, name_suffix="_1")
            out_name = f"{tensor.name}"
            transfer_node, updated_tensors = self.generate_transfer_node([dst], new_tensor, transfer_type, out_name)
            new_nodes[transfer_node.name] = transfer_node
            new_src = self.update_source_tensor(tensor, src, new_nodes, new_tensor)
            self.update_destination_tensor(tensor, new_src, new_nodes, dst, updated_tensors)
            return
        # First transfer node from source to on-chip buffer
        transfer_type_1 = self.determine_transfer_type(src, dsts, dst_type="memory")
        new_tensor = self.generate_transfer_input_tensor(tensor, src, name_suffix="_1")
        out_name_1 = f"{tensor.name}_2"
        transfer_node_1, updated_tensors_1 = self.generate_transfer_node([src], new_tensor, transfer_type_1, out_name_1)
        new_nodes[transfer_node_1.name] = transfer_node_1
        new_src = self.update_source_tensor(tensor, src, new_nodes, new_tensor)
        # Second transfer node from on-chip buffer to destination
        transfer_type_2 = self.determine_transfer_type(new_src, dsts, src_type="memory")
        out_name_2 = f"{tensor.name}"
        transfer_node_2, updated_tensors_2 = self.generate_transfer_node(
            [dst], updated_tensors_1[0], transfer_type_2, out_name_2
        )
        new_nodes[transfer_node_2.name] = transfer_node_2
        # Update the dst input to the second transfer node
        self.update_destination_tensor(tensor, new_src, new_nodes, dst, updated_tensors_2)

    def update_destination_tensor(self, tensor, src, new_nodes, dst, updated_tensors_2):
        dst_new = new_nodes[dst.name]
        assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
        input_idx = dst_new.inputs.index(tensor)
        new_inputs = dst_new.inputs[:input_idx] + (updated_tensors_2[0],) + dst_new.inputs[input_idx + 1 :]
        new_dst = OutEdge(
            name=dst_new.name,
            inputs=new_inputs,
        )
        new_nodes[new_dst.name] = new_dst
        # No need to update mapping of dst as it's an OutEdge

    def update_source_tensor(self, tensor, src, new_nodes, new_tensor) -> ComputationNode:
        output_idx = src.outputs.index(tensor)
        new_outputs = src.outputs[:output_idx] + (new_tensor,) + src.outputs[output_idx + 1 :]
        new_src = replace(src, outputs=new_outputs)
        new_nodes[new_src.name] = new_src
        # Update the mapping entry for this new_src node to be the same as the original src node
        self.mapping.set(new_src, self.mapping.get(src))
        # Remove the original src node from the mapping as it has been updated with new outputs
        self.mapping.remove(src)
        return new_src

    def update_fusion_splits(self) -> dict[LayerDim, int]:
        # Update the fusion_splits based on the new workload with transfer nodes
        updated_fusion_splits = {}
        for dim, size in self.fusion_splits.items():
            new_dim = get_equivalent_dimension(self.workload, self.ssw, dim)
            updated_fusion_splits[new_dim] = size
        return updated_fusion_splits

    def update_mapping(self):
        # Update inter_core_tiling of computation node to unique dimensions
        for node in self.ssw.get_computation_nodes():
            unique_dims_tiling = (self.ssw.get_unique_dims_inter_core_tiling(node, self.mapping),)
            self.mapping.update_inter_core_tiling(node, unique_dims_tiling)
        # Add transfer node mappings
        for node in self.ssw.get_transfer_nodes():
            assert len(node.inputs) == 1, "Transfer node must have exactly one input tensor."
            src = cast(HasOutputs, list(self.ssw.predecessors(node))[0])
            dsts = tuple(cast(HasInputs, n) for n in self.ssw.successors(node))
            self.update_mapping_for_transfer(node, src, dsts)
        return self.mapping.with_updated_workload(self.ssw, self.workload)  # updates FusedGroups

    def update_mapping_with_allocations(
        self,
        transfer_allocations: TransferAlloc,
        memory_allocations: MemoryAlloc,
    ):
        for tr, alloc in transfer_allocations.items():
            if tr in memory_allocations:
                assert isinstance(memory_allocations[tr], tuple)
                memory_allocation = memory_allocations[tr]
            else:
                memory_allocation = tuple()
            self.mapping.set_for_node(
                tr,
                resource_allocation=(alloc,),
                inter_core_tiling=tuple(),
                memory_allocation=memory_allocation,
            )
        for tr in self.ssw.get_transfer_nodes():
            assert len(self.mapping.get(tr).resource_allocation) == 1, (
                f"Transfer node {tr.name} should have exactly one resource allocation after update."
            )

    def update_cost_lut(self):
        # The new workload contains same computation node names but with different input tensors
        for new_node in self.ssw.get_computation_nodes():
            old_node = next(n for n in self.cost_lut.get_nodes() if n.name == new_node.name)
            self.cost_lut.replace_node(old_node, new_node)
        return self.cost_lut

    def generate_transfer_node(
        self, dsts: list[HasInputs], tensor: Tensor, transfer_type: TransferType, out_name: str = ""
    ) -> tuple[TransferNode, list[Tensor]]:
        transfer_outputs = self.generate_transfer_output_tensors(tensor, dsts, out_name)
        operand_mapping = tuple(AffineMap.identity(len(tensor.shape)) for _ in range(1 + len(dsts)))
        transfer_node = TransferNode(
            name=f"Transfer({tensor.name})",
            inputs=(tensor,),
            outputs=tuple(transfer_outputs),
            transfer_type=transfer_type,
            operand_mapping=operand_mapping,
        )
        return transfer_node, transfer_outputs

    def generate_transfer_input_tensor(self, tensor: Tensor, src: HasOutputs, name_suffix: str = "") -> Tensor:
        assert len(src.outputs) == 1, "Src must have exactly one output tensor for index below."
        input_tensor = Tensor(
            name=f"{tensor.name}{name_suffix}",
            operand_type=tensor.operand_type,
            shape=tensor.shape,
            subview=tensor.subview,
        )
        return input_tensor

    def generate_transfer_output_tensors(
        self, tensor: Tensor, dsts: list[HasInputs], out_name: str = ""
    ) -> list[Tensor]:
        transfer_outputs = []
        for i, _ in enumerate(dsts):
            suffix = f".{i}" if len(dsts) > 1 else ""
            transfer_output = Tensor(
                name=f"{out_name}{suffix}",
                operand_type=tensor.operand_type,
                shape=tensor.shape,
                subview=tensor.subview,
            )
            transfer_outputs.append(transfer_output)
        return transfer_outputs

    def generate_ssis(self) -> dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]:
        ssis = generate_steady_state_iteration_spaces(
            self.ssw,
            self.mapping,
            self.fusion_splits,
        )
        ssis = self.update_tensor_ssis(self.ssw, ssis)
        return ssis

    def update_tensor_ssis(
        self, workload: Workload, ssis: dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]
    ) -> dict[HasIterationSpace | Tensor, SteadyStateIterationSpace]:
        # Generate the tensor SSIS of InEdge(s) output
        for in_edge in workload.get_in_edges():
            for tensor in in_edge.outputs:
                assert tensor not in ssis, (
                    f"Tensor {tensor.name} already has an SSIS, cannot assign the same tensor multiple SSIS."
                )
                succ = next(workload.successors(in_edge))
                tensor_ssis = self.generate_tensor_ssis(workload, tensor, succ, ssis)
                ssis[tensor] = tensor_ssis
        # Generate the new tensor SSIS of node outputs
        for node in workload.get_iteration_space_nodes():
            for tensor in node.outputs:
                assert tensor not in ssis, (
                    f"Tensor {tensor.name} already has an SSIS, cannot assign the same tensor multiple SSIS."
                )
                tensor_ssis = self.generate_tensor_ssis(workload, tensor, node, ssis)
                ssis[tensor] = tensor_ssis
        return ssis

    def generate_tensor_ssis(
        self,
        workload: Workload,
        tensor: Tensor,
        node: HasIterationSpace,
        ssis: dict[HasIterationSpace | Tensor, SteadyStateIterationSpace],
    ) -> SteadyStateIterationSpace:
        producer_ssis = ssis.get(node, None)
        if producer_ssis is None:
            raise KeyError(f"Node {node.name} does not have a valid producer SSIS.")
        tensor_dims = workload.get_tensor_dimensions(tensor)
        tensor_ivs = []
        for prod_iv in producer_ssis.variables:
            prod_iv_dim = prod_iv.dimension
            if prod_iv_dim in tensor_dims:
                tensor_effect = LoopEffect.VARYING
            else:
                tensor_effect = LoopEffect.ABSENT if prod_iv.effect == LoopEffect.ABSENT else LoopEffect.INVARIANT
            tensor_ivs.append(
                IterationVariable(
                    dimension=prod_iv_dim,
                    size=prod_iv.size,
                    type=prod_iv.type,
                    effect=tensor_effect,
                )
            )
        tensor_ssis = SteadyStateIterationSpace(variables=tuple(tensor_ivs))
        return tensor_ssis

    def update_mapping_for_transfer(self, node: TransferNode, src: HasOutputs, dsts: tuple[HasInputs, ...]) -> None:
        possible_dst_allocs = self.determine_possible_memory_allocations(node, src, dsts)
        possible_inter_core_tiling = self.determine_possible_inter_core_tiling(node, possible_dst_allocs, dsts)
        possible_allocations = self.determine_possible_transfer_plans(src, possible_dst_allocs)
        self.mapping.set_for_node(
            node,
            resource_allocation=possible_allocations,
            inter_core_tiling=possible_inter_core_tiling,
            memory_allocation=possible_dst_allocs,
        )

    def determine_possible_memory_allocations(
        self, node: TransferNode, src: HasOutputs, dsts: tuple[HasInputs, ...]
    ) -> tuple[tuple[Core, ...], ...]:
        """
        Determine the memory allocation of the transfer node.
        The memory alloc is always for the destination side tensors. For input transfers, the
        MEM_TO_MEM output tensor is allocated on memory cores; for output transfers the
        COMPUTE_TO_MEM output tensor is. Otherwise, allocations follow destination nodes.
        """
        if node.transfer_type in (TransferType.MEM_TO_MEM,) and isinstance(src, InEdge):
            # Find the dst with max number of compute allocations to determine possible memory cores
            compute_dsts = get_compute_predecessors_successors(
                tr=node, workload=self.ssw
            )  # won't have any compute preds
            dst = get_node_with_largest_resource_allocation(compute_dsts, self.mapping)
            possible_memory_cores = self._get_possible_memory_core_allocations(dst, node)
        elif node.transfer_type in (TransferType.MEM_TO_MEM,) and any(isinstance(dst, OutEdge) for dst in dsts):
            assert len(dsts) == 1, "Currently only support single destination for constant output transfer."
            dst = dsts[0]
            possible_memory_cores = self._retrieve_core_allocation(dst)
        elif node.transfer_type in (TransferType.COMPUTE_TO_MEM,):
            possible_memory_cores = self._get_possible_memory_core_allocations(src, node)
            if not possible_memory_cores:
                # No on-chip memory tiles — fall back to offchip core as the destination
                offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
                possible_memory_cores = ((offchip_core,),)
        else:
            # The destination order is the order the consumers declared, because that is what
            # pairs a spatial index with a core. Sorting here would silently hand each index a
            # different core than the computation node it feeds whenever a layer's cores are
            # not listed in ascending id order.
            destinations: dict[Core, None] = {}
            for dst in dsts:
                assert len(self._retrieve_core_allocation(dst)) == 1, "TODO: Support multiple compute allocations."
                destinations.update(dict.fromkeys(self._retrieve_core_allocation(dst)[0]))
            possible_memory_cores = (tuple(destinations),)
        return possible_memory_cores

    def determine_possible_inter_core_tiling(
        self, node: TransferNode, possible_dst_allocs: tuple[tuple[Core, ...], ...], dsts: tuple[HasInputs, ...]
    ) -> tuple[InterCoreTiling, ...]:
        possible_inter_core_tiling = []
        node_dims = set(self.ssw.get_dims(node))
        # The first hop of a transfer chain lands on a memory tile, so its own destinations
        # are transfers; the consumers that decide the split are the compute nodes behind it.
        compute_dsts = [dst for dst in dsts if isinstance(dst, ComputationNode)] or [
            dst
            for dst in get_compute_predecessors_successors(tr=node, workload=self.ssw)
            if isinstance(dst, ComputationNode)
        ]
        # A tensor whose dimensions the consumers do not split is held whole by every tile
        # it sits on, so its transfer carries no tiling however many copies exist.
        # Only the hop that lands on the memory tiles carries copies; the hop from them to
        # the cores still describes how those cores split the work.
        replicated = (
            node.transfer_type is TransferType.MEM_TO_MEM
            and bool(compute_dsts)
            and not any(
                dim in node_dims
                for dst in compute_dsts
                for dim, _ in self.ssw.get_unique_dims_inter_core_tiling(dst, self.mapping)
            )
        )
        for dst_allocs in possible_dst_allocs:
            nb_cores = len(dst_allocs)
            if nb_cores == 1 or replicated:
                dst_tiling = tuple()
            elif all(isinstance(dst, ComputationNode) for dst in dsts):
                # For fan-out: use the destination with the largest resource allocation
                # as the tiling reference (consistent with determine_possible_memory_allocations strategy)
                dst = get_node_with_largest_resource_allocation(dsts, self.mapping)
                dst_tiling = tuple(self.ssw.get_unique_dims_inter_core_tiling(dst, self.mapping))
            else:
                dst_tiling = self.get_inter_core_tiling_for_transfer(node, dst_allocs)
            possible_inter_core_tiling.append(dst_tiling)
        return tuple(possible_inter_core_tiling)

    def get_inter_core_tiling_for_transfer(
        self, node: TransferNode, memory_allocs: tuple[tuple[Core, ...], ...]
    ) -> tuple[InterCoreTiling, ...]:
        assert isinstance(node, TransferNode), "Node must be a TransferNode for inter-core tiling determination."
        assert node.transfer_type in (TransferType.COMPUTE_TO_MEM, TransferType.MEM_TO_MEM), (
            "This function should only be called for MEM_TO_MEM (input) or COMPUTE_TO_MEM (output) transfers."
        )
        # Get the compute preds and succs
        compute_preds_succs = get_compute_predecessors_successors(tr=node, workload=self.ssw)
        # Get the largest allocation one of these
        largest_alloc_node = get_node_with_largest_resource_allocation(compute_preds_succs, self.mapping)
        # Get its compute tiling and find the tiling loop that matches the number of memory allocs
        largest_alloc_tiling = self.ssw.get_unique_dims_inter_core_tiling(largest_alloc_node, self.mapping)
        mem_tiling = self.get_matching_tiling(largest_alloc_tiling, memory_allocs)
        return (mem_tiling,)

    def get_matching_tiling(
        self, compute_tiling: InterCoreTiling, dst_allocs: tuple[Core, ...]
    ) -> tuple[LayerDim, int]:
        # TODO: Make sure that the selected tiling_loop is relevant for the transfer node
        nb_allocs = len(dst_allocs)
        for tiling_loop in compute_tiling:
            _, size = tiling_loop
            if size == nb_allocs:
                return tiling_loop
        # No size with exact match found, try to find one that is a multiple of the number of dst allocs
        for tiling_loop in compute_tiling:
            dim, size = tiling_loop
            if size % nb_allocs == 0:
                return (dim, nb_allocs)
        # No clean divisor: split the best loop into the largest even share that fits, instead of raising.
        best = max(compute_tiling, key=lambda loop: largest_divisor_leq(loop[1], nb_allocs), default=None)
        if best is None:
            raise ValueError(f"No tiling loop to reconcile with dst allocs {dst_allocs}")
        dim, size = best
        return (dim, largest_divisor_leq(size, nb_allocs))

    def determine_possible_transfer_plans(
        self, src: HasOutputs, possible_dst_allocs: tuple[tuple[Core, ...], ...]
    ) -> tuple[MulticastPathPlan, ...]:
        all_possible_resource_plans = []
        possible_src_allocs = self._retrieve_core_allocation(src)
        for src_allocs in possible_src_allocs:
            for dst_allocs in possible_dst_allocs:
                possible_resource_plans = self.accelerator.communication_manager.get_possible_transfer_plan(
                    src_allocs=src_allocs,
                    dst_allocs=dst_allocs,
                )
                all_possible_resource_plans.extend(possible_resource_plans)
        return tuple(all_possible_resource_plans)

    def calculate_iterations(self) -> int:
        """Calculate the amount of steady state iterations based on all nodes' SSIS."""
        iterations_per_node = {node: prod(ssis.get_temporal_sizes()) for node, ssis in self.ssis.items()}
        # For now, return the minimum number of iterations across all nodes
        return min(iterations_per_node.values())

    def calculate_multiplicities(self) -> dict[ComputationNode, int]:
        """Calculate the multiplicity of each computation node in the steady state workload."""
        multiplicities = {}
        for node, ssis in self.ssis.items():
            total_iterations = prod(ssis.get_temporal_sizes())
            multiplicities[node] = total_iterations // self.iterations
        return multiplicities

    def _retrieve_core_allocation(self, node: Node) -> tuple[tuple[Core, ...], ...]:
        if isinstance(node, InEdge):
            return ((self.accelerator.get_core(self.accelerator.offchip_core_id),),)
        if isinstance(node, OutEdge):
            return ((self.accelerator.get_core(self.accelerator.offchip_core_id),),)
        if isinstance(node, HasOutputs):
            if isinstance(node, TransferNode):
                return self.mapping.get(node).memory_allocation
            return self.mapping.get(node).resource_allocation
        raise ValueError(f"Unexpected source node type: {type(node)}")

    def determine_transfer_type(
        self, src: HasOutputs, dsts: tuple[HasInputs, ...], src_type: str | None = None, dst_type: str | None = None
    ) -> TransferType:  # noqa: PLR0912
        """Determine the type of transfer needed based on the allocation types of src and dst nodes."""
        if src_type is None:
            src_allocation = self._retrieve_core_allocation(src)
            assert len(src_allocation) == 1, "TODO: Handle multiple source allocations for transfer type determination."
            src_type = self._effective_allocation_type(src_allocation[0])
        if dst_type is None:
            dst_allocations = [self._retrieve_core_allocation(dst)[0] for dst in dsts]
            # A constant input/output may be spread across a mix of allocations -- a compute core that
            # produces/consumes it and a memory/offchip core that stages it. Resolve to the type the
            # transfer must actually serve (compute when any allocation is compute; the PE is where the
            # data originates or is needed) instead of asserting the allocations are homogeneous.
            dst_type = self._effective_allocation_type([alloc for dst_alloc in dst_allocations for alloc in dst_alloc])
        if src_type == "compute" and dst_type == "compute":
            return TransferType.COMPUTE_TO_COMPUTE
        elif src_type == "compute" and dst_type in ("memory", "shim", "offchip"):
            return TransferType.COMPUTE_TO_MEM
        elif src_type in ("memory", "shim", "offchip") and dst_type == "compute":
            return TransferType.MEM_TO_COMPUTE
        elif src_type in ("memory", "shim", "offchip") and dst_type in ("memory", "shim", "offchip"):
            return TransferType.MEM_TO_MEM
        raise ValueError(f"Unsupported transfer type from {src_type} to {dst_type}")

    def _effective_allocation_type(self, allocs: list[Core]) -> str:
        """The transfer type a set of (possibly mixed) allocations must serve.

        Homogeneous allocations return their single type. Mixed allocations -- a constant that is on a
        compute core *and* staged in memory/offchip -- resolve to ``compute`` when any allocation is
        compute (that is where the data originates or is needed), else to the memory-family type. This
        keeps the transfer graph well-defined for the valid allocations the MILP can produce, instead
        of asserting homogeneity.
        """
        alloc_types = set(alloc.type for alloc in allocs)
        if not alloc_types:
            raise ValueError("no allocations to determine transfer type")
        if len(alloc_types) == 1:
            return alloc_types.pop()
        if "compute" in alloc_types:
            return "compute"
        for preferred in ("offchip", "shim", "memory"):
            if preferred in alloc_types:
                return preferred
        return alloc_types.pop()

    def _get_accelerator_memory_cores(self) -> set[Core]:
        """
        Get all memory cores in the accelerator.
        """
        memory_cores = set()
        for core in self.accelerator.core_list:
            if (
                core.type == "memory"
                and not core.id == self.accelerator.offchip_core_id
                and core.col_id < self.nb_cols_to_use
            ):
                memory_cores.add(core)
        return memory_cores

    def _get_possible_memory_core_allocations(self, src: HasOutputs, node: Node) -> tuple[tuple[Core, ...], ...]:
        # An input transfer feeds every unrolled destination separately; an output transfer is
        # gathered per column, since a column's compute cores share its memory tile.
        MAX_RELEVANT_FACTOR_PER_TRANSFER_TYPE = {
            TransferType.MEM_TO_MEM: 1,  # for input transfers to mem tile
            TransferType.COMPUTE_TO_MEM: 4,  # for output transfers to mem tile
        }
        # Check the dims of node and find their unrolling factors in inter_core_tiling of src
        node_dims = self.ssw.get_dims(node)
        inter_core_tiling_entries = self.mapping.get(src).inter_core_tiling
        if not inter_core_tiling_entries:
            inter_core_tiling_src = ()
        else:
            inter_core_tiling_src = inter_core_tiling_entries[0]
        total_relevant_unrolling = 1
        for dim in node_dims:
            for tiling_dim, size in inter_core_tiling_src:
                if tiling_dim == dim:
                    total_relevant_unrolling *= size
        columns = self._columns_of(src)
        if columns and node.transfer_type in (TransferType.COMPUTE_TO_MEM, TransferType.MEM_TO_MEM):
            # A column's cores share its one memory tile, so both need one tile per occupied column, not per core.
            required_nb_memory_cores = min(columns, total_relevant_unrolling)
        else:
            required_nb_memory_cores = ceil(
                total_relevant_unrolling / MAX_RELEVANT_FACTOR_PER_TRANSFER_TYPE[node.transfer_type]
            )
        # Snap the count down to a divisor of the compute split so the transfer is one even inter-core tiling.
        if total_relevant_unrolling > 1:
            required_nb_memory_cores = largest_divisor_leq(total_relevant_unrolling, required_nb_memory_cores)
        # Unrolling binds allocation position to spatial index, so column order keeps a tile with its own cores.
        all_mem_cores = sorted(self._get_accelerator_memory_cores(), key=lambda core: (core.col_id, core.id))
        candidates = [tuple(combo) for combo in combinations(all_mem_cores, required_nb_memory_cores)]
        # A tensor its consumers do not split is held whole, so a single tile can be left
        # feeding every column. One copy per occupied column is the alternative: each tile
        # serves its own column, and the shim broadcasts the tensor to them.
        if total_relevant_unrolling == 1 and node.transfer_type is TransferType.MEM_TO_MEM:
            per_column = tuple(c for c in all_mem_cores if c.col_id in self._column_ids_of(src))
            if len(per_column) > 1 and per_column not in candidates:
                candidates.append(per_column)
        return tuple(candidates)

    def _column_ids_of(self, src: HasOutputs) -> set[int]:
        """The array columns the source's cores sit in, empty if they carry no coordinates."""
        allocations = self.mapping.get(src).resource_allocation
        cores = allocations[0] if allocations else ()
        return {core.col_id for core in cores if getattr(core, "col_id", None) is not None}

    def _columns_of(self, src: HasOutputs) -> int:
        """How many array columns the source's cores sit in, 0 if unknown.

        Accelerator descriptions need not give their cores coordinates; without them the
        caller falls back to the rows-per-column estimate.
        """
        return len(self._column_ids_of(src))
