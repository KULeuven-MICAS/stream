import logging
import os
from math import ceil
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import Constants, MemoryOperand
from zigzag.mapping.temporal_mapping import TemporalMappingType
from zigzag.stages.stage import Stage as ZigZagStage
from zigzag.utils import pickle_deepcopy

from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.estimation.aie_cost_estimator import AIECostEstimator
from stream.stages.estimation.zigzag_cost_estimator import ZigZagCostEstimator
from stream.stages.stage import Stage, StageCallable
from stream.visualization.cost_model_evaluation_lut import (
    visualize_cost_lut_pickle,
)
from stream.workload.workload import ComputationNode, Workload

logger = logging.getLogger(__name__)


class _KwargsMainStage:
    def __init__(self, list_of_callables, **kwargs: Any):
        self.kwargs = kwargs
        self.list_of_callables = list_of_callables

    def run(self):
        answers = []
        for cme, extra_info in self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs).run():
            answers.append((cme, extra_info))
        return answers


class CoreCostEstimationStage(Stage):
    """
    Stage that computes and caches core cost entries for each valid node-core allocation.
    """

    REQUIRED_FIELDS = (
        "workload",
        "accelerator",
        "mapping",
        "loma_lpf_limit",
        "output_path",
        "temporal_mapping_type",
    )

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        ctx: StageContext,
    ):
        """
        Initialize the stage by:
        - extracting all the unique nodes that will have to be evaluated
        - initializing the valid node-core allocations (which are used later by the InterCoreMappingStage)
        """
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.accelerator: Accelerator = self.ctx.get("accelerator")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.loma_lpf_limit = self.ctx.get("loma_lpf_limit")
        self.output_path = self.ctx.get("output_path")
        self.temporal_mapping_type: TemporalMappingType = self.ctx.get("temporal_mapping_type")
        self.loma_show_progress_bar: bool = self.ctx.get("loma_show_progress_bar", False)
        self.cost_lut_path: str = os.path.join(self.output_path, "core_cost_lut.pickle")
        self.visualize_cost_lut_path: str = os.path.splitext(self.cost_lut_path)[0] + ".png"

        self.valid_allocations: dict[ComputationNode, list[Core]] = {
            node: self.mapping.get(node).core_allocation for node in self.workload.get_computation_nodes()
        }
        self.cost_lut: CoreCostLUT = CoreCostLUT(self.cost_lut_path)

    def run(self):
        logger.info("Start CoreCostEstimationStage.")
        self.update_cost_lut()
        # self.visualize_cost_lut()
        logger.info("Finished CoreCostEstimationStage.")

        self.ctx.set(workload=self.workload, accelerator=self.accelerator, cost_lut=self.cost_lut)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def update_cost_lut(self):
        for node in self.workload.get_computation_nodes():
            seen_new = False
            cores = self.valid_allocations[node]
            for core in cores:
                if self.cost_lut.has_cost(node, core):
                    continue
                equal_node = self.cost_lut.get_equal_node(node)
                equal_core = self.cost_lut.get_equal_core(equal_node, core) if equal_node else None
                if equal_node and equal_core:
                    cost = pickle_deepcopy(self.cost_lut.get_cost(equal_node, equal_core))
                    self.cost_lut.add_cost(node, core, cost, allow_overwrite=False)
                    continue
                estimator = self.get_estimator(core)
                cost_entry = estimator.estimate(node, core)
                self.cost_lut.add_cost(node, core, cost_entry, allow_overwrite=False)
                seen_new = True
            if seen_new:
                self.cost_lut.save()

    def get_estimator(self, core: Core):
        if self.is_aie_compute_core(core):
            return AIECostEstimator(self.workload, self.mapping)
        return ZigZagCostEstimator(
            cost_lut=self.cost_lut,
        )

    def is_aie_compute_core(self, core: Core) -> bool:
        return str(core.core_type).startswith("aie2.") and core.type == "compute"

    def visualize_cost_lut(self):
        scale_factors = {
            n: len([cn for cn in self.workload.node_list if cn.has_same_performance(n)])
            for n in self.cost_lut.get_nodes()
        }
        visualize_cost_lut_pickle(self.cost_lut, scale_factors, self.visualize_cost_lut_path)


class MinimalBandwidthLatencyStage(ZigZagStage):
    """Class that keeps yields only the cost model evaluation that has minimal objective function of all cost model
    evaluations generated by it's substages created by list_of_callables.
    The objective function is defined as:
        `ceil(nb_parallel_nodes * required_dram_bandwidth / total_dram_bandwidth) * latency`
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        reduce_minimal_keep_others: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the compare stage.
        """
        super().__init__(list_of_callables, **kwargs)
        self.ctx = StageContext.from_kwargs(**kwargs)
        self.keep_others = reduce_minimal_keep_others
        accelerator: Core = self.ctx.require_value("accelerator", self.__class__.__name__)
        self.nb_parallel_nodes: int = self.ctx.get("nb_parallel_nodes", 1)
        self.has_dram_level: bool = self.ctx.get("has_dram_level", False)

        self.mem_ops_with_dram: list[MemoryOperand] = []
        self.mem_ops = list(accelerator.memory_hierarchy.operands)
        self.total_dram_bandwidth: int | None = None

        if self.has_dram_level:
            nb_levels_per_op = [len(accelerator.memory_hierarchy.get_memory_levels(op)) for op in self.mem_ops]
            dram_mem_level = max(nb_levels_per_op)  # start at 1
            self.mem_ops_with_dram = [
                op for op, nb_levels in zip(self.mem_ops, nb_levels_per_op, strict=False) if nb_levels == dram_mem_level
            ]
            ports = accelerator.get_top_memory_instance(self.mem_ops_with_dram[0]).ports
            self.total_dram_bandwidth = max((port.bw_max for port in ports), default=0)

    def get_used_dram_bandwidth_for_op(self, cme: CostModelEvaluation, mem_op: MemoryOperand):
        if mem_op not in self.mem_ops_with_dram:
            return 0
        bw_per_direction = 100  # TODO: maybe this is wrong
        total_bw = sum(bw_per_direction.data.values())
        return total_bw

    def objective_function(self, cme: CostModelEvaluation) -> float:
        """
        # TODO this does not cover all cases
        """
        latency: int = int(cme.latency_total2)

        if not self.has_dram_level:
            return latency

        assert self.total_dram_bandwidth is not None

        match len(self.mem_ops_with_dram):
            case 1:
                total_used_dram_bw = self.nb_parallel_nodes * self.get_used_dram_bandwidth_for_op(
                    cme, self.mem_ops_with_dram[0]
                )
            case 2:
                # Assume that 1 operand is broadcasted to all cores and only needs 1 simultaneous transfer for all cores
                # We don't know which operand is broadcasted, so just pick one that is not the output
                broadcast_op = next(op for op in self.mem_ops_with_dram if op != Constants.OUTPUT_MEM_OP)
                other_op = next(op for op in self.mem_ops_with_dram if op != broadcast_op)
                bw_for_broadcasting = 1 * self.get_used_dram_bandwidth_for_op(cme, broadcast_op)
                bw_for_blocking = self.nb_parallel_nodes * self.get_used_dram_bandwidth_for_op(cme, other_op)
                total_used_dram_bw = bw_for_blocking + bw_for_broadcasting
            case 3:
                # We don't know broadcast op, just pick one that is not the output
                broadcast_op = next(op for op in self.mem_ops_with_dram if op != Constants.OUTPUT_MEM_OP)
                other_ops = [op for op in self.mem_ops_with_dram if op != broadcast_op]

                bw_for_broadcasting = 1 * self.get_used_dram_bandwidth_for_op(cme, broadcast_op)
                bw_for_blocking = self.nb_parallel_nodes * sum(
                    self.get_used_dram_bandwidth_for_op(cme, mem_op) for mem_op in other_ops
                )
                total_used_dram_bw = bw_for_blocking + bw_for_broadcasting
            case _:
                raise NotImplementedError

        return ceil(total_used_dram_bw / self.total_dram_bandwidth) * latency

    def run(self):
        """! Run the compare stage by comparing a new cost model output with the current best found result."""
        sub_list_of_callables = self.list_of_callables[1:]
        substage: ZigZagStage = self.list_of_callables[0](sub_list_of_callables, **self.ctx.data)

        other_cmes: list[tuple[CostModelEvaluation, Any]] = []
        best_cme: CostModelEvaluation | None = None
        for cme, extra_info in substage.run():
            assert isinstance(cme, CostModelEvaluation)
            if best_cme is None or self.objective_function(cme) < self.objective_function(best_cme):
                best_cme = cme
            if self.keep_others:
                other_cmes.append((cme, extra_info))

        assert best_cme is not None
        yield best_cme, other_cmes
