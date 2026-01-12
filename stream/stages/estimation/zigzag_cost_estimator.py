from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.utils import pickle_deepcopy

from stream.cost_model.core_cost import CoreCostEntry
from stream.hardware.architecture.core import Core
from stream.workload.workload import ComputationNode


@dataclass
class ZigZagCostEstimator:
    cost_lut: Any

    def estimate(self, node: ComputationNode, core: Core) -> CoreCostEntry:
        raise NotImplementedError("Not yet updated for new workload representation.")
        node_duplicate = pickle_deepcopy(node)
        self.cost_lut.remove_cores_with_same_id(node, core)
        too_large_operands_for_cme = self.check_core_capacity_for_node(core, node_duplicate)
        node_duplicate.set_chosen_core_allocation(core)
        if core.dataflows:
            node_duplicate.spatial_mapping = core.dataflows
        cme = self.run_zigzag(node_duplicate, too_large_operands_for_cme, core)
        cme = self.increase_cc_per_op(cme, node.type)
        node_duplicate.set_chosen_core_allocation(None)
        return CoreCostEntry(
            energy_total=getattr(cme, "energy_total", 0),
            latency_total=getattr(cme, "latency_total2", getattr(cme, "ideal_cycle", 0)),
            ideal_cycle=getattr(cme, "ideal_cycle", 0),
            ideal_temporal_cycle=getattr(cme, "ideal_temporal_cycle", 0),
            mem_energy_breakdown=getattr(cme, "mem_energy_breakdown", {}),
            cme=cme,
            mapping=getattr(cme, "mapping", None),
            layer=node,
        )

    def run_zigzag(
        self, node: ComputationNode, too_large_operands: list[MemoryOperand], core_id: int
    ) -> CostModelEvaluation:
        """Run the ZigZag flow to estimate performance of a given node on a core."""

        main_stage = self.instantiate_zigzag_flow(node, too_large_operands, core_id)
        logger.info(f"Launching intra-core mapping optimization for {node} -> core {core_id} ...")
        answers = main_stage.run()
        assert len(answers) == 1, "CoreCostEstimationStage's subflow returned more than one cost entry"
        cme: CostModelEvaluation = answers[0][0]  # type: ignore
        return cme

    def instantiate_zigzag_flow(self, node: ComputationNode, too_large_operands: list[MemoryOperand], core_id: int):
        """Instantiate a runnable ZigZag mainstage"""
        core = self.accelerator.get_core(core_id)
        nb_parallel_nodes: int = (
            1 if contains_wildcard(node.inter_core_tiling) else prod(size for _, size in node.inter_core_tiling)
        )  # type: ignore

        if too_large_operands:
            core = self.add_offchip_to_core(core, too_large_operands, node.id)

        main_stage = _KwargsMainStage(
            [  # Initializes the MainStage as entry point
                MinimalBandwidthLatencyStage,  # type: ignore
                SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
                MinimalBandwidthLatencyStage,  # Reduces all CMEs, returning minimal EDP one
                TemporalMappingGeneratorStage,  # Generates multiple temporal mappings (TM)
                CostModelStage,  # Evaluates generated SM and TM through cost model
            ],
            layer=node,
            accelerator=core,  # Accelerator in zigzag corresponds to Core in stream
            loma_lpf_limit=self.loma_lpf_limit,  # required by LomaEngine
            loma_show_progress_bar=self.loma_show_progress_bar,
            temporal_mapping_type=self.temporal_mapping_type,
            nb_parallel_nodes=nb_parallel_nodes,
            has_dram_level=(len(too_large_operands) > 0),
        )
        return main_stage

    def get_cc_per_op(self, op_type: str):
        """Return the number of cycles that the operational units need to finish the given operation."""
        match op_type:
            case "silu":
                return 4
            case "sigmoid":
                return 4
            case "exp":
                return 4
            case _:
                return 1

    def increase_cc_per_op(self, cme: CostModelEvaluation, op_type: str):
        """Given a ZigZag that assumes each operation takes one cycle, generate a new one that takes into account that
        the operation might take more than one cycle."""
        cc_per_op = self.get_cc_per_op(op_type)
        if cc_per_op > 1:
            logger.warning(f"Setting cycles per mac of {op_type} node to {cc_per_op}")

        new_cme = CostModelEvaluation(
            accelerator=cme.accelerator,
            layer=cme.layer,
            spatial_mapping=cme.spatial_mapping,
            spatial_mapping_int=cme.spatial_mapping_int,
            temporal_mapping=cme.temporal_mapping,
            access_same_data_considered_as_no_access=cme.access_same_data_considered_as_no_access,
            cycles_per_op=cc_per_op,
        )

        return new_cme
