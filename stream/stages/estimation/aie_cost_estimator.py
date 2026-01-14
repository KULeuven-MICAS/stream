from dataclasses import dataclass
from math import ceil, floor, prod

from xdsl.dialects.builtin import BFloat16Type, FixedBitwidthType

from stream.cost_model.core_cost import CoreCostEntry
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.workload.workload import ComputationNode, Workload


@dataclass
class AIECostEstimator:
    """Simple utilization-based estimator for AIE compute cores."""

    workload: Workload
    mapping: Mapping

    def estimate(self, node: ComputationNode, core: Core) -> CoreCostEntry:
        dim_sizes = [self.workload.get_dimension_size(dim) for dim in self.workload.get_dims(node)]
        macs = prod(dim_sizes)
        kernel = self.mapping.get(node).kernel
        assert kernel is not None, "Kernel must be defined in mapping for AIE cost estimation."
        utilization = kernel.utilization
        ideal_ops_per_cycle = self.ops_per_cycle(node, core)
        ideal_cycles = ceil(macs / ideal_ops_per_cycle)
        ops_per_cycle = floor(ideal_ops_per_cycle * (utilization / 100.0))
        cycles = ceil(macs / ops_per_cycle)
        energy = 0  # TODO
        return CoreCostEntry(
            energy_total=energy,
            latency_total=cycles,
            ideal_cycle=ideal_cycles,
            ideal_temporal_cycle=ideal_cycles,
            mem_energy_breakdown={},
            cme=None,
            mapping=None,
            layer=node,
            metadata={"utilization": utilization},
        )

    def ops_per_cycle(self, node: ComputationNode, core: Core) -> int:
        """Depending on the node inputs and output data type and core type,
        return the number of operations per cycle."""
        inputs_datatype = [inp.operand_type for inp in node.inputs]
        assert all(dt == inputs_datatype[0] for dt in inputs_datatype), "All input datatypes must be the same."
        input_datatype = inputs_datatype[0]
        output_datatype = node.outputs[0].operand_type
        return self.ops_per_cycle_for_datatypes(input_datatype, output_datatype, core)

    def ops_per_cycle_for_datatypes(
        self,
        input_datatype: FixedBitwidthType,
        output_datatype: FixedBitwidthType,
        core: Core,
    ) -> int:
        if isinstance(input_datatype, BFloat16Type) and isinstance(output_datatype, BFloat16Type):
            if core.core_type == "aie2.compute":
                return 32
            elif core.core_type == "aie.compute":
                return 16
            else:
                raise self.raise_not_implemented_for_datatypes(input_datatype, output_datatype, core)
        else:
            raise self.raise_not_implemented_for_datatypes(input_datatype, output_datatype, core)

    def raise_not_implemented_for_datatypes(self, input_datatype, output_datatype, core) -> NotImplementedError:
        return NotImplementedError(
            f"Ops per cycle not implemented for input datatype {input_datatype} "
            f"and output datatype {output_datatype} on core type {core.core_type}."
        )
