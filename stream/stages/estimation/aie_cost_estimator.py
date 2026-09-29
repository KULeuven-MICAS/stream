from dataclasses import dataclass, field
from math import ceil, prod

from xdsl.dialects.builtin import BFloat16Type, FixedBitwidthType, Float32Type

from stream.cost_model.core_cost import CoreCostEntry
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.mapping.work_share import computed_fraction, core_work_share, split_steps
from stream.workload.workload import ComputationNode, Workload


@dataclass
class AIECostEstimator:
    """Estimator for AIE compute cores, priced from the kernel library."""

    workload: Workload
    mapping: Mapping
    fusion_splits: dict = field(default_factory=dict)

    def estimate(self, node: ComputationNode, core: Core) -> CoreCostEntry:
        dim_sizes = [self.workload.get_dimension_size(dim) for dim in self.workload.get_dims(node)]
        macs = round(prod(dim_sizes) * core_work_share(self.workload, self.mapping, node, core, self._steps(node)))
        kernel = self.mapping.get(node).kernel
        ideal_ops_per_cycle = self.ops_per_cycle(node, core)
        ideal_cycles = ceil(macs / ideal_ops_per_cycle)
        cycles, metadata = self._kernel_cycles(kernel, macs)
        if cycles is None:
            cycles, metadata = ideal_cycles, {"backend": "aie"}
        metadata["computed_fraction"] = computed_fraction(self.workload, self.mapping, node, core, self._steps(node))
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
            metadata=metadata,
        )

    @staticmethod
    def _kernel_cycles(kernel, macs: int) -> tuple[int | None, dict]:
        """Cycles for ``macs`` operations of this kernel, from its measured call or its family rate."""
        if kernel is None or kernel.library is None or kernel.library.spec(kernel.library_key) is None:
            return None, {}
        spec = kernel.spec
        if measured := spec.call_cycles(kernel.call_shape()):
            call_cycles, call_ops = measured
            return ceil(call_cycles * macs / call_ops), {"backend": "aie", "measured_symbol": spec.symbol}
        rate = kernel.library.families[spec.family].ops_per_cycle
        return ceil(macs / rate), {"backend": "aie", "family": spec.family}

    def _steps(self, node: ComputationNode) -> int:
        return split_steps(self.workload, self.mapping, node, self.fusion_splits)

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
        elif isinstance(input_datatype, Float32Type) and isinstance(output_datatype, Float32Type):
            if core.core_type == "aie2.compute":
                return 16
            elif core.core_type == "aie.compute":
                return 8
            else:
                raise self.raise_not_implemented_for_datatypes(input_datatype, output_datatype, core)
        else:
            raise self.raise_not_implemented_for_datatypes(input_datatype, output_datatype, core)

    def raise_not_implemented_for_datatypes(self, input_datatype, output_datatype, core) -> NotImplementedError:
        return NotImplementedError(
            f"Ops per cycle not implemented for input datatype {input_datatype} "
            f"and output datatype {output_datatype} on core type {core.core_type}."
        )
