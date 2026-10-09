from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from math import ceil

import zigzag.mapping.spatial_mapping as zigzag_spatial_mapping
import zigzag.mapping.temporal_mapping as zigzag_temporal_mapping
import zigzag.workload.layer_attributes as zigzag_layer_attributes
import zigzag.workload.layer_node as zigzag_layer_node
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.datatypes import LayerDim as ZigZagLayerDim
from zigzag.datatypes import LayerOperand as ZigZagLayerOperand
from zigzag.stages.evaluation.cost_model_evaluation import CostModelStage
from zigzag.stages.main import MainStage as _KwargsMainStage
from zigzag.stages.mapping.spatial_mapping_generation import SpatialMappingGeneratorStage
from zigzag.stages.mapping.temporal_mapping_generator_stage import TemporalMappingGeneratorStage
from zigzag.stages.results.reduce_stages import MinimalLatencyStage

from stream.cost_model.core_cost import IDEAL_CYCLE_BACKEND, CoreCostEntry
from stream.datatypes import ELEMENT_BITS, LayerDim
from stream.hardware.architecture.accelerator import Accelerator
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.workload.utils import affine_bounds, is_mac_operator_type
from stream.workload.workload import ComputationNode, Tensor, Workload

ZigZagLayerNode = zigzag_layer_node.LayerNode
ZigZagLayerNodeAttributes = zigzag_layer_node.LayerNodeAttributes
ZigZagMappingAttributes = zigzag_layer_node.MappingAttributes
ZigZagSpatialMapping = zigzag_spatial_mapping.SpatialMapping
ZigZagSpatialMappingHint = zigzag_spatial_mapping.SpatialMappingHint
ZigZagTemporalMappingType = zigzag_temporal_mapping.TemporalMappingType

ZigZagLayerDimSizes = zigzag_layer_attributes.LayerDimSizes
ZigZagLayerEquation = zigzag_layer_attributes.LayerEquation
ZigZagLayerDimRelation = zigzag_layer_attributes.LayerDimRelation
ZigZagLayerOperandPrecision = zigzag_layer_attributes.LayerOperandPrecision
ZigZagInputOperandSource = zigzag_layer_attributes.InputOperandSource
ZigZagLayerPadding = zigzag_layer_attributes.LayerPadding
ZigZagLayerTemporalOrdering = zigzag_layer_attributes.LayerTemporalOrdering
ZigZagMemoryOperandLinks = zigzag_layer_attributes.MemoryOperandLinks

logger = logging.getLogger(__name__)


@dataclass
class ZigZagCostEstimator:
    workload: Workload
    accelerator: Accelerator
    mapping: Mapping
    nb_spatial_mappings_generated: int = 1
    temporal_mapping_type: ZigZagTemporalMappingType = ZigZagTemporalMappingType.EVEN
    loma_lpf_limit: int = 8

    input_operand_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    loma_show_progress_bar = False
    supported_pr_length = 2

    def _affine_binary_op_expr_to_dims_and_coefficients(
        self, expr: AffineBinaryOpExpr
    ) -> tuple[list[ZigZagLayerDim], list[int]]:
        """Convert an AffineBinaryOpExpr of the form c1*D1 + c2*D2 + C into its ZigZagLayerDims and their coefficients;
        the constant, a padding, only shifts the interior tile ZigZag costs."""
        dims: list[ZigZagLayerDim] = []
        coefficients: list[int] = []

        def process_expr(e: AffineDimExpr | AffineBinaryOpExpr | AffineConstantExpr, coeff: int) -> None:
            if isinstance(e, AffineDimExpr):
                dim = ZigZagLayerDim(f"D{e.position}")
                dims.append(dim)
                coefficients.append(coeff)
            elif isinstance(e, AffineBinaryOpExpr):
                match e.kind:
                    case AffineBinaryOpKind.Add:
                        process_expr(e.lhs, coeff)
                        process_expr(e.rhs, coeff)
                    case AffineBinaryOpKind.Mul:
                        if isinstance(e.lhs, AffineDimExpr) and isinstance(e.rhs, AffineConstantExpr):
                            process_expr(e.lhs, coeff * e.rhs.value)
                        elif isinstance(e.rhs, AffineDimExpr) and isinstance(e.lhs, AffineConstantExpr):
                            process_expr(e.rhs, coeff * e.lhs.value)
                        else:
                            raise NotImplementedError(
                                "Multiplication between two non-constant expressions is not supported."
                            )
                    case _:
                        raise NotImplementedError(f"Unsupported operation {e.kind} in AffineBinaryOpExpr.")
            elif not isinstance(e, AffineConstantExpr):
                raise NotImplementedError(f"Unsupported expression type {type(e)}.")

        process_expr(expr, 1)
        return dims, coefficients

    def create_equation_and_dimension_relations_and_pr_sizes(
        self, node: ComputationNode
    ) -> tuple[ZigZagLayerEquation, list[ZigZagLayerDimRelation], ZigZagLayerDimSizes]:
        """Create the ZigZag equation, e.g. O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][ix][iy].
        If a node has a AffineBinaryOpExpr as one of its dims, it is replaced with a generic dim name,
        and the dimension_relations attribute is used to capture the relation."""
        base_dims = [ZigZagLayerDim(f"D{i}") for i in range(node.num_dims)]
        extra_dims: list[ZigZagLayerDim] = []
        dimension_relations: list[ZigZagLayerDimRelation] = []
        pr_sizes: dict[ZigZagLayerDim, int] = {}
        per_core_dim_sizes = self._per_core_sizes(node)
        tensors = (node.outputs[0],) + self._operands(node)
        operand_names = ["O"] + self.input_operand_names[: len(tensors) - 1]
        equation_str = ""
        for tensor, operand_name in zip(tensors, operand_names, strict=True):
            tensor_shape = self.workload.get_tensor_shape_with_dimension_sizes(tensor, per_core_dim_sizes, node)
            mapping = node.get_mapping(tensor)
            operand_dims: list[ZigZagLayerDim] = []
            for i, expr in enumerate(mapping.results):
                if isinstance(expr, AffineDimExpr):
                    dim = base_dims[expr.position]
                    operand_dims.append(dim)
                elif isinstance(expr, AffineBinaryOpExpr):
                    dims_in_expr, coefficients = self._affine_binary_op_expr_to_dims_and_coefficients(expr)
                    if len(dims_in_expr) == 1 and coefficients[0] == 1:
                        # Single-dimension self-offset (recurrence state read, e.g. h[t-1]). The
                        # cross-iteration carry is handled in scheduling, not costing, so treat it
                        # as a plain access to that dimension (drop the offset). No pr relation.
                        operand_dims.append(dims_in_expr[0])
                    else:
                        # Genuine projection-relevant (pr) expression, e.g. conv ix = s*ox + d*fx.
                        dim = ZigZagLayerDim(f"D{len(base_dims) + len(extra_dims)}")
                        extra_dims.append(dim)
                        operand_dims.append(dim)
                        assert len(dims_in_expr) == len(coefficients) == self.supported_pr_length, (
                            "Mismatch in dims and coefficients length."
                        )
                        dimension_relations.append(
                            ZigZagLayerDimRelation(
                                dim_1=dim,
                                coef_2=coefficients[0],
                                dim_2=dims_in_expr[0],
                                coef_3=coefficients[1],
                                dim_3=dims_in_expr[1],
                            )
                        )
                        # Set pr dim sizes
                        pr_sizes[dim] = tensor_shape[i]  # logical size of the tensor (without padding)
                else:
                    raise NotImplementedError(f"Unsupported affine expr type {type(expr)} in mapping.")
            # Create equation string part for this operand

            equation_str += operand_name + "[" + "][".join(dim.name.lower() for dim in operand_dims) + "]"
            if operand_name == "O":
                equation_str += " = "
            elif tensor != tensors[-1]:
                equation_str += " * "
        return ZigZagLayerEquation(equation_str), dimension_relations, ZigZagLayerDimSizes(pr_sizes)

    @staticmethod
    def _operands(node: ComputationNode) -> tuple[Tensor, ...]:
        """The inputs ZigZag prices: a multiply-accumulate's two operands, so not a further input such as a bias."""
        return node.inputs[:2] if is_mac_operator_type(node.type) else node.inputs

    def _per_core_sizes(self, node: ComputationNode) -> dict[LayerDim, int]:
        """Each unique dim's extent on one core of the node's inter-core split, a split that does not divide its dim
        evenly, or a node without a mapping, leaving it whole."""
        try:
            tiling = self.workload.get_unique_dims_inter_core_tiling(node, self.mapping)
        except Exception:  # noqa: BLE001
            tiling = ()
        factors: dict[LayerDim, int] = {}
        for dim, factor in tiling:
            factors[dim] = factors.get(dim, 1) * factor
        sizes = {dim: self.workload.get_dimension_size(dim) for dim in self.workload.unique_dimensions()[0]}
        return {dim: size // f if size % (f := factors.get(dim, 1)) == 0 else size for dim, size in sizes.items()}

    def create_layer_dim_sizes(self, node: ComputationNode) -> ZigZagLayerDimSizes:
        """Each node dim's extent on one core, a strided reader's producer dim ``2*z + r`` spanning its unique dims."""
        sizes = list(self._per_core_sizes(node).values())
        bounds = (affine_bounds(dim, sizes) for dim in self.workload.get_dims(node))
        return ZigZagLayerDimSizes({ZigZagLayerDim(f"D{i}"): high - low + 1 for i, (low, high) in enumerate(bounds)})

    def create_operand_precision(self, node: ComputationNode, core: Core | None = None) -> ZigZagLayerOperandPrecision:
        """Each operand at its tensor's element type; a matmul's or convolution's partial sums at the accumulator
        precision its core declares, cast to the output's type as they are written out."""
        precisions: dict[str, int] = {
            self.input_operand_names[i]: tensor.operand_type.bitwidth for i, tensor in enumerate(self._operands(node))
        }
        assert len(node.outputs) == 1, "Only single output nodes are supported."
        precisions["O_final"] = node.outputs[0].operand_type.bitwidth
        accumulator = (getattr(core, "operand_precision", None) or {}).get("accumulator")
        precisions["O"] = (
            ELEMENT_BITS[accumulator] if accumulator and is_mac_operator_type(node.type) else precisions["O_final"]
        )
        data: dict[ZigZagLayerOperand, int] = {
            ZigZagLayerOperand(operand_str): size for operand_str, size in precisions.items()
        }
        return ZigZagLayerOperandPrecision(data)

    def create_constant_operands(self, node: ComputationNode) -> list[ZigZagLayerOperand]:
        # Assume all operands constant for a single node workload

        constant_operands: list[ZigZagLayerOperand] = []
        for i in range(len(self._operands(node))):
            constant_operands.append(ZigZagLayerOperand(self.input_operand_names[i]))
        return constant_operands

    def create_operand_source(self, node: ComputationNode) -> ZigZagInputOperandSource:
        # For now, assume all input operands originate from the layer id itself
        operand_source: ZigZagInputOperandSource = {}
        for i in range(len(self._operands(node))):
            operand_source[ZigZagLayerOperand(self.input_operand_names[i])] = 0
        return operand_source

    def get_layer_node_attributes(self, node: ComputationNode, core: Core | None = None) -> ZigZagLayerNodeAttributes:
        layer_type: str = node.type
        equation, dimension_relations, pr_sizes = self.create_equation_and_dimension_relations_and_pr_sizes(node)
        layer_dim_sizes = self.create_layer_dim_sizes(node)
        operand_precision = self.create_operand_precision(node, core)
        constant_operands = self.create_constant_operands(node)
        input_operand_source = self.create_operand_source(node)
        return ZigZagLayerNodeAttributes(
            layer_type=layer_type,
            equation=equation,
            layer_dim_sizes=layer_dim_sizes,
            operand_precision=operand_precision,
            dimension_relations=dimension_relations,
            constant_operands=constant_operands,
            input_operand_source=input_operand_source,
            padding=ZigZagLayerPadding.empty(),
            pr_layer_dim_sizes=pr_sizes,
        )

    def get_memory_operand_links(self, node: ComputationNode, core: Core) -> ZigZagMemoryOperandLinks:
        # Check that the core memory hierarchy contains two input memory operands I1 and I2 and one output O
        memory_operands = list(core.mem_hierarchy_dict.keys())
        assert len(memory_operands) > len(self._operands(node))
        assert any(op.name == "I1" for op in memory_operands), (
            f"Core {core.id} memory hierarchy must contain memory operand I1."
        )
        assert any(op.name == "I2" for op in memory_operands), (
            f"Core {core.id} memory hierarchy must contain memory operand I2."
        )
        assert any(op.name == "O" for op in memory_operands), (
            f"Core {core.id} memory hierarchy must contain memory operand O."
        )
        memory_operand_links: ZigZagMemoryOperandLinks = {}
        for i in range(len(self._operands(node))):
            mem_op = next(op for op in memory_operands if op.name == f"I{i + 1}")
            layer_op = ZigZagLayerOperand(self.input_operand_names[i])
            memory_operand_links[layer_op] = mem_op
        output_mem_op = next(op for op in memory_operands if op.name == "O")
        memory_operand_links[ZigZagLayerOperand("O")] = output_mem_op
        return ZigZagMemoryOperandLinks(memory_operand_links)

    def get_mapping_attributes(self, node: ComputationNode, core: Core) -> ZigZagMappingAttributes:
        # A core is costed through the ZigZag backend even when it is not itself ZigZag-backed (e.g. an
        # AIE tile): such a core exposes no `dataflows`, so fall back to an empty spatial mapping.
        dataflows = getattr(core, "dataflows", None)
        spatial_mapping = dataflows if dataflows else ZigZagSpatialMapping.empty()
        spatial_mapping_hint = ZigZagSpatialMappingHint.empty()
        memory_operand_links = self.get_memory_operand_links(node, core)
        temporal_ordering = ZigZagLayerTemporalOrdering.empty()
        return ZigZagMappingAttributes(
            spatial_mapping=spatial_mapping,
            spatial_mapping_hint=spatial_mapping_hint,
            memory_operand_links=memory_operand_links,
            temporal_ordering=temporal_ordering,
        )

    def get_layer_node(self, node: ComputationNode, core: Core) -> ZigZagLayerNode:
        node_attr = self.get_layer_node_attributes(node, core)
        mapping_attr = self.get_mapping_attributes(node, core)
        return ZigZagLayerNode(
            layer_id=0,
            node_name=node.name,
            node_attr=node_attr,
            mapping_attr=mapping_attr,
        )

    def estimate(self, node: ComputationNode, core: Core) -> CoreCostEntry:
        try:
            layer_node = self.get_layer_node(node, core)
            cme = self.run_zigzag(layer_node, core)
            cme = self.increase_cc_per_op(cme, node.type)
            return CoreCostEntry(
                energy_total=getattr(cme, "energy_total", 0),
                latency_total=getattr(cme, "latency_total2", getattr(cme, "ideal_cycle", 0)),
                ideal_cycle=getattr(cme, "ideal_cycle", 0),
                ideal_temporal_cycle=getattr(cme, "ideal_temporal_cycle", 0),
                mem_energy_breakdown=getattr(cme, "mem_energy_breakdown", {}),
                cme=cme,
                mapping=getattr(cme, "mapping", None),
                layer=node,
                metadata={"backend": "zigzag"},
            )
        except Exception as exc:
            # Fallback: this core is not costable by ZigZag -- either it has no ZigZag backend (e.g. an
            # AIE tile, whose `dataflows`/`mem_hierarchy_dict` do not exist) or spatial-mapping
            # generation rejected the pair.
            logger.warning(
                "ZigZag estimation failed for %s on core %s (%s: %s). Falling back to an ideal-cycle estimate.",
                node.name,
                core.id,
                type(exc).__name__,
                exc,
            )
            from functools import reduce  # noqa: PLC0415

            dim_sizes = self.get_layer_node_attributes(node).layer_dim_sizes
            total_ops = float(reduce(lambda a, b: a * b, dim_sizes.data.values(), 1))
            # Spread the work over the operational array and charge the op's real cycle cost. The
            # getattr default guards an aie2 tile, whose Core.__getattr__ raises instead of returning None.
            array = getattr(core, "operational_array", None)
            unit_count = max(1, int(getattr(array, "total_unit_count", 1) or 1))
            ideal_cycle = ceil(total_ops / unit_count) * self.get_cc_per_op(node.type)
            return CoreCostEntry(
                energy_total=0.0,
                latency_total=float(ideal_cycle),
                ideal_cycle=float(ideal_cycle),
                ideal_temporal_cycle=float(ideal_cycle),
                cme=None,
                mapping=None,
                layer=node,
                metadata={"backend": IDEAL_CYCLE_BACKEND},
            )

    def run_zigzag(self, node: ComputationNode, core: Core) -> CostModelEvaluation:
        """Run the ZigZag flow to estimate performance of a given node on a core: its best spatial mapping unrolling
        one loop per array dimension or, where unrolling several fills more of the array, as an array maps an im2col
        convolution's whole contraction over its rows, the faster of the two. ZigZag ranks spatial mappings by how
        much of the array they fill, and among those that fill it alike prefers the one spreading over most loops,
        which can starve an operand of bandwidth; only the cost model tells which of the two is faster."""
        logger.info(f"Launching intra-core mapping optimization for {node} -> {core} ...")
        cmes: list[CostModelEvaluation] = []
        failure: Exception | None = None
        for mix in (False, True):
            if mix and cmes and self._array_fill(node, core, True) <= self._array_fill(node, core, False):
                break
            try:
                answers = self.instantiate_zigzag_flow(copy.deepcopy(node), core, mix).run()
            except Exception as exc:  # noqa: BLE001 -- the other kind of spatial mapping may still cost it
                failure = exc
                continue
            assert len(answers) == 1, "CoreCostEstimationStage's subflow returned more than one cost entry"
            cmes.append(answers[0][0])  # type: ignore
        if not cmes:
            assert failure is not None
            raise failure
        return min(cmes, key=lambda cme: cme.latency_total2)

    @staticmethod
    def _array_fill(node: ComputationNode, core: Core, mix: bool) -> int:
        """The units of the array the best spatial mapping of either kind keeps busy, found without costing it."""
        stage = SpatialMappingGeneratorStage(
            [CostModelStage],
            accelerator=core.to_zigzag_core(),
            layer=copy.deepcopy(node),
            enable_mix_spatial_mapping_generation=mix,
            nb_mappings_generated=1,
        )
        return next(stage.generate_spatial_mappings()).hw_utilization

    def instantiate_zigzag_flow(self, node: ComputationNode, core: Core, mix: bool = False) -> _KwargsMainStage:
        """Instantiate a runnable ZigZag mainstage, generating spatial mappings that unroll several loops over one
        array dimension if ``mix``."""
        main_stage = _KwargsMainStage(
            [  # Initializes the MainStage as entry point
                MinimalLatencyStage,  # type: ignore
                SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
                MinimalLatencyStage,  # Reduces all CMEs, returning minimal EDP one
                TemporalMappingGeneratorStage,  # Generates multiple temporal mappings (TM)
                CostModelStage,  # Evaluates generated SM and TM through cost model
            ],
            layer=node,
            accelerator=core.to_zigzag_core(),  # Pass the inner ZigZag core to ZigZag stages
            loma_lpf_limit=self.loma_lpf_limit,  # required by LomaEngine
            loma_show_progress_bar=self.loma_show_progress_bar,
            temporal_mapping_type=self.temporal_mapping_type,
            nb_mappings_generated=self.nb_spatial_mappings_generated,
            enable_mix_spatial_mapping_generation=mix,
        )
        return main_stage

    def get_cc_per_op(self, op_type: str):
        """Return the number of cycles that the operational units need to finish the given operation."""
        match op_type.lower():
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
        new_cme = CostModelEvaluation(
            accelerator=cme.accelerator,
            layer=cme.layer,
            spatial_mapping=cme.spatial_mapping,
            spatial_mapping_int=cme.spatial_mapping_int,
            temporal_mapping=cme.temporal_mapping,
            access_same_data_considered_as_no_access=cme.access_same_data_considered_as_no_access,
            cycles_per_op=cc_per_op,
        )
        if cc_per_op > 1:
            logger.warning(
                f"ZigZagCostEstimator: Increasing cycles per operation for op type {op_type} to {cc_per_op} cycles."
            )
        return new_cme
