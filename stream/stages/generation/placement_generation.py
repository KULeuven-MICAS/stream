import logging
from dataclasses import replace

from stream.datatypes import LayerDim
from stream.hardware.architecture.core import Core
from stream.mapping.chain_placement import (
    bandwidth_bound,
    column_budgets,
    layer_cost,
    row_counts,
    widest_columns,
)
from stream.mapping.mapping import Mapping
from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.workload.workload import ComputationNode, Workload

logger = logging.getLogger(__name__)

ELEMENT_BYTES = 2
BUFFERS = 2


class PlacementGenerationStage(Stage):
    """Place the layers a mapping leaves unplaced, from the workload and the kernels.

    A fused group that is a linear chain short enough to stack gets a row tenancy: rows
    per layer from the bottleneck of measured kernel cycles (a layer wider than its
    consumer pays the tiled-handover factor), the shared dimension split over the widest
    column count it divides, cores ordered rows-outermost so every handover lands in the
    consumer's column. Any other multi-layer group gets disjoint column tenancies by the
    same bottleneck rule. A layer alone in its group takes the whole array -- all rows
    for a contracting kernel, one core per column beside the memory tile for a
    bandwidth-bound one. Declared placements are never touched.
    """

    REQUIRED_FIELDS = ("workload", "mapping", "accelerator")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.workload: Workload = self.ctx.get("workload")
        self.mapping: Mapping = self.ctx.get("mapping")
        self.accelerator = self.ctx.get("accelerator")

    def run(self):
        grid = {
            (c.col_id, c.row_id): c
            for c in self.accelerator.core_list
            if c.type == "compute" and c.col_id is not None and c.row_id is not None
        }
        if grid:
            for group in self.mapping.fused_groups:
                nodes = [n for name in group.layers if isinstance(n := self._node(name), ComputationNode)]
                if nodes and all(self._unplaced(n) for n in nodes):
                    self._place_group(nodes, grid)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _node(self, name: str):
        try:
            return self.workload.get_node_by_name(name)
        except Exception:  # noqa: BLE001 -- a group may name a node this sub-workload lacks
            return None

    def _unplaced(self, node: ComputationNode) -> bool:
        nm = self.mapping.get(node)
        return not nm.resource_allocation and nm.kernel is not None

    def _place_group(self, nodes: list[ComputationNode], grid: dict[tuple[int, int], Core]) -> None:
        columns = sorted({col for col, _ in grid})
        rows = sorted({row for _, row in grid})
        if len(nodes) == 1:
            self._place_alone(nodes[0], grid, columns, rows)
        elif len(nodes) <= len(rows) and self._linear(nodes):
            self._place_stacked(nodes, grid, columns, rows)
        else:
            self._place_tenancies(nodes, grid, columns, rows)

    def _linear(self, nodes: list[ComputationNode]) -> bool:
        members = set(nodes)
        for i, node in enumerate(nodes[:-1]):
            downstream = {s for s in self.workload.successors(node) if s in members}
            if downstream != {nodes[i + 1]}:
                return False
        return True

    def _place_alone(self, node: ComputationNode, grid, columns, rows) -> None:
        kernel = self.mapping.get(node).kernel
        if bandwidth_bound(kernel):
            row = rows[0]  # beside the memory tile that feeds it
            width = self._fitting_split(node, 0, len(columns))
            cores = tuple(grid[(col, row)] for col in columns[:width])
            self._assign(node, cores, ((0, width),))
            width = self._row_width(node, kernel, next(iter(grid.values())))
            self._retile(node, kernel, m=1, n=width, layout="contiguous")
            return
        granule = dict(kernel.granule())
        split = [(0, self._fitting_split(node, 0, len(rows), granule.get(0, 1)))]
        cols_used = 1
        # D2 splits across all columns or not at all: every measured design does one of
        # the two, and a partial-width scatter of a short output dimension is exactly the
        # shape that hung the generated k=3 attention scores on hardware.
        if len(granule) > 2 and self._fitting_split(node, 2, len(columns), granule.get(2, 1)) == len(columns):
            cols_used = len(columns)
            split.append((2, cols_used))
        cores = tuple(grid[(col, row)] for col in columns[:cols_used] for row in rows[: split[0][1]])
        self._assign(node, cores, tuple(split))

    def _place_stacked(self, nodes: list[ComputationNode], grid, columns, rows) -> None:
        kernels = [self.mapping.get(n).kernel for n in nodes]
        counts = row_counts([layer_cost(k) for k in kernels], len(rows))
        granule = dict(kernels[0].granule()).get(0, 1)
        extent = self.workload.get_dimension_size(self.workload.get_dims(nodes[0])[0])
        width = widest_columns(extent, granule, max(counts), len(columns))
        offset = 0
        for node, kernel, r in zip(nodes, kernels, counts):
            layer_rows = rows[offset : offset + r]
            offset += r
            cores = tuple(grid[(col, row)] for row in layer_rows for col in columns[:width])
            self._assign(node, cores, ((0, width * r),))
        for node, kernel, r, r_next in zip(nodes, kernels, counts, (*counts[1:], counts[-1])):
            if r > r_next:
                self._retile(node, kernel, tiled_out=True)

    def _place_tenancies(self, nodes: list[ComputationNode], grid, columns, rows) -> None:
        kernels = [self.mapping.get(n).kernel for n in nodes]
        budgets = column_budgets([layer_cost(k) for k in kernels], len(columns))
        first = 0
        for node, kernel, budget in zip(nodes, kernels, budgets):
            tenant = columns[first : first + budget]
            first += budget
            granule = dict(kernel.granule())
            split = [(0, self._fitting_split(node, 0, len(rows), granule.get(0, 1)))]
            width = 1
            if len(granule) > 2 and budget > 1:
                width = self._fitting_split(node, 2, budget, granule.get(2, 1))
                if width > 1:
                    split.append((2, width))
            cores = tuple(grid[(col, row)] for col in tenant[:width] for row in rows[: split[0][1]])
            self._assign(node, cores, tuple(split))

    def _fitting_split(self, node: ComputationNode, position: int, target: int, granule: int = 1) -> int:
        """The widest split that still hands every core whole granules."""
        extent = self.workload.get_dimension_size(self.workload.get_dims(node)[position])
        for split in range(target, 0, -1):
            if extent % (split * granule) == 0:
                return split
        return 1

    def _row_width(self, node: ComputationNode, kernel, core: Core) -> int:
        extent = self.workload.get_dimension_size(self.workload.get_dims(node)[1])
        operands = max(2, len(kernel.operand_layouts() or ()))
        budget = (core.get_memory_capacity() // 8) // (operands * BUFFERS * ELEMENT_BYTES)
        return max(w for w in range(1, min(extent, budget) + 1) if extent % w == 0)

    def _assign(self, node: ComputationNode, cores: tuple[Core, ...], split: tuple[tuple[int, int], ...]) -> None:
        nm = self.mapping.get(node)
        nm.resource_allocation = (tuple(cores),)
        nm.inter_core_tiling = (tuple((LayerDim(position=p, prefix="d"), s) for p, s in split),)
        logger.info(
            "Placed %s on %s cores, split %s",
            node.name,
            len(cores),
            [(str(d), s) for d, s in nm.inter_core_tiling[0]],
        )

    def _retile(self, node: ComputationNode, kernel, **kwargs) -> None:
        try:
            self.mapping.get(node).kernel = replace(kernel, **kwargs)
        except TypeError:
            logger.info("Kernel of %s takes no %s; placement keeps it as declared", node.name, sorted(kwargs))
