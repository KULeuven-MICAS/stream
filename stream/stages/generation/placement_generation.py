import logging
from dataclasses import replace

from stream.datatypes import LayerDim
from stream.hardware.architecture.core import Core
from stream.mapping.blocks import block_options, with_block
from stream.mapping.chain_placement import (
    MIN_CALL_OPERANDS,
    bandwidth_bound,
    call_sizes,
    column_budget_options,
    is_matmul,
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
    """Place the layers a mapping leaves unplaced, from the workload and the kernels."""

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
        self._pending_fallbacks: list = []
        self._pending_reserves: list = []
        if grid:
            declared = self.mapping
            variants = self._block_variants(declared)
            self._place(declared, grid)
            alternatives = self._materialize(declared, self._pending_fallbacks)
            reserves = self._materialize(declared, self._pending_reserves)
            for variant in variants:
                self._pending_fallbacks, self._pending_reserves = [], []
                self._place(variant, grid)
                alternatives.append(variant)
                alternatives.extend(self._materialize(variant, self._pending_fallbacks))
            self.mapping = declared
            self.ctx.set(mapping=declared, placement_alternatives=alternatives, placement_reserves=reserves)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()

    def _place(self, mapping: Mapping, grid: dict[tuple[int, int], Core]) -> None:
        self.mapping = mapping
        for group in mapping.fused_groups:
            nodes = [n for name in group.layers if isinstance(n := self._node(name), ComputationNode)]
            if nodes and all(self._unplaced(n) for n in nodes):
                self._place_group(nodes, grid)

    @staticmethod
    def _materialize(base: Mapping, pending: list) -> list[Mapping]:
        shapes: list[Mapping] = []
        for narrow in pending:
            alt = (shapes[-1] if shapes else base).copy()
            narrow(alt)
            shapes.append(alt)
        return shapes

    def _block_variants(self, mapping: Mapping) -> list[Mapping]:
        """One unplaced mapping per other compiled block the groups offer, each placed and priced in full."""
        variants: list[Mapping] = []
        for gi, group in enumerate(mapping.fused_groups):
            for dim, sizes in block_options(self.workload, mapping, group).items():
                extent = self.workload.get_dimension_size(dim)
                for size in sizes:
                    if size > extent or extent % size or (dim, size) in group.intra_core_tiling:
                        continue
                    try:
                        variants.append(with_block(self.workload, mapping, gi, group, dim, size))
                    except ValueError as e:
                        logger.info("Block %s=%d is not buildable: %s", dim, size, e)
        return variants

    def _node(self, name: str):
        try:
            return self.workload.get_node_by_name(name)
        except Exception:  # noqa: BLE001
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
        granule = call_sizes(kernel, node)
        at = kernel.positions(node)
        if bandwidth_bound(kernel):
            self._place_one_row(node, kernel, grid, columns, rows, self.mapping)
            return
        if not is_matmul(kernel):
            self._place_one_row(node, kernel, grid, columns, rows, self.mapping)
            depth = self._fitting_split(node, at["m"], len(rows), granule.get(at["m"], 1))
            narrow = tuple(grid[(columns[0], row)] for row in rows[:depth])
            self._pending_reserves.append(lambda m, n=node, c=narrow, sp=((at["m"], depth),): self._assign(n, c, sp, m))
            return
        split = [(at["m"], self._fitting_split(node, at["m"], len(rows), granule.get(at["m"], 1)))]
        cols_used = 1
        if self._fitting_split(node, at["n"], len(columns), granule.get(at["n"], 1)) == len(columns):
            cols_used = len(columns)
            split.append((at["n"], cols_used))
        cores = tuple(grid[(col, row)] for col in columns[:cols_used] for row in rows[: split[0][1]])
        self._assign(node, cores, tuple(split))

    def _place_one_row(self, node, kernel, grid, columns, rows, mapping) -> None:
        row = rows[0]
        at = kernel.positions(node)
        width = self._fitting_split(node, at["m"], len(columns))
        cores = tuple(grid[(col, row)] for col in columns[:width])
        self._assign(node, cores, ((at["m"], width),), mapping)
        width = self._row_width(node, kernel, next(iter(grid.values())))
        self._retile(node, kernel, mapping, m=1, n=width, layout="contiguous")

    def _place_stacked(self, nodes: list[ComputationNode], grid, columns, rows) -> None:
        kernels = [self.mapping.get(n).kernel for n in nodes]
        state_consumers = frozenset(
            i + 1
            for i, kernel in enumerate(kernels[:-1])
            if kernel is not None and any(s.handover for s in kernel.state_operands())
        )
        counts = row_counts([layer_cost(k) for k in kernels], len(rows), state_consumers=state_consumers)
        rows_at = [kernel.positions(node)["m"] for node, kernel in zip(nodes, kernels, strict=True)]
        granule = max(
            call_sizes(kernel, node).get(at, 1) for node, kernel, at in zip(nodes, kernels, rows_at, strict=True)
        )
        extent = self.workload.get_dimension_size(self.workload.get_dims(nodes[0])[rows_at[0]])
        width = widest_columns(extent, granule, max(counts), len(columns))
        offset = 0
        for node, r, at in zip(nodes, counts, rows_at, strict=True):
            layer_rows = rows[offset : offset + r]
            offset += r
            cores = tuple(grid[(col, row)] for row in layer_rows for col in columns[:width])
            self._assign(node, cores, ((at, width * r),))
        for node, kernel, r, r_next in zip(nodes, kernels, counts, (*counts[1:], counts[-1]), strict=True):
            if r > r_next:
                self._retile(node, kernel, tiled_out=True)

    def _place_tenancies(self, nodes: list[ComputationNode], grid, columns, rows) -> None:
        kernels = [self.mapping.get(n).kernel for n in nodes]
        caps = [len(columns) if is_matmul(k) else 1 for k in kernels]
        options = column_budget_options([layer_cost(k) for k in kernels], len(columns), caps)
        for budgets in options[1:]:
            self._pending_fallbacks.append(
                lambda m, b=budgets: self._assign_tenancy(nodes, kernels, b, grid, columns, rows, m)
            )
        self._assign_tenancy(nodes, kernels, options[0], grid, columns, rows, self.mapping)

    def _assign_tenancy(self, nodes, kernels, budgets, grid, columns, rows, mapping) -> None:
        first = 0
        for node, kernel, budget in zip(nodes, kernels, budgets, strict=True):
            tenant = columns[first : first + budget]
            first += budget
            granule = call_sizes(kernel, node)
            at = kernel.positions(node)
            width = 1
            split = [(at["m"], self._fitting_split(node, at["m"], len(rows), granule.get(at["m"], 1)))]
            if is_matmul(kernel) and budget > 1:
                width = self._fitting_split(node, at["n"], budget, granule.get(at["n"], 1))
                if width > 1:
                    split.append((at["n"], width))
            cores = tuple(grid[(col, row)] for col in tenant[:width] for row in rows[: split[0][1]])
            self._assign(node, cores, tuple(split), mapping)

    def _fitting_split(self, node: ComputationNode, position: int, target: int, granule: int = 1) -> int:
        """The widest split that still hands every core whole granules."""
        extent = self.workload.get_dimension_size(self.workload.get_dims(node)[position])
        for split in range(target, 0, -1):
            if extent % (split * granule) == 0:
                return split
        return 1

    def _row_width(self, node: ComputationNode, kernel, core: Core) -> int:
        extent = self.workload.get_dimension_size(self.workload.get_dims(node)[kernel.positions(node)["n"]])
        operands = max(MIN_CALL_OPERANDS, len(kernel.operand_layouts() or ()))
        budget = (core.get_memory_capacity() // 8) // (operands * BUFFERS * ELEMENT_BYTES)
        return max(w for w in range(1, min(extent, budget) + 1) if extent % w == 0)

    def _assign(
        self, node: ComputationNode, cores: tuple[Core, ...], split: tuple[tuple[int, int], ...], mapping=None
    ) -> None:
        nm = (mapping or self.mapping).get(node)
        nm.resource_allocation = (tuple(cores),)
        nm.inter_core_tiling = (tuple((LayerDim(position=p, prefix="d"), s) for p, s in split),)
        logger.info(
            "Placed %s on %s cores, split %s",
            node.name,
            len(cores),
            [(str(d), s) for d, s in nm.inter_core_tiling[0]],
        )

    def _retile(self, node: ComputationNode, kernel, mapping=None, **kwargs) -> None:
        try:
            (mapping or self.mapping).get(node).kernel = replace(kernel, **kwargs)
        except TypeError:
            logger.info("Kernel of %s takes no %s; placement keeps it as declared", node.name, sorted(kwargs))
