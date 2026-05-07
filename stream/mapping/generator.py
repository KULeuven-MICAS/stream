from __future__ import annotations

import copy
import os
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product
from typing import Any

import yaml

from stream.hardware.architecture.accelerator import Accelerator
from stream.workload.workload import Workload


class MappingOrder(Enum):
    """Ordering strategy applied to the enumerated mapping variants before emission."""

    RANDOM = auto()
    """Shuffle variants randomly (reproducible via the fixed RNG seed)."""

    UTILIZATION_DESCENDING = auto()
    """Sort variants by total core utilisation (descending): variants that allocate more
    cores across all layers come first."""

    UTILIZATION_ASCENDING = auto()
    """Sort variants by total core utilisation (ascending): variants that allocate more
    cores across all layers come last."""


@dataclass(frozen=True)
class SplitSpec:
    """
    Inter-core tiling spec.
    dim: dimension name WITHOUT changing it (ex: "D0", "D1", "D2")
    split: number of splits along that dim (must be >= 1)
    """

    dim: str
    split: int


def _divisors(n: int) -> list[int]:
    ds = []
    for d in range(1, n + 1):
        if n % d == 0:
            ds.append(d)
    return ds


class MappingGenerator:
    """
    Generates multiple YAML mapping files by varying ONLY inter-core tiling split sizes,
    keeping the tiling dimensions the same.

    Key rule enforced:
      - For each layer: len(core_allocation) == product(split values in inter_core_tiling)
    Core ids used:
      - Must be compute core ids from the provided 'accelerator' object.
    """

    def __init__(  # noqa: PLR0913
        self,
        accelerator: Accelerator,
        workload: Workload,
        output_dir: str,
        *,
        last_gemm_down: bool,
        # Intra-core tile sizes (kept fixed here)
        seq_len_tile_size: int,
        embedding_tile_size: int,
        hidden_tile_size: int,
        # How many variants to emit (None means "all")
        max_variants: int | None = None,
        # If True, require all layers to use disjoint cores in each mapping.
        disjoint_cores_per_layer: bool = True,
        layer_core_splits: dict[str, list[int]] | None = None,
        # Cap on the number of inter-core split shapes kept PER (layer, total) bucket,
        # after the array-shape and balance filters. For example, on a 4x8 array a
        # Gemm layer with ``layer_core_splits=[4, 8, 16]`` and a cap of 1 will yield
        # one shape per total -> 3 shapes total: e.g. ``(2,2) / (4,2) / (4,4)``.
        # Within each total bucket, surviving shapes are sorted by the split on the
        # FIRST inter-core dim (descending) so the largest unrolling on dim[0] wins.
        # Layer names not present in the dict get no cap. Pass alongside
        # ``layer_core_splits`` since both express per-layer policy.
        layer_max_shapes_per_total: dict[str, int] | None = None,
        ordering: MappingOrder = MappingOrder.UTILIZATION_DESCENDING,
        # Physical compute-array shape used to constrain inter-core tiling.
        # If both are provided, a layer's per-dim splits must fit the rows x cols
        # rectangle (see _splits_fit_array_shape). If either is None, the
        # constraint is skipped (legacy behaviour).
        nb_rows: int | None = None,
        nb_cols: int | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.workload = workload
        self.output_dir = output_dir
        self.layer_core_splits = layer_core_splits or {}
        self.rng = random.Random(42)

        input_tensor = workload.tensors[0]
        assert "input" in input_tensor.name.lower(), (
            "Expected workload to have an input tensor with 'input' in its name."
        )
        seq_len, embedding_dim = input_tensor.shape
        weights_1_tensor = next(t for t in workload.tensors if "weights_1" in t.name.lower())
        assert weights_1_tensor.shape[0] == embedding_dim, (
            "Expected weights_1 tensor's first dim to match embedding_dim from input tensor."
        )
        hidden_dim = weights_1_tensor.shape[1]
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.last_gemm_down = last_gemm_down

        self.seq_len_tile_size = seq_len_tile_size
        self.embedding_tile_size = embedding_tile_size
        self.hidden_tile_size = hidden_tile_size

        self.max_variants = max_variants
        self.disjoint_cores_per_layer = disjoint_cores_per_layer
        self.ordering = ordering
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.layer_max_shapes_per_total = layer_max_shapes_per_total or {}

        os.makedirs(self.output_dir, exist_ok=True)

        self.compute_core_ids = self._get_compute_core_ids(self.accelerator)
        if not self.compute_core_ids:
            raise ValueError("No compute core ids found from accelerator object.")

        self._validate_problem_sizes()

    # -------------------------
    # Public API
    # -------------------------
    def run(self) -> Iterator[tuple[int, list[tuple[str, list[SplitSpec]]], dict[str, Any]]]:
        """
        Yields (idx, variant, mapping_dict) for VALID mappings only.
        idx is contiguous over valid mappings: 0..N-1.

        The order in which variants are visited is controlled by ``self.ordering``:
        - ``MappingOrder.RANDOM``      - random shuffle (reproducible via the fixed seed)
        - ``MappingOrder.UTILIZATION`` - descending total core count across all layers
        """
        layer_templates = self._build_layer_templates()
        per_layer_split_options = self._enumerate_inter_core_split_options(layer_templates)

        if any(len(opts) == 0 for opts in per_layer_split_options):
            return

        all_variants = list(product(*per_layer_split_options))
        all_variants = self._order_variants(all_variants)

        emitted = 0
        target = self.max_variants if self.max_variants is not None else float("inf")

        for combo in all_variants:
            if emitted >= target:
                break

            variant = list(combo)
            mapping = self._assemble_mapping(layer_templates, variant)
            if mapping is None:
                continue

            yield emitted, variant, mapping
            emitted += 1

    # -------------------------
    # Variant ordering strategies
    # -------------------------
    def _order_variants(
        self,
        variants: list[tuple[tuple[str, list[SplitSpec]], ...]],
    ) -> list[tuple[tuple[str, list[SplitSpec]], ...]]:
        """Dispatch to the appropriate ordering function based on ``self.ordering``."""
        match self.ordering:
            case MappingOrder.RANDOM:
                return self._sort_variants_random(variants)
            case MappingOrder.UTILIZATION_DESCENDING:
                return self._sort_variants_by_utilization(variants)
            case MappingOrder.UTILIZATION_ASCENDING:
                return self._sort_variants_by_utilization(variants, reverse=False)

    def _sort_variants_random(
        self,
        variants: list[tuple[tuple[str, list[SplitSpec]], ...]],
    ) -> list[tuple[tuple[str, list[SplitSpec]], ...]]:
        """Shuffle variants in-place using the seeded RNG and return them."""
        self.rng.shuffle(variants)
        return variants

    def _sort_variants_by_utilization(
        self, variants: list[tuple[tuple[str, list[SplitSpec]], ...]], reverse: bool = True
    ) -> list[tuple[tuple[str, list[SplitSpec]], ...]]:
        """Sort variants by total cores allocated summed across all layers, descending.

        A variant that allocates more cores in total (higher utilisation) appears first.
        """

        def _total_cores(variant: tuple[tuple[str, list[SplitSpec]], ...]) -> int:
            return sum(self._num_cores_needed(specs) for (_, specs) in variant)

        return sorted(variants, key=_total_cores, reverse=reverse)

    # -------------------------
    # Core building blocks
    # -------------------------
    def _validate_problem_sizes(self) -> None:
        if self.seq_len % 4 != 0:
            raise AssertionError("seq_len must be divisible by 4 for these mappings.")
        if self.seq_len < self.seq_len_tile_size * 4:
            raise AssertionError(f"seq_len must be at least {self.seq_len_tile_size * 4} for this mapping.")
        if self.embedding_dim % self.embedding_tile_size != 0:
            raise AssertionError(f"embedding_dim must be divisible by {self.embedding_tile_size}.")
        if self.hidden_dim % self.hidden_tile_size != 0:
            raise AssertionError(f"hidden_dim must be divisible by {self.hidden_tile_size}.")

    def _get_compute_core_ids(self, accelerator: Accelerator) -> list[int]:
        """
        Get the available compute cores from the accelerator object.
        """
        return [core.id for core in accelerator.core_list if core.type == "compute"]

    def _kernel_gemm(self) -> dict[str, Any]:
        if self.seq_len_tile_size == 1:
            return {"name": "matvec", "kwargs": {"utilization": 61.8, "layout": "default"}}

        return {
            "name": "gemm",
            "kwargs": {
                "utilization": 61.8,
                "m": self.seq_len_tile_size,
                "k": self.embedding_tile_size,
                "n": self.hidden_tile_size,
                "layout": "default",
            },
        }

    def _kernel_silu(self) -> dict[str, Any]:
        return {"name": "silu", "kwargs": {"utilization": 50.0, "layout": "default"}}

    def _kernel_mul(self) -> dict[str, Any]:
        return {"name": "eltwise_mul", "kwargs": {"utilization": 50.0, "layout": "default"}}

    def _build_layer_templates(self) -> list[dict[str, Any]]:
        """
        Defines layer names, kernels, and the inter-core dimension order that must NOT change.
        Only split sizes will vary.
        """
        layers: list[dict[str, Any]] = []

        # Gemm_Left (keep dims D2 then D0, only split values vary)
        layers.append(
            {
                "name": "Gemm_Left",
                "kernel": self._kernel_gemm(),
                "inter_core_dims": ["D0", "D2"],
                # Dimension sizes for choosing legal split factors (edit if needed)
                "dim_sizes": {"D0": self.seq_len, "D2": self.hidden_dim},
            }
        )

        # Gemm_Right
        layers.append(
            {
                "name": "Gemm_Right",
                "kernel": self._kernel_gemm(),
                "inter_core_dims": ["D0", "D2"],
                "dim_sizes": {"D0": self.seq_len, "D2": self.hidden_dim},
            }
        )

        # Silu (SIMDParser for (B,H) -> treat as split along D0 only)
        layers.append(
            {
                "name": "Silu",
                "kernel": self._kernel_silu(),
                "inter_core_dims": ["D0"],
                "dim_sizes": {"D0": self.seq_len},
            }
        )

        # Eltwise Mul
        layers.append(
            {
                "name": "Elt_Mul",
                "kernel": self._kernel_mul(),
                "inter_core_dims": ["D0"],
                "dim_sizes": {"D0": self.seq_len},
            }
        )

        # Optional Gemm_Down (keep dims D1 then D0)
        if self.last_gemm_down:
            layers.append(
                {
                    "name": "Gemm_Down",
                    "kernel": self._kernel_gemm(),
                    "inter_core_dims": ["D0", "D2"],
                    # If your Gemm_Down semantics differ, update these sizes.
                    "dim_sizes": {"D0": self.seq_len, "D2": self.embedding_dim},
                }
            )

        return layers

    def _splits_fit_array_shape(self, splits: Sequence[int]) -> bool:
        """
        Reject inter-core split combinations that cannot be physically laid out on the
        ``nb_rows x nb_cols`` compute array.

        - 1D split ``(s,)``     -> must fit a single row OR a single column, i.e. ``s <= max(nb_rows, nb_cols)``.
        - 2D split ``(s0, s1)`` -> the two factors must fit the rectangle in either orientation:
              ``(s0 <= nb_rows AND s1 <= nb_cols)`` OR ``(s0 <= nb_cols AND s1 <= nb_rows)``.
              This rejects cases like ``(16, 1)`` on a 4x8 array where one dim alone exceeds both axes.
        - Higher-dim splits are passed through unchanged (no current layer uses >2 inter-core dims).
        - If ``nb_rows`` or ``nb_cols`` is unset, the check is skipped (legacy behaviour).
        """
        if self.nb_rows is None or self.nb_cols is None:
            return True

        rows, cols = self.nb_rows, self.nb_cols
        n = len(splits)

        if n == 1:
            return int(splits[0]) <= max(rows, cols)
        if n == 2:  # noqa: PLR2004
            s0, s1 = int(splits[0]), int(splits[1])
            return (s0 <= rows and s1 <= cols) or (s0 <= cols and s1 <= rows)
        return True

    def _enumerate_inter_core_split_options(
        self, layer_templates: Sequence[dict[str, Any]]
    ) -> list[list[tuple[str, list[SplitSpec]]]]:
        """
        For each layer, produce candidate inter_core_tiling specs, but constrained by:
        - per-dim split must divide the dim size
        - total cores (product of splits) must be in layer_core_splits[layer_name] if provided
        - total cores must be <= available compute cores
        """
        max_cores = len(self.compute_core_ids)
        per_layer_options: list[list[tuple[str, list[SplitSpec]]]] = []

        for tpl in layer_templates:
            lname = tpl["name"]
            dims: list[str] = list(tpl["inter_core_dims"])
            dim_sizes: dict[str, int] = dict(tpl["dim_sizes"])

            # Allowed totals for this layer (if not specified, fall back to "anything up to max_cores")
            allowed_totals = self.layer_core_splits.get(lname, None)
            if allowed_totals is not None:
                # sanitize and keep only feasible totals
                allowed_totals = sorted({int(x) for x in allowed_totals if 1 <= int(x) <= max_cores})
                if not allowed_totals:
                    # No feasible totals -> no options for this layer
                    per_layer_options.append([])
                    continue

            # split choices per dim must be divisors and <= max_cores
            split_choices_per_dim: list[list[int]] = []
            for d in dims:
                size = int(dim_sizes[d])
                choices = [x for x in _divisors(size) if x <= max_cores]
                split_choices_per_dim.append(choices)

            options_for_layer: list[tuple[str, list[SplitSpec]]] = []
            for splits in product(*split_choices_per_dim):
                needed = 1
                for s in splits:
                    needed *= int(s)

                if needed > max_cores:
                    continue
                if allowed_totals is not None and needed not in allowed_totals:
                    continue
                if not self._splits_fit_array_shape(splits):
                    continue

                specs = [SplitSpec(dim=dims[i], split=int(splits[i])) for i in range(len(dims))]
                options_for_layer.append((lname, specs))

            options_for_layer = self._filter_first_dim_priority(options_for_layer)
            options_for_layer = self._cap_options_per_layer(options_for_layer)
            per_layer_options.append(options_for_layer)

        return per_layer_options

    def _cap_options_per_layer(self, options: list[tuple[str, list[SplitSpec]]]) -> list[tuple[str, list[SplitSpec]]]:
        """
        Within each ``total = prod(splits)`` bucket of this layer, keep at most
        ``layer_max_shapes_per_total[lname]`` shapes. The cap is applied INSIDE
        each total, not across totals, so a layer with totals ``[4, 8, 16]`` and
        a cap of 1 still yields three shapes (one per total).

        Within a bucket, shapes are sorted by the split on the FIRST entry in
        ``inter_core_dims`` (descending) so the largest unrolling on dim[0]
        wins; remaining dims are used as secondary sort keys for determinism.

        Layer names not present in ``self.layer_max_shapes_per_total`` are
        passed through unmodified.
        """
        if not options:
            return options

        lname = options[0][0]
        cap = self.layer_max_shapes_per_total.get(lname)
        if cap is None or cap <= 0:
            return options

        def _sort_key(entry: tuple[str, list[SplitSpec]]) -> tuple[int, ...]:
            specs = entry[1]
            # Descending by leading-dim split, then by remaining splits to break ties.
            return tuple(-int(s.split) for s in specs)

        # Group by total core count, then keep the top-`cap` shapes per group.
        buckets: dict[int, list[tuple[str, list[SplitSpec]]]] = {}
        for entry in options:
            total = 1
            for s in entry[1]:
                total *= int(s.split)
            buckets.setdefault(total, []).append(entry)

        kept: list[tuple[str, list[SplitSpec]]] = []
        for total in sorted(buckets):
            kept.extend(sorted(buckets[total], key=_sort_key)[:cap])
        return kept

    def _filter_first_dim_priority(
        self, options: list[tuple[str, list[SplitSpec]]]
    ) -> list[tuple[str, list[SplitSpec]]]:
        """
        Per-column growth: for each total bucket, prefer the 2D shape whose first
        inter-core dim saturates ``nb_rows`` before the second dim starts to grow.

        Concretely, for total ``N`` on a ``nb_rows x nb_cols`` array we want
        ``first_dim = min(N, nb_rows)`` and ``second_dim = N / first_dim``. So on a
        4x8 array:

            total=4  -> (4, 1)
            total=8  -> (4, 2)
            total=16 -> (4, 4)
            total=32 -> (4, 8)

        This guarantees the first dim is the one that "grows" as the total core
        count grows, until it caps at ``nb_rows``; only then does the second dim
        grow.

        The selection is implemented as a sort key over candidates inside each
        total bucket:
          1. ``|first_dim - target|`` (smaller = closer to "fills rows first")
          2. penalty for ``first_dim > nb_rows`` (rejects axis-swap orientations
             like ``(8, 2)`` even when they otherwise fit the rectangle)
          3. ``-first_dim`` to break remaining ties in favour of larger leading dim

        Only the candidates tied for the best score in each bucket are kept.

        1D options pass through unchanged. If ``nb_rows`` is unset, the filter
        is skipped (legacy behaviour) and all candidates remain.
        """
        if not options or self.nb_rows is None:
            return options

        rows = int(self.nb_rows)

        kept: list[tuple[str, list[SplitSpec]]] = []
        buckets: dict[int, list[tuple[str, list[SplitSpec]]]] = {}

        for entry in options:
            _, specs = entry
            if len(specs) <= 1:
                kept.append(entry)
                continue
            total = 1
            for s in specs:
                total *= int(s.split)
            buckets.setdefault(total, []).append(entry)

        for total, entries in buckets.items():
            target = min(total, rows)

            def _score(entry: tuple[str, list[SplitSpec]], target: int = target) -> tuple[int, int, int]:
                s0 = int(entry[1][0].split)
                return (abs(s0 - target), max(0, s0 - rows), -s0)

            best = min(_score(e) for e in entries)
            kept.extend(e for e in entries if _score(e) == best)

        return kept

    def _assemble_mapping(
        self,
        layer_templates: Sequence[dict[str, Any]],
        variant: Sequence[tuple[str, list[SplitSpec]]],
    ) -> dict[str, Any] | None:
        """
        Given a variant (layer_name -> split specs), allocate cores and construct the mapping dict.
        Returns None if allocation is impossible.
        """
        # Build a dict name->splitspecs
        splits_by_name = {name: specs for (name, specs) in variant}

        # Allocate cores
        remaining = list(self.compute_core_ids)
        core_alloc_by_name: dict[str, list[int]] = {}

        for tpl in layer_templates:
            lname = tpl["name"]
            specs = splits_by_name[lname]
            needed = self._num_cores_needed(specs)

            if needed > len(remaining):
                return None

            if self.disjoint_cores_per_layer:
                core_alloc_by_name[lname] = remaining[:needed]
                remaining = remaining[needed:]
            else:
                # If you allow reuse, just take the first needed cores every time.
                core_alloc_by_name[lname] = self.compute_core_ids[:needed]

        # Build layers list in template order
        layers: list[dict[str, Any]] = []
        for tpl in layer_templates:
            lname = tpl["name"]
            specs = splits_by_name[lname]
            layer_entry = {
                "name": lname,
                "core_allocation": [
                    copy.deepcopy(core_alloc_by_name[lname]),
                ],
                "inter_core_tiling": [
                    [{"dim": s.dim, "split": s.split} for s in specs],
                ],
                "kernel": copy.deepcopy(tpl["kernel"]),
            }
            layers.append(layer_entry)

        fused_groups = self._build_fused_groups([layer["name"] for layer in layers])
        runtime_args = self._build_runtime_args()

        return {
            "layers": layers,
            "fused_groups": [fused_groups],
            "runtime_args": runtime_args,
        }

    def _num_cores_needed(self, specs: Sequence[SplitSpec]) -> int:
        n = 1
        for s in specs:
            n *= int(s.split)
        return n

    def _build_fused_groups(self, layer_names: list[str]) -> dict[str, Any]:
        """
        Kept identical to your original fused group tiling (intra-core tiling),
        since you asked to vary inter-core tiling first.
        """
        fused_groups = {
            "name": "Fused_Group_1",
            "layers": layer_names,
            "intra_core_tiling": [
                {"dim": "Gemm_Left.D1", "tile": self.embedding_tile_size},
                {"dim": "Gemm_Left.D2", "tile": self.hidden_tile_size},
                {"dim": "Gemm_Left.D0", "tile": self.seq_len_tile_size},
            ],
        }
        if self.last_gemm_down:
            fused_groups["intra_core_tiling"].insert(1, {"dim": "Gemm_Down.D2", "tile": self.embedding_tile_size})
        return fused_groups

    def _build_runtime_args(self) -> dict[str, Any]:
        if self.last_gemm_down:
            return {
                "input": {},
                "weights_1": {},
                "weights_2": {},
                "weights_3": {},
                "output": {},
            }
        return {
            "input": {},
            "weights_1": {},
            "weights_2": {},
            "output": {},
        }

    def save_mapping(
        self,
        mapping: dict[str, Any],
        variant: Sequence[tuple[str, list[SplitSpec]]],
        idx: int,
        output_dir: str,
    ) -> str:
        """
        Save mapping YAML into `output_dir` and return the full path.
        """
        os.makedirs(output_dir, exist_ok=True)

        def fmt_layer(layer_name: str, specs: list[SplitSpec]) -> str:
            parts = [layer_name]
            for s in specs:
                parts.append(f"{s.dim}x{s.split}")
            return "_".join(parts)

        tag = "__".join(fmt_layer(n, specs) for (n, specs) in variant)

        base = f"swiglu_{self.seq_len}_{self.embedding_dim}_{self.hidden_dim}"
        filename = f"{base}_mapping.yaml"
        out_path = os.path.join(output_dir, filename)

        mapping = dict(mapping)
        mapping["meta"] = {
            "idx": idx,
            "variant_tag": tag,
            "asctime": time.asctime(),
            "seq_len": self.seq_len,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "seq_len_tile_size": self.seq_len_tile_size,
            "embedding_tile_size": self.embedding_tile_size,
            "hidden_tile_size": self.hidden_tile_size,
            "last_gemm_down": self.last_gemm_down,
            "disjoint_cores_per_layer": self.disjoint_cores_per_layer,
            "layer_core_splits": getattr(self, "layer_core_splits", {}),
            "ordering": self.ordering.name,
        }

        # Atomic write
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(mapping, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, out_path)

        return out_path


# print("Generated mapping files:")
# for p in paths:
#     print("  ", p)
