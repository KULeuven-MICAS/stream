# Variable Tile Size Constraint Optimization

This document explains how the constraint optimizer (CO) treats tile sizes as
decision variables, allowing the Gurobi MILP solver to jointly select tile sizes
alongside memory allocation, transfer scheduling, and FIFO sizing.

## Background

The CO allocates tensors to cores, selects transfer paths, and schedules
timeslots for an AIE array. Each workload dimension (e.g. seq_len, embedding,
hidden) has a **tile size** that determines how much data each core processes
per invocation. Previously tile sizes were fixed scalars chosen before the CO
ran. Now they are decision variables *inside* the MILP.

## Variables at a glance

| Symbol | Type | Meaning |
|--------|------|---------|
| `dim` | identifier | A unique workload dimension, e.g. `z0` (embedding=2048) |
| `k` | index | Index into the candidate tile list for a dimension |
| `w[dim, k]` | BINARY | 1 if candidate `k` is selected for dimension `dim` |
| `tile_var[dim]` | INTEGER | The chosen tile size, reconstructed from `w` |
| `s` (stop) | index | Reuse depth in the temporal loop nest (-1 = no reuse) |
| `z_stop[t, s]` | BINARY | 1 if tensor `t` is buffered up to loop level `s` |
| `jw` (joint_w) | BINARY | AND-product of `w` variables across multiple dimensions |
| `K` | scalar | Kernel loop size = tile size |
| `S` | scalar | Spatial unrolling factor (fixed per dimension) |
| `T` | scalar | Temporal loop iterations = workload_size / (K * S) |
| `y[tr, path]` | BINARY | 1 if transfer `tr` uses a particular multicast path |
| `u` | BINARY | 1 if a tensor is placed on a specific core |
| `lc` | CONTINUOUS | Auxiliary variable for linearizing expression * binary |
| `M` | scalar | Big-M bound, computed as tight per-constraint maximum |
| `per_iter_net` | CONTINUOUS | Per-iteration net latency: sum(slot_lat) - overlap |
| `aux_k` | CONTINUOUS | Linearisation auxiliary: `jw_global_k * per_iter_net` |

## 1. Tile selection variables

For each unique workload dimension, one binary variable is created per
candidate tile size. A one-hot constraint ensures exactly one is selected.

```
w[z0, 0], w[z0, 1], w[z0, 2]    (candidates: 16, 32, 64)
sum_k w[z0, k] == 1              (one-hot)
tile_var[z0] == sum_k(tile[k] * w[z0, k])
```

**Code:** `__create_tile_selection_vars` (~L730).

## 2. Joint candidate enumeration

A tensor's size may depend on multiple tiled dimensions. For example, a tensor
shaped by both `z0` and `z1` requires enumerating all (z0_candidate, z1_candidate)
combinations:

```
z0 candidates: [16, 32]     z1 candidates: [128, 256]
joint combos:  (16,128), (16,256), (32,128), (32,256)
```

For each combo, the tensor size is pre-computed as a scalar. The combo is
selected by a **joint binary**: the AND-product of the individual `w` variables.

```
jw = w[z0, k1] AND w[z1, k2]
```

This AND is linearised into a standard MILP auxiliary variable via
`_add_binary_product`. For three or more dimensions the products are chained
recursively. Results are cached by `(dim, option_index)` key in `_jw_cache`
so that different subsystems (SSIS coefficients, tensor sizing, latency
constraints) share the same Gurobi variable for the same combination.

The result is a linear expression for the tensor size:

```
tensor_size_expr = sum_combo(size[combo] * jw[combo])
```

Since the `w` variables are one-hot per dimension, exactly one `jw` is 1, so
the expression evaluates to the correct pre-computed size.

**Code:** `_joint_candidates_for_tensor` (~L1850),
`_joint_binary_for_combo` (~L1920).

## 3. Memory capacity constraints

Each tensor placed on a core contributes to that core's memory load. The
contribution depends on three things:

- **u**: whether the tensor is on this core (binary)
- **z_stop**: which reuse level is selected (binary)
- **tensor size**: now a linear expression over tile candidates

The product `u * z_stop * size_expr` is a triple product of two binaries and
a linear expression. This is linearised using a continuous auxiliary variable
`lc` with big-M activation:

```
uz = u AND z_stop[t, s]                    (binary product)
lc <= combined_tile_expr                    (upper-bound by expression value)
lc <= M * uz                               (zero when uz = 0)
lc >= combined_tile_expr - M * (1 - uz)    (equals expression when uz = 1)
core_load[c] += lc
```

The big-M bound is tight: `M = max over all candidate combos of
ceil(size_factor * tensor_size)`.

When a tensor depends on multiple tiled dimensions, the (SSIS candidate,
tensor candidate) pairs are enumerated, producing pre-computed
`ceil(sf * size)` coefficients for each pair.

**Code:** `_memory_capacity_constraints` (~L960).

## 4. SSIS loop sizes, reuse, and fire rates

The Steady-State Iteration Space (SSIS) defines the temporal loop nest for
each transfer. Each dimension has a loop decomposition:

```
K * S * T = workload_size
K = tile_size (variable)
S = spatial_unrolling (fixed)
T = workload_size / (K * S) (derived)
```

`_ssis_coefficients_for_transfer` enumerates all tile candidate combinations
across the transfer's **applicable** temporal dimensions (those with
LoopEffect.VARYING or INVARIANT, excluding ABSENT). For each combo, it
computes the temporal loop sizes T. From those it derives:

- **fires**: how many times the transfer fires at each stop level
- **size_factor (sf)**: data reuse factor at each stop level
- **tiles_needed / bds_needed**: buffer depth requirements

These are stored as lists of `(coefficient, jw)` pairs per (tensor, stop_level).

The fire rate for a transfer then becomes:

```
fires[tr] = sum_s( z_stop[t,s] * sum_k(fires_coeff[k,s] * jw[k]) )
```

The inner sum is a linear expression; multiplying by the `z_stop` binary uses
the same big-M `lc` auxiliary pattern.

**Code:** `_ssis_coefficients_for_transfer` (~L300),
`_transfer_fire_rate_constraints` (~L780),
`_reuse_factor_rate_constraints` (~L830).

## 5. Slot latency constraints

### 5a. Compute nodes

For each compute node, the latency estimator evaluates the per-iteration
compute cost for every candidate tile combination. These are **raw
per-iteration** latencies (no iteration-count scaling):

```
for each tile combo k:
    raw_lat[k] = latency_estimator.estimate(node, core, tiling_k).latency_total

slot_latency[s] >= sum_k(raw_lat[k] * jw[k])
```

**Code:** `_slot_latency_constraints` (~L1260).

### 5b. Transfer nodes

Transfer latency depends on the tensor size (tile-dependent) and the reuse
factor (also tile-dependent). The amortised latency formula is:

```
amortised_latency = ceil(tensor_size / bandwidth) / reuse_factor
```

Both numerator and denominator are tile-dependent. To keep this a pure MILP
(no nonlinear division), the approach enumerates all **(k, s) pairs** -
tile candidate `k` crossed with stop level `s`:

```
for each stop level s:
    for each tile candidate k:
        lat_num = ceil(size_bits[k] / min_bw)     # scalar
        amort[k,s] = lat_num / sf_coeff[k,s]      # scalar (pre-computed float)

    expr_s = sum_k(amort[k,s] * jw[k])            # linear expression
    lc_s = z_stop[s] * expr_s                      # big-M linearisation

lat_sum = sum_s(lc_s)
active_latency = y * lat_sum                       # gated by path choice
```

**Code:** `_active_transfer_latency` (~L1960).

## 6. Objective - variable iteration count

The total latency across all steady-state iterations is:

```
total = iterations * sum(slot_lat) - (iterations - 1) * overlap
      = iterations * (sum(slot_lat) - overlap) + overlap
      = iterations * per_iter_net + overlap
```

When tile sizes are variable, the iteration count changes because
`T = workload_size / (K * S)` and `iterations = prod(T)` across all
dimensions. Using a fixed iteration count with per-slot scaling (the naive
approach) breaks down because transfers and compute nodes may have different
subsets of dimensions, leading to inconsistent scaling.

The solution: make **iterations** itself a decision variable by enumerating
all global tile combinations.

### 6a. Variable iteration count

For each global tile combination (Cartesian product of all candidates across
all search-space dimensions), the true iteration count is a known constant:

```
true_iter_k = base_iterations * prod(base_tile_d / cand_tile_d  for all dims d)
```

where `base_iterations` comes from the SSIS temporal sizes and
`base_tile_d` is the first candidate for each dimension.

The effective iteration count is a one-hot selection:

```
iter_var = sum_k( true_iter_k * jw_global_k )
```

### 6b. Linearisation of the objective

The product `iter_var * per_iter_net` is bilinear (linear in binaries *
continuous). It is linearised using one auxiliary variable per global combo:

```
per_iter_net == sum(slot_latency) - overlap       (definition)

for each global combo k:
    aux_k <= M * jw_global_k                      (zero when inactive)
    aux_k <= per_iter_net                          (bounded by expression)
    aux_k >= per_iter_net - M * (1 - jw_global_k) (equals expr when active)

total_latency == sum_k(true_iter_k * aux_k) + overlap
```

Since exactly one `jw_global_k = 1`, the total reduces to
`true_iter_chosen * per_iter_net + overlap`, which is the correct formula.

With 4 dimensions * 4 candidates each, there are 256 global combos.  Each
adds 1 continuous auxiliary variable and 3 constraints - a negligible ~0.01%
increase in model size.

**Code:** `_set_total_latency_and_objective` (~L1585).

### 6c. Why not scale slot latencies instead?

An earlier approach applied iteration scaling factors to individual slot
latencies: `scaled_lat = raw_lat * prod(base/cand)`.  This works for
**compute** nodes (whose dims equal the global dim set), but fails for
**transfer** nodes that have ABSENT dimensions - their SSIS only covers a
subset of dims, so the scale factor is incomplete. The resulting model
systematically overestimates latency for larger tiles on absent dimensions,
biasing the solver toward smaller tiles.

The variable-iteration-count formulation avoids this entirely: slot latencies
are unscaled per-iteration values, and the iteration count is an exact
function of the global tile choice.

## 7. Solution analysis

After each solve, a YAML report is written to
`{output_path}/solution_report.yaml` via `build_solution_report()`. Sections:

| Section | Contents |
|---------|----------|
| `tile_selection` | Candidates and chosen tile per dimension |
| `iterations` | Base (SSIS), true (after tile scaling), per-dim breakdown |
| `slot_latencies` | Per-slot latency value and which node occupies each slot |
| `transfer_summary` | Reuse stop level and fire count per transfer |
| `compute_summary` | Raw latency and slot latency per compute node |
| `objective` | Total latency, per-iteration sum, overlap, cross-check |

The `check_total` field in the objective section recomputes
`true_iterations * per_iter - (true_iterations - 1) * overlap` from the
extracted variable values, serving as an independent sanity check.

**Code:** `build_solution_report`, `save_solution_report` (~L1780).

## The big-M linearisation pattern

Every tile-dependent constraint in the model uses the same pattern to linearise
the product of a linear expression and a binary variable:

```
lc <= expr                     (lc can't exceed the expression)
lc <= M * binary               (lc is zero when binary = 0)
lc >= expr - M * (1 - binary)  (lc equals expr when binary = 1)
```

`M` is a tight upper bound on `expr`, computed as the maximum over all candidate
combinations. Tight bounds reduce the LP relaxation gap, helping the solver
converge faster.

## Degenerate case (single candidate)

When a dimension has only one tile candidate, `w[dim, 0] = 1` is forced by the
one-hot constraint. All joint binaries collapse to 1, all linear expressions
collapse to their single coefficient, and the formulation is equivalent to the
original fixed-tile scalar path. This ensures backward compatibility.

In the objective, a single global combo yields one `jw_global = 1` with
`true_iter = base_iterations`, so the total reduces to the original formula.
