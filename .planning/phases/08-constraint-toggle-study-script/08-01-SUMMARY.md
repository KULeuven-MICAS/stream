---
phase: 08-constraint-toggle-study-script
plan: "01"
subsystem: tests
tags: [study-script, constraint-toggles, matplotlib, dse]
dependency_graph:
  requires: []
  provides: [tests/study_constraint_toggles.py]
  affects: []
tech_stack:
  added: []
  patterns: [study-script, itertools.product, matplotlib-Agg, horizontal-bar-chart, heatmap-table]
key_files:
  created:
    - tests/study_constraint_toggles.py
  modified: []
decisions:
  - "Plot functions implemented alongside core in single file write — both tasks committed together"
  - "matplotlib.use('Agg') set at module level for headless-safe operation"
  - "Horizontal bar charts (not vertical) chosen for 16-row readability"
  - "Heatmap uses ax.table with colored cells (green=ON, red=OFF) rather than imshow — simpler and consistent with study_swiglu_backends.py pattern"
  - "Baseline row is index 0 (all-enabled, first itertools.product result)"
metrics:
  duration_s: 152
  completed_date: "2026-05-08"
  tasks_completed: 2
  files_modified: 1
---

# Phase 08 Plan 01: Constraint Toggle Study Script Summary

**One-liner:** Standalone script enumerating all 16 ConstraintSelection combinations via itertools.product, printing delta-table and saving 3 matplotlib plots (objective bars, solve time bars, ON/OFF heatmap).

## What Was Built

`tests/study_constraint_toggles.py` (743 lines) — a standalone runnable study script following the pattern of `tests/verify_backends.py` and `tests/study_swiglu_backends.py`.

### Key components

**_all_combinations():** Generates all 16 (label, ConstraintSelection) tuples using `itertools.product([True, False], repeat=4)`. The first tuple is always the all-enabled baseline. Labels are human-readable comma-joined group names or "None (all disabled)".

**Pipeline runners:** `_run_gemm_pipeline` and `_run_swiglu_pipeline` follow verify_backends.py exactly, adding `constraint_selection=constraint_selection` kwarg to `optimize_allocation_co()`.

**run_study():** Iterates all 16 combinations, prints `[1/16] Running "..." ...` progress, returns list of result dicts with keys: label, constraint_selection (dict), status, objective, solve_time_s.

**_print_results_table():** Formatted table with columns #, Constraints Enabled (width 60), Objective (comma-formatted), Delta % (signed, relative to baseline), Solve Time (s).

**3 matplotlib plots (all using Agg backend):**
- `constraint_study_objective.png` — horizontal bar chart, green baseline / blue others / red failed
- `constraint_study_solve_time.png` — same layout for solve time
- `constraint_study_heatmap.png` — ax.table with ON (green #C6EFCE) / OFF (red #FFC7CE) cells + Objective + Delta % column (green/yellow/red gradient by magnitude)

**Argparse:** `--workload gemm|swiglu`, full GEMM/SwiGLU dimension params, `--backend`, `--output-dir`, `--output-yaml`.

## Deviations from Plan

### Auto-fixed Issues

None. Plan executed exactly as written.

### Implementation Notes

Both Task 1 (core structure) and Task 2 (plots) were implemented in a single file write — the entire 743-line script was created atomically since it is a single output file. The Task 1 commit (110b050) contains the complete implementation.

## Known Stubs

None. The script is fully wired: `_run_single` calls `optimize_allocation_co` with `constraint_selection=cs`, `run_study` iterates all 16, `plot_results` calls all three plot functions.

## Verification Results

- `_all_combinations()` returns exactly 16 tuples: PASSED
- "None (all disabled)" label present: PASSED
- All-enabled label present: PASSED
- argparse `--workload gemm` parses correctly: PASSED
- 3 PNG files generated from mock data (each >1KB): PASSED
- `plt.savefig` count == 3: PASSED
- `constraint_selection=` kwarg wired through pipeline: PASSED
- Line count 743 >= min_lines 350: PASSED

## Self-Check: PASSED

Files created:
- tests/study_constraint_toggles.py: FOUND

Commits:
- 110b050 feat(08-01): create study_constraint_toggles.py — core structure, 16-combo enumeration, table output: FOUND
