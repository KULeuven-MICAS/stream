---
phase: 19-ga-removal
plan: 01
subsystem: cleanup
tags: [ga-removal, dead-code, deap, api]
dependency_graph:
  requires: []
  provides: [clean-api-no-ga, no-deap-dependency]
  affects: [stream/api.py, pyproject.toml, main_aie_co.py, main_aie_codegen_conv2d.py, main_stream_co.py, stream/visualization/perfetto.py, CLAUDE.md]
tech_stack:
  added: []
  patterns: [atomic-deletion-with-import-removal]
key_files:
  created: []
  modified:
    - stream/api.py
    - pyproject.toml
    - main_aie_co.py
    - main_aie_codegen_conv2d.py
    - main_stream_co.py
    - stream/visualization/perfetto.py
    - CLAUDE.md
  deleted:
    - main_aie_ga.py
    - main_stream_ga.py
    - stream/stages/allocation/genetic_algorithm_allocation.py
    - stream/opt/allocation/genetic_algorithm/__init__.py
    - stream/opt/allocation/genetic_algorithm/fitness_evaluator.py
    - stream/opt/allocation/genetic_algorithm/genetic_algorithm.py
    - stream/opt/allocation/genetic_algorithm/statistics_evaluator.py
    - stream/inputs/examples/mapping/tpu_like_quad_core_ga.yaml
decisions:
  - "Fixed optimize_allocation_co return type annotation to StageContext (was incorrectly annotated StreamCostModelEvaluation — the function returns ctx, a StageContext)"
  - "Removed pickle_load import from perfetto.py after __main__ block deletion (only __main__ used it)"
metrics:
  duration: 7 minutes
  completed_date: "2026-05-11"
  tasks_completed: 2
  files_changed: 15
  files_deleted: 8
---

# Phase 19 Plan 01: GA Removal Summary

**One-liner:** Deleted all 8 genetic algorithm allocation files and removed GA imports, dead variables, and DEAP dependency — stream.api now importable without DEAP.

## Tasks Completed

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Delete GA files and remove GA code from api.py + pyproject.toml | e3d15c7 | 8 GA files deleted, GA imports removed from api.py, deap removed from pyproject.toml |
| 2 | Clean dead variables from scripts, remove perfetto __main__ block, update CLAUDE.md | 409a43b | Dead mode/layer_stacks/nb_ga_* vars cleaned, CostModelEvaluationLUT replaced with CoreCostLUT, perfetto __main__ block removed, CLAUDE.md updated |

## Outcome

- `import stream.api` succeeds without DEAP installed
- Zero GA files remain in repository (8 deleted)
- `deap` absent from pyproject.toml
- All 176 tests pass (1 pre-existing failure in test_core_cost_lut_caching.py unrelated to GA removal — mapping format issue targeted in Phase 20)
- No dead mode/layer_stacks/nb_ga_* variables in any remaining script
- CLAUDE.md contains no GA script references
- ruff check passes on all modified files

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed optimize_allocation_co return type annotation**
- **Found during:** Task 1
- **Issue:** After removing `optimize_allocation_ga`, `StreamCostModelEvaluation` import was dropped, but `optimize_allocation_co` still had `-> StreamCostModelEvaluation:` annotation. The function actually returns a `StageContext` (it returns `ctx`).
- **Fix:** Changed return type annotation from `StreamCostModelEvaluation` to `StageContext`
- **Files modified:** stream/api.py
- **Commit:** e3d15c7

**2. [Rule 1 - Bug] Removed unused pickle_load import from perfetto.py**
- **Found during:** Task 2, Step 4
- **Issue:** After removing the `__main__` block from perfetto.py, the `from zigzag.utils import pickle_load` import became unused
- **Fix:** Removed the unused import
- **Files modified:** stream/visualization/perfetto.py
- **Commit:** 409a43b

## Pre-existing Test Failure (Out of Scope)

`tests/test_core_cost_lut_caching.py::test_core_cost_lut_caches_and_loads` fails with a mapping validation error. Verified via git stash that this failure predates all changes in this plan. Root cause: mapping format mismatch (`inter_core_tiling` dict format vs. expected type). This is the Phase 20 target (fix mapping format).

## Self-Check: PASSED

Files verified:
- FOUND: stream/api.py (modified, GA-free)
- FOUND: pyproject.toml (modified, deap-free)
- FOUND: main_aie_co.py (modified, dead-vars-free)
- FOUND: main_aie_codegen_conv2d.py (modified, CoreCostLUT used)
- FOUND: main_stream_co.py (modified, dead-vars-free)
- FOUND: stream/visualization/perfetto.py (modified, no __main__ block)
- FOUND: CLAUDE.md (modified, no GA references)
- MISSING (as expected): main_aie_ga.py, main_stream_ga.py, stream/stages/allocation/genetic_algorithm_allocation.py, stream/opt/allocation/genetic_algorithm/

Commits verified:
- FOUND: e3d15c7 (Task 1)
- FOUND: 409a43b (Task 2)
