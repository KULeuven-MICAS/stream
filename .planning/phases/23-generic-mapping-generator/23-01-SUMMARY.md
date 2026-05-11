---
phase: 23-generic-mapping-generator
plan: 01
subsystem: mapping-generation
tags: [hardware, mapping, milp, core-selection, operator-types]
dependency_graph:
  requires: []
  provides: [GenericMappingGenerator, operator_types-on-cores]
  affects: [stream/mapping/generic_generator.py, stream/parser/accelerator_factory.py, stream/parser/accelerator_validator.py]
tech_stack:
  added: []
  patterns:
    - Stream-level extension fields stripped before ZigZag namespace validation, re-injected after normalization
    - operator_types priority: specialized cores take precedence over generic compute cores
    - MappingValidator called inline to guard generated dict before YAML write
key_files:
  created:
    - stream/mapping/generic_generator.py
  modified:
    - stream/inputs/examples/hardware/cores/pooling.yaml
    - stream/inputs/examples/hardware/cores/simd.yaml
    - stream/parser/accelerator_factory.py
    - stream/parser/accelerator_validator.py
decisions:
  - Strip operator_types from core_data before ZigZag AcceleratorValidator to avoid unknown-field rejection, re-inject into normalized_core_data after
  - Specialized cores (non-None operator_types) take priority over generic cores (None operator_types) — MaxPool to pooling core, Add to simd core
  - generate_all_groups() returns (paths, sub_workloads) tuple so callers avoid re-calling split_fusion_groups()
  - Intra-core tiling uses full tile size (tile = dim_size) for no temporal splitting — always valid per MappingValidator
metrics:
  duration: 11 minutes
  completed: "2026-05-11T22:41:35Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 4
---

# Phase 23 Plan 01: Generic Mapping Generator Summary

Implemented `GenericMappingGenerator` class and `operator_types` hardware field so any Workload+Accelerator pair produces a MappingValidator-accepted mapping YAML without hand-written YAML.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Add operator_types to hardware YAMLs and propagate through AcceleratorFactory | 3dba660 | pooling.yaml, simd.yaml, accelerator_factory.py, accelerator_validator.py |
| 2 | Implement GenericMappingGenerator class | 2fbb333 | stream/mapping/generic_generator.py |

## What Was Built

**operator_types field (Task 1):**
- `pooling.yaml` gains `operator_types: [MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool]`
- `simd.yaml` gains `operator_types: [Add, Relu]`
- `AcceleratorValidator.validate_single_core` extracts Stream-level extension fields (currently just `operator_types`) before passing core data to ZigZag namespace validators, then re-injects them into normalized_core_data. This avoids ZigZag's `allow_unknown=False` rejection of unknown fields.
- `AcceleratorFactory.create_core` reads `core_data.get("operator_types", None)` and assigns it to `core.operator_types` in both aie2 and zigzag branches. Cores without the field get `None` (accept all ops per D-06).

**GenericMappingGenerator (Task 2):**
- `generate_all_groups()` calls `split_fusion_groups()`, generates one YAML per group, returns `(paths, sub_workloads)` tuple
- `_select_cores_for_node()` implements D-06/D-09/D-10: specialized cores (non-None operator_types) take priority; if none, use all generic cores; offchip/shim excluded
- `_build_mapping_dict()` produces correct nested-list format: `core_allocation: [[id, ...]]`, `inter_core_tiling: [[{"dim": "D{n}", "split": k}]]`
- `_find_split_dim()` finds the largest dimension divisible by n_cores for the inter-core split
- `_build_intra_core_tiling()` uses full tile size (no temporal splitting — always valid)
- All generated dicts validated via `MappingValidator` before YAML write; `ValueError` raised on failure

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ZigZag AcceleratorValidator rejects operator_types as unknown field**
- **Found during:** Task 1 verification
- **Issue:** `ZigZagBaseCoreValidator` uses cerberus with `allow_unknown=False`, so `operator_types` in the YAML caused validation failure and log errors
- **Fix:** Modified `AcceleratorValidator.validate_single_core` to extract `_STREAM_EXTENSION_FIELDS` (currently `operator_types`) from `core_data` before passing to the namespace validator, then re-inject into normalized data after. File modified: `stream/parser/accelerator_validator.py`
- **Commit:** 3dba660

**2. [Rule 1 - Bug] Core selection included generic compute cores even when specialized cores exist**
- **Found during:** Task 2 testing
- **Issue:** Original `_select_cores_for_node` logic returned both generic (operator_types=None) and specialized cores for a given op type. MaxPool was getting all 5 compute cores (4 generic + 1 pooling) instead of just the pooling core.
- **Fix:** Implemented priority logic: collect specialized cores first, use them if any exist; otherwise fall back to generic cores. Ensures MaxPool → pooling core only, Add → simd core only, Conv → 4 generic compute cores.
- **Commit:** 2fbb333 (in the same task)

## Known Stubs

None — the generator produces fully wired mappings. Intra-core tiling uses full tile size (no temporal splitting), which is intentional for Phase 23 scope. Future phases may add memory-aware tile size computation.

## Verification Results

- `python -c "from stream.mapping.generic_generator import GenericMappingGenerator; print('import OK')"` → passes
- `grep "operator_types" stream/inputs/examples/hardware/cores/pooling.yaml` → shows field
- `grep "operator_types" stream/inputs/examples/hardware/cores/simd.yaml` → shows field
- `pytest tests/ --ignore=tests/integration -q` → 181 passed
- `pytest tests/test_co.py tests/test_core_validation.py tests/test_accelerator_ir.py` → 37 passed
- End-to-end: `GenericMappingGenerator` with two-conv workload produces valid mapping, MappingValidator passes
- Core selection verified: Conv→[0,1,2,3], MaxPool→[4], Add→[5]
