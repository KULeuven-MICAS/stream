# Stages

This document explains the concept of stages within the **Stream** framework. It details the different implemented stages and explains how to create your own.

---

## Introduction

Stages in Stream allow modular customization of the framework’s behavior. The sequence of stages defines what the framework will execute. These are configured in the `MainStage`. Example:

```python
mainstage = MainStage(
    [
        AcceleratorParserStage,
        StreamONNXModelParserStage,
        LayerSplittingStage,
        StreamONNXModelParserStage,
        GenerateCNWorkloadHybridStage,
        IntraCoreMappingStage,
        InterCoreMappingStage,
    ],
    accelerator=accelerator,
    workload_path=workload_path,
    mapping_path=mapping_path,
    loma_lpf_limit=6,
    nb_ga_individuals=32,
    nb_ga_generations=100,
    cost_lut_path=cost_lut_path,
    plot_hof=True,
    plot_file_name=plot_file_name,
    plot_full_schedule=plot_full_schedule,
    plot_data_transfer=plot_data_transfer,
    cn_define_mode=CN_define_mode,
    hint_loops=hint_loops,
    scheduler_candidate_selection="memory",
    operands_to_prefetch=[],
    split_onnx_model_path=split_onnx_model_path,
    split_W_double_buffered=split_W_double_buffered,
)

scme, _ = mainstage.run()
scme = scme[0]
```

---

## Implemented Stages

See the [stream/stages/](https://github.com/KULeuven-MICAS/stream/tree/master/stream/stages) folder for up-to-date source definitions.

### [CustomSpatialMappingGeneratorStage](https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/CustomSpatialMappingGeneratorStage.py#L23)

Finds spatial mappings given an accelerator, core allocation, and interconnection pattern. Uses the innermost memory levels to determine dataflow.

---

### [GenerateCNWorkloadHybridStage](https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/GenerateCNWorkloadHybridStage.py#L29)

Transforms a layer-by-layer workload into a fine-grained CN workload graph. Configurable via `cn_define_mode` and `hint_loops`.

Modes include:

1. `hint_loops` defines outer-cn loops for splitting.
2. `hint_loops` defines inner-cn loops; all others become outer-cn.
3. `hint_loops` is a list-of-lists; `layer_cutoffs` defines to which layers each applies.
4. `hint_loops` defines outer-cn; `split_W_percentage` limits constant operand memory footprint. Layers exceeding the limit are split in K.

---

### [InterCoreMappingStage](https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/InterCoreMappingStage.py#L17)

Performs inter-core mapping using a genetic algorithm. Starts from CMEs collected during intra-core mapping.

---

### [IntraCoreMappingStage](https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/IntraCoreMappingStage.py#L22)

Finds optimal CMEs per node-core allocation. Groups nodes based on `loop_ranges` differences in relevant dimensions (e.g., `K` for convolution).

---

### [ONNXModelParserStage](https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/ModelParserStage.py#L11)

Parses a workload file into a `NetworkX` graph. Converts ONNX into Stream’s internal representation.

---

You can also reuse [ZigZag’s implemented stages](https://kuleuven-micas.github.io/zigzag/stages.html#implemented-stages) within Stream.

---

## Creating a Custom Stage

To create a custom stage (e.g., optimize for something other than energy), copy an existing stage and modify it as needed. Make sure to:

- Inherit from the abstract `Stage` class
- Initialize your substage at the beginning of your callables list:
  ```python
  substages = [YourNextStage(...), ...]
  stage = YourStage(substages, **kwargs)
  ```
- Iterate over `for cme, extra_info in substage.run():` to yield results
- If you're reducing (like `MinimalLatencyStage`), yield **outside** the loop

---

Creating custom stages enables targeted optimization and full control over the pipeline logic.
