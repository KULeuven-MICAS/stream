======
Stages
======

This document explains the concept of stages within the Stream framework. It details the different implemented stages and explains how to create your own.

Introduction
============

Stages within Stream are used to modularly and easily adapt the functionality of the framework. The different stages and their sequence of execution determine the goal of running the framework. The sequence of stages the framework will run through are defined in the main file. An example as follows:

.. code-block:: python

    mainstage = MainStage(
        [  # Initializes the MainStage as entry point
            AcceleratorParserStage,  # Parses the accelerator
            StreamONNXModelParserStage,  # Parses the ONNX Model into the workload
            LayerSplittingStage,  # Split the workload
            StreamONNXModelParserStage,  # Parses the potentially split ONNX model into the workload
            GenerateCNWorkloadHybridStage,  # Generate fine-grained CN workload graph
            IntraCoreMappingStage,  # Find the optimal CME for each valid layer-core assignment
            InterCoreMappingStage,  # Find the optimal layer-core assignment for the entire workload
        ],
        accelerator=accelerator,  # required by AcceleratorParserStage
        workload_path=workload_path,  # required by ModelParserStage
        mapping_path=mapping_path,  # required by ModelParserStage
        loma_lpf_limit=6,  # required by LomaStage
        nb_ga_individuals=32,  # number of individuals in each genetic algorithm generation
        nb_ga_generations=100,  # number of genetic algorithm generations
        node_hw_performances_path=node_hw_performances_path,  # saved node_hw_performances to skip re-computation
        plot_hof=True,  # Save schedule and memory usage plot of each individual in the Genetic Algorithm hall of fame
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

    # Launch the MainStage
    scme, _ = mainstage.run() # Run the MainStage
    scme = scme[0] # Select one of the returned cost models for later inspection

Implemented stages
==================

This section is still being updated. For a missing description, please look at the stages requirements in `__init__.py <https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/__init__.py>`_ and the stage implementation in the `stages <https://github.com/KULeuven-MICAS/stream/tree/master/stream/classes/stages>`_ folder.

.. _custom-stages-label:

The following stages are implemented in Stream:

`CustomSpatialMappingGeneratorStage <https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/CustomSpatialMappingGeneratorStage.py#L23>`_
-------------------------------------------------------------------------------------------------------------------------------------------------------------

Stage that finds spatial mappings given a accelerator, core allocation, interconnection pattern on the allocated core, layer. The spatial mappings are found using the interconnection pattern present on the core. The inner-most memory level served dimensions is used, as this is how the memories connect to the operational array.


`GenerateCNWorkloadHybridStage <https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/GenerateCNWorkloadHybridStage.py#L29>`_
----------------------------------------------------------------------------------------------------------------------------------------------------

Stage that transforms the layer-by-layer workload into finer CN workload graph.
Multiple modes are applicable through the `cn_define_mode` parameter in conjunction with the `hint_loops` parameter:

1. `hint_loops` specifies the outer-cn loops based on which the layer will be split.
2. `hint_loops` specifies the inner-cn loops. The outer-cn loops are all remaining loops.
3. `hint_loops` specifies a nested list of loops. `layer_cutoffs` specifies until which layer index each list of outer-cn loops is applicable.
4. `hint_loops` specifies the outer-cn loops. `split_W_percentage` specifies the maximal percentage the constant operands may occupy on the respective memories in the cores they can be allocated to. If multiple cores have a different constant operand memory capacity, the capacity is taken to be the smallest. If a layer has a larger footprint, it will be split in terms of output channels by appending the `K` loops to the `hint_loops`.


`InterCoreMappingStage <https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/InterCoreMappingStage.py#L17>`_
----------------------------------------------------------------------------------------------------------------------------------

Stage that finds the best inter-core mapping using a genetic algorithm. From the IntraCoreMappingStage we receive the `node_hw_performances`, containing for each node and its valid core allocations the best CME. We then initialize the genetic algorithm.

`IntraCoreMappingStage <https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/IntraCoreMappingStage.py#L22/>`_
-----------------------------------------------------------------------------------------------------------------------------------

Stage that finds the optimal ZigZag CME for each valid node-core allocation. This is saved to a dictionary which is passed to the subsequent stages.
The `loop_ranges` attribute op each CN determines the unique nodes to be evaluated. If two nodes have a difference in `loop_ranges` in a dimension that is relevant for the constant operands of the node, e.g. the `K` loop in a traditoinal convolutional layer, the node is assigned to a different group which will be allocated separately in the `InterCoreMappingStage`. 


`ONNXModelParserStage <https://github.com/KULeuven-MICAS/stream/blob/master/stream/classes/stages/ModelParserStage.py#L11>`_
----------------------------------------------------------------------------------------------------------------------------

Stage that parses the input workload residing in accelerator_path. The "workload" dict is converted to a NetworkX graph.

Besides these stages, the `implemented stages from the ZigZag framework <https://kuleuven-micas.github.io/zigzag/stages.html#implemented-stages>`_ can be used as well.


Creating your custom stage
==========================

Let's say you are not interested in saving the CME with minimal energy, but want to save based on another metric provided by the CME, or you want to define a new temporal mapping generator stage, you can easily create a custom stage. The easiest way is copying an existing stage class definition, and modifying it according to your intended behaviour. To guarantee correctness, following aspects have to be taken into account when creating a custom stage:

* It must inherit from the abstract ``Stage`` class.
* It must create its ``substage`` as the first element of the list of callables, with the remaining list as its first argument, and ``**kwargs`` as the second argument. These kwargs can be updated to change e.g. the accelerator, spatial mapping, temporal mapping, etc.
* It must iterate over the different ``(CME, extra_info)`` tuples yielded by the ``substage.run()`` call in a for loop.
* If the stage is a reduction (like e.g. the ``MinimalLatencyStage``), its ``yield`` statement must be outside the for loop which iterates over the returned ``(CME, extra_info)`` tuples, where some processing happens inside the for loop.
