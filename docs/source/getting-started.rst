===============
Getting Started
===============

Stream allows you to run a design space exploration for both, traditional layer-by-layer processing as well as layer-fused processing of DNN workloads. The framework can be used to explore the performace of a workload on multi-core and single-core architectures.

In a first run, we are going to run ResNet-18 on quad-core architecture similar to a TPU like hardware [1]. We provide an `onnx <https://onnx.ai/>`_ model of this network in ``stream/inputs/examples/workload/resnet18.onnx`` and the HW architecture in ``stream/inputs/examples/hardware/TPU_like_quad_core.py``.

The onnx model has been shape inferred, which means that besied the input and output tensor shapes, all intermediate tensor shapes have been inferred, which is information required by Stream. 

.. warning::
    ZigZag requires an inferred onnx model, as it needs to know the shapes of all intermediate tensors to correctly infer the layer shapes. You can find more information on how to infer an onnx model `here <https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model>`_.

Besides the workload and HW architecture, a mapping file must be provided which, as the name suggests, provides information about which layer can be mapped to which core in the hardware architecture. The mapping is provided in ``stream/inputs/examples/mapping/tpu_like_quad_core.py``.

The framework is generally ran through a main file which parses the provided inputs and contains the program flow through the stages defined in the main file.

.. note::

    You can find more information in the :doc:`stages` document.

Layer-by-layer processing of workload
=====================================

Now, we would like to run the previously introduced workload in a layer-by-layer fashion, which means that one layer is exectued at once on a certain core and the next layer can only start as soon as all previous layers are completely done.

For this we have to exectue

.. code:: sh

    python main_stream.py

which parses the given workload, hw architecture and the corresponding mapping. Stream will now evaluate how efficently the workload can be executed on the given hardware with a layer-by-layer approach.

Layer-fused processing of workload
==================================

In a second run, we would like to run the same workload on the same HW with the same mapping. The difference will be that a layer-fused approach is used instead of a layer-by-layer approach.

For this we have to execute

.. code:: sh

    python main_stream_layer_splitting.py

which starts another run of Stream. Now the given inputs are processed in a layer-fused approach which means that each layer is split in several smaller parts. 

Analyzing results
=================

During the run of each experiement, Streams saves the results in the ``outputs`` folder based on the paths provided in the ``main_stream.py`` and ``main_stream_layer_splitting.py`` files. In this folder, there will be four ``.png`` files. Two of them show the schedule of workload's layer on the different cores of the hw architecture (one file for the layer-by-layer approach and one file for the layer-fused approach). Besides this, the other two ``.png`` files show the memory utilization of the different cores in the system for the two different experiements. More explanation about the results can be found on the :doc:`outputs` page.

[1] N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa,
S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P.-l. Cantin,
C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V.
Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho,
D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski,
A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy,
J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin,
G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan,
R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick,
N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani,
C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing,
M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan,
R. Walter, W. Wang, E. Wilcox, and D. H. Yoon, “In-datacenter
performance analysis of a tensor processing unit,” SIGARCH Comput.
Archit. News, vol. 45, no. 2, p. 1–12, jun 2017. 