=======
Mapping
=======

Stream requires two different specification about the possible mapping of a workload to a hardware architecture. These two mapping specification are

1. the **spatial mapping** of each core (defined as the 'dataflow' in :doc:`hardware`)
2. the possible **core allocation** of each layer type (defined in a file in the `mapping folder <https://github.com/KULeuven-MICAS/stream/tree/master/stream/inputs/examples/mapping>`_)

These two specifications will be further explained on this page:

Spatial Mapping
---------------

The spatial mapping describes the spatial parallelization strategy used in a certain core. The spatial mapping has to be specified in the hardware architecture as an attribute to each core (see `explanation here <https://kuleuven-micas.github.io/stream/hardware.html#core>`_ and `example here <https://github.com/KULeuven-MICAS/stream/blob/master/stream/inputs/examples/hardware/cores/Eyeriss_like.py#L198>`_). An example dataflow could look like the following:

.. code-block:: python

    [{"D1": ("K", 16), "D2": ("C", 16)}]

In this example the Operational Array has two dimensions (i.e. D1 and D2). The output channels ("K") are unrolled over D1 and the input channels ("C") are unrolled over D2. Both dimensions have an unrolling factor of 16. This spatial mapping would be similar to the one used in Eyeriss. 

Core Allocation
---------------

Besides the spatial mapping, the user has to provide information about which layer type can be exectued on which core in the hardware architecture. An `example core allocation <https://github.com/KULeuven-MICAS/stream/blob/master/stream/inputs/examples/mapping/eyeriss_like_quad_core_pooling_simd_offchip.py>`_ could look like the following

.. code-block:: python

    mapping = {
        "default": {
            "core_allocation": [0, 1, 2, 3],
        },
        "pooling": {
            "core_allocation": 4,
        },
        "simd": {
            "core_allocation": 5,
        }
    }

In this example all regular layers can be executed on core 0 to 3 while pooling layers have to be executed on core 4 and layers with SIMD operations (e.g. addition) have to be executed on core 5. The available layer types are all the ones introduced in :doc:`workload`.