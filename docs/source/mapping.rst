=======
Mapping
=======

Stream requires two different specification about the possible mapping of a workload to a hardware architecture. These two mapping specification are

1. the **spatial mapping** of each core (defined as the 'dataflow' in :doc:`hardware`)
2. the possible **core allocation** of each layer type (defined in a file in the `mapping folder <https://github.com/KULeuven-MICAS/stream/tree/master/stream/inputs/examples/mapping>`_)

These two specifications will be further explained on this page:

Spatial Mapping
===============

The spatial mapping describes the spatial parallelization strategy used in a certain core. The spatial mapping has to be specified in the hardware architecture as an attribute to each core (see explanation `here <https://kuleuven-micas.github.io/stream/hardware.html#core>`_ and example `here <https://github.com/KULeuven-MICAS/stream/blob/master/stream/inputs/examples/hardware/cores/Eyeriss_like.py#L198>`_). An example dataflow could look like:

.. code-block:: python

    [{"D1": ("K", 16), "D2": ("C", 16)}]

In this example the Operational Array has two dimensions (i.e. D1 and D2). The output channels ("K") are unrolled over D1 and the input channels ("C") are unrolled over D2. Both dimensions have an unrolling factor of 16. 

Core Allocation
===============

Besides the spatial mapping, the user has to provide information about which layer type can be exectued on which core in the hardware architecture. An example core allocation for the architecture `here <https://github.com/KULeuven-MICAS/stream/blob/master/stream/inputs/examples/mapping/tpu_like_quad_core.py>`_ could look like:

.. code-block:: python

    mapping = {
        "/conv1/Conv": {
            "core_allocation": 2  # or (2,)
        },
        "/conv2/Conv": {
            "core_allocation": (0, 1, 2, 3)
        },
        "pooling": {
            "core_allocation": 4,
        },
        "simd": {
            "core_allocation": 5,
        }
        "default": {
            "core_allocation": [0, 1, 2, 3],
        },
    }

In this example:

1. The layer with name "/conv1/Conv" will have a fixed core allocation onto core 2.
2. The layer with name "/conv2/Conv" will have a fixed core allocation for its groups (see `IntraCoreMappingStage` in :doc:`stages` for more information regarding groups).
3. All layers of type "pooling" will be allocated to core 4.
4. All layers of type "simd" (case insensitive) will be allocated to core 5.
5. All other layers can be allocated to cores 0, through 3 by default.

When determining the possible core allocations for a node, the name is checked first, then the type, then the default is used as a last resort.
The available layer types are all the ones introduced in :doc:`workload`.

Saving an SCME's allocation
---------------------------

When you have run Stream for optimizing a layer-core allocation, it can be interesting to save the obtained allocation for future use as a fixed mapping, without having to re-run the genetic algorithm.
The obtained allocation can be saved to a python file through use of the `save_core_allocation <TODO>`_ function. The code below demonstrates its use:

.. code-block:: python

    from pprint import pprint
    from stream.utils import load_scme, save_core_allocation

    scme_path = 'my/saved.scme'
    scme = load_scme(scme_path)
    d = save_core_allocation(scme.workload, "my/fixed/mapping.py")
    pprint(d)
