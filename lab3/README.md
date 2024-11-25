# Lab 3: Layer Fusion

## Objective
The goal of this lab is to understand how Stream can be used to partition layers in a different manner, such that output of one layer can be reused by subsequent layers before completely finishing. This is possible for many machine learning models, because of e.g. the limited receptive field of convolutions or elementwise operations directly following another operation. We call this `layer fusion` or `depth-first` processing. This can enable higher on-chip data reuse because the smaller parts don't require offloading to the offchip memory.

Similarly to the previous lab, this will be enforced through an additional field in the `mapping` input. Where as the previous lab defined a dimension and splits to partition a layer across multiple cores, here we will define how to partition a layer within a core to enable fusion.

## Setup
1. Ensure you have installed the requirements in `requirements.txt`.
2. Make sure you are in the base directory, as `lab3/main.py` automatically inserts the working directory into PATH which is needed for the Stream imports.

## Inputs
There are three main inputs defined in the `inputs/` folder:
1. **Workload**: _[same as lab1/2]_ Four back-to-back convolutional layers. The layer names are `Layer0`, `Layer`, etc. You can use [Netron](https://netron.app) to visualize the model.
2. **Hardware**: _[same as lab1/2]_ A sample accelerator is encoded in `hda_bus.yaml`. There are three computing cores, `accelerator1.yaml` through `accelerator3.yaml`. These cores are defined using the ZigZag architecture definition (more information on the ZigZag architecture representation can be found [here](https://kuleuven-micas.github.io/zigzag/hardware.html)). Additionally, there is an `offchip_core` field which specifies the description of the offchip core. This offchip core also uses the ZigZag description, of which only the memory information is used (as no computations can be allocated to the offchip core). The cores are interconnected using the `core_connectivity` field of the HDA description. This specifies on each line a communication channel between two or more cores. A link is automatically added from the offchip core to the different compute cores.
3. **Mapping**: The `Layer0`-`Layer3` layers are all allocated identically, using a single `default` key in the mapping input. The `core_allocation_is_fixed` field is set to `False`, which means each layer can be allocated flexibly to all cores listed in `core_allocation`. Each layer will be allocated to a single core as there is no `inter_core_tiling`, and will be partitioned within a core according to the `intra_core_tiling` field. To enable fusion of subsequent layers, the layers are partitioned in the `OY` layer dimension (output activation rows). The `all` field is used to partition flexibly for changing dimension sizes: it partitions into a number of parts equal to the dimension size. Thus, each tile is computing a single output row.


## Running the Experiment
Run the main file:
``` bash
python lab3/main.py
```

**Note**: Notice the `mode` flag which is now set to `fused` to signal to the different stages that the `intra_core_tiling` should be used. 

As the `core_allocation_is_fixed` is `False`, the genetic algorithm will now optimize the workload allocation by trying different layer-core combinations. Each combination is evaluated by COALA, and the genetic algorithm will crossover, mutat, etc. different individuals to create new offspring for the next generation.

## Outputs _[same as lab1/2]_
The results of the experiment will be saved in the `outputs/` folder under the created experiment id.


- `cost_lut.png` visualizes the ZigZag layer-core costs. Because the core allocations are fixed here, the cost of each layer is only extracted for the core it's allocated to.
- `schedule.html` is a Plotly-based visualization of the obtained schedule through time on the different computation and communication resources of the HDA. You can download this and view it in your favourite web browser (Firefox). 
- `schedule.json` is a conversion of the schedule to json format for usage with the [Perfetto](https://ui.perfetto.dev/) visualization tool. This visualization scales better for very large workload graphs with a lot of nodes. Note that the colors here are not the same as in the Plotly visualization, as we don't have control over this.
- `memory.png` visualizes the memory usage on the different Core memories through time. This can help to identify memories that could benefit from increased capacity.

## Questions & Answers

- Take a look inside the generated output `schedule.html`. Has the latency improved compared to the previous layer-by-layer lab?
    > <details>
    > <summary>Answer</summary>
    >     
    > The latency has improved to `2.98e4`compared to `1.15e5` for `lab2`. This is due to the higher overlap of execution of different layers compared to the equal splits of `lab2` where the dataflow of `accelerator1` caused idling on the other cores.
    > 
    > **Note**: The obtained latency may vary because a genetic algorithm is a non-deterministic optimization algorithm.
    >   
    > </details>

- Do you see the fusion pattern? How many output rows of the first layer are required to start execution of the first output row of the second layer?
    > <details>
    > <summary>Answer</summary>
    >     
    > There are two parts needed of the first layer (`Layer0`), i.e. the first two output rows, before the first output row of `Layer1` can be computed. This is because the convolutional layer has a 3x3 filter with padding of 1.
    >   
    > </details>

- How many output rows of the first layer are required before the first part of the last layer (`Layer3`) can start?
    > <details>
    > <summary>Answer</summary>
    >     
    > When hovering over a Task in the Plotly visualization, or looking the `Tensors` field of the Perfetto visualization the following can be observed: 
    > 
    > - The first part of `Layer3`(sub-id 0) requires first two parts of `Layer2`: sub-ids 0 and 1.
    > 
    > - The second part of `Layer2` (sub-id 1) requires the first three parts of `Layer1`: sub-ids 0, 1 and 2.
    > 
    > - The third part of `Layer1` (sub-id 2) requires the second, third and fourth part of `Layer0`: sub-ids 1, 2 and 3.
    > 
    > There are thus 4 output rows of the first layer needed before the first part of the last layer can start. This can be verified against the convolutional operators: each layer has a 3x3 filter with a stride and paddding of 1.
    >   
    > </details>
