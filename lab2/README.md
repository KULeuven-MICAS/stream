# Lab 2: Partitioning a layer across cores

## Objective
The goal of this lab is to get more familiar with additional fields that can be specified in the `mapping` input to partition layers across cores. Such partitioning can enable lower latency if the cores would otherwise remain unused.

**Note**: Keep in mind that this is not layer fusion, as a single layer at a time is parallelized across the cores. Thus, we still refer to it as layer-by-layer processing.

## Setup
1. Ensure you have installed the requirements in `requirements.txt`.
2. Make sure you are in the base directory, as `lab2/main.py` automatically inserts the working directory into PATH which is needed for the Stream imports.

## Inputs
There are three main inputs defined in the `inputs/` folder:
1. **Workload**: _[same as lab1]_ Four back-to-back convolutional layers. The layer names are `Layer0`, `Layer`, etc. You can use [Netron](https://netron.app) to visualize the model.
2. **Hardware**: _[same as lab1]_ A sample accelerator is encoded in `hda_bus.yaml`. There are three computing cores, `accelerator1.yaml` through `accelerator3.yaml`. These cores are defined using the ZigZag architecture definition (more information on the ZigZag architecture representation can be found [here](https://kuleuven-micas.github.io/zigzag/hardware.html)). Additionally, there is an `offchip_core` field which specifies the description of the offchip core. This offchip core also uses the ZigZag description, of which only the memory information is used (as no computations can be allocated to the offchip core). The cores are interconnected using the `core_connectivity` field of the HDA description. This specifies on each line a communication channel between two or more cores. A link is automatically added from the offchip core to the different compute cores.
3. **Mapping**: The `Layer0`-`Layer3` layers are all allocated identically, using a single `default` key in the mapping input. This `default` entry is always used as a fallback in case the layer name or the layer type is not present in the mapping input. They are all parallelized across all three cores. Additionally, the mapping specifies how we want these layers to be tiled across the cores through the `inter_core_tiling` field. We tile them in the `K` (output channel) dimension with a factor of 3. The specified number is the number of splits that will be generated, with each split containing a smaller portion of output channels. This number should thus match with the length of `core_allocation`.

## Running the Experiment
Run the main file:
``` bash
python lab2/main.py
```

The mode of execution is still layer-by-layer, as one layer finished completely before we move on to the next layer.

The experiment still consists of a single COALA evaluation, as the allocation is fixed and there are no degrees of freedom for the genetic algorithm to explore.

## Outputs _[same as lab1]_
The results of the experiment will be saved in the `outputs/` folder under the created experiment id.


- `cost_lut.png` visualizes the ZigZag layer-core costs. Because the core allocations are fixed here, the cost of each layer is only extracted for the core it's allocated to.
- `schedule.html` is a Plotly-based visualization of the obtained schedule through time on the different computation and communication resources of the HDA. You can download this and view it in your favourite web browser (Firefox). 
- `schedule.json` is a conversion of the schedule to json format for usage with the [Perfetto](https://ui.perfetto.dev/) visualization tool. This visualization scales better for very large workload graphs with a lot of nodes. Note that the colors here are not the same as in the Plotly visualization, as we don't have control over this.
- `memory.png` visualizes the memory usage on the different Core memories through time. This can help to identify memories that could benefit from increased capacity.

## Questions & Answers

- Take a look inside the generated output `schedule.html`. Are the latencies of the layer parts on each core balanced?
    > <details>
    > <summary>Answer</summary>
    >     
    > The latencies are not matched, as the cores have different dataflows which perform differently for the same layer part. Stream currently only supports equal partitioning through the mapping file input. Stream's internals do support unequal partitioning, though. If you're interested in unequal partitioning, take a look at the `TiledWorkloadGenerationStage` in `stream/stages/generation/tiled_workload_generation.py`.
    >   
    > </details>

- How do the inputs of the first layer get to the cores?
    > <details>
    > <summary>Answer</summary>
    >     
    > The inputs of the first layer are transferred using the offchip CommunicationLink named `Core(3) <-> Any` in the Plotly visualization. However, since the layer parts only differ in K, they require the same input activations. The COALA scheduler detects this behavior and 'broadcasts' this input tensor to all cores by reusing a previous task if it regards the same tensor.
    >   
    > </details>

- Why are there no more 'Block' tasks on the communication links?
    > <details>
    > <summary>Answer</summary>
    >     
    > The required and generated data for each part of the layers is smaller, and thus can fit completely within the core's memories. As such, the data is transferred to/from cores directly without requiring more tiling to fit in the core's memories.
    >   
    > </details>
