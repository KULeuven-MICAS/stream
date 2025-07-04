# Stream [Documentation](https://kuleuven-micas.github.io/stream/)
Stream is a HW architecture-mapping design space exploration (DSE) framework for multi-core deep learning accelerators. The mapping can be explored at different granularities, ranging from classical layer-by-layer processing to fine-grained layer-fused processing. Stream builds on top of the ZigZag DSE framework, found [here](https://zigzag-project.github.io/zigzag/). 

More information with respect to the capabilities of Stream can be found in the following paper:

[A. Symons, L. Mei, S. Colleman, P. Houshmand, S. Karl and M. Verhelst, “Stream: Design Space Exploration of Layer-Fused DNNs on Heterogeneous Dataflow Accelerators”.](https://ieeexplore.ieee.org/abstract/document/10713407)


## Install required packages:
```bash
pip install -r requirements.txt
```

## The first run
```bash
git checkout tutorial
python lab1/main.py
```

## Documentation
You can find extensive documentation of Stream [here](https://kuleuven-micas.github.io/stream/).
