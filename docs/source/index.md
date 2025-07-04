# Welcome to Stream

**Stream** is a design space exploration (DSE) framework for mapping deep neural networks (DNNs) onto **multi-core heterogeneous dataflow accelerators**. It supports a wide spectrum of architectural designs and scheduling granularities—from traditional layer-by-layer execution to advanced **layer-fused processing**—enabling scalable, efficient deployment of modern DNNs.

Stream builds upon the [ZigZag framework](https://zigzag-project.github.io/zigzag/) but significantly extends its capabilities to multi-core and fine-grained scheduling contexts.

---

## ✨ Key Features

- **Layer Fusion Support**  
  Enables splitting and scheduling *parts of layers* across multiple cores for higher utilization and lower memory access costs.

- **Heterogeneous Multi-Core Scheduling**  
  Models realistic accelerator architectures including cores with different compute/memory capabilities and interconnects.

- **Memory & Communication-Aware Analysis (COALA)**  
  Stream integrates COALA: a validated latency and energy model that captures data reuse, communication overhead, and memory hierarchies.

- **Workload Allocation via Constraint Optimization (WACO)**  
  Stream includes a built-in engine that explores valid allocations across cores using constraint-based optimization.

- **Validated Against Real Hardware**  
  Performance models and predictions are benchmarked against three state-of-the-art accelerator designs.

- **Modular & Extensible**  
  Stages of the mapping process are customizable, enabling easy experimentation and research integration.

---

## 🚀 Get Started

1. **Clone and install requirements**

   ```bash
   git clone https://github.com/KULeuven-MICAS/stream.git
   cd stream
   pip install -r requirements.txt
   ```

2. **Try the tutorial**

   ```bash
   git checkout tutorial
   python lab1/main.py
   ```

More step-by-step setup help can be found in the [Getting Started](getting-started.md) and [Installation](installation.md) pages.

---

## 📚 Publications

The framework and methodology are described in:

> A. Symons, L. Mei, S. Colleman, P. Houshmand, S. Karl and M. Verhelst,  
> *“Stream: Design Space Exploration of Layer-Fused DNNs on Heterogeneous Dataflow Accelerators”*,  
> IEEE Transactions on Computers, 2025.  
> [📄 Read our paper](https://ieeexplore.ieee.org/abstract/document/10713407)

---

Stream enables researchers and developers to design, evaluate, and optimize novel DNN hardware accelerators — particularly for **latency-sensitive, power-constrained edge applications**.

Happy exploring!
