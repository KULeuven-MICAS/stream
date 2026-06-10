# Welcome to Stream

**Stream** is a design-space-exploration (DSE) and constraint-optimization framework for **heterogeneous dataflow accelerators** — systems built by combining cores that each have their own dataflow and performance model. **AIE** and **TPU-like** cores are two example core types among others.

Given a neural-network workload (ONNX) and a hardware description (YAML), Stream schedules the workload **layer-fused** across the cores and uses **MILP (Mixed-Integer Linear Programming)** — the *TETRA* constraint optimization — to decide tensor placement and the transfer paths between cores. Stream builds on the [ZigZag framework](https://zigzag-project.github.io/zigzag/) for per-core cost estimation.

---

## ✨ What it does

- **Heterogeneous multi-core modelling** — accelerators are described as a system of cores with different compute/memory capabilities, connected by links and buses.
- **Layer-fused scheduling** — parts of layers can be split and co-scheduled across cores for higher utilization and lower memory traffic.
- **Constraint-optimization allocation (TETRA)** — a MILP `TransferAndTensorAllocator` decides where each tensor lives and how it is routed.
- **Pluggable solvers** — OR-Tools **GSCIP** (default, license-free), OR-Tools **HiGHS**, and **Gurobi** (commercial license), all behind one API.
- **Memory- and communication-aware cost model** — captures data reuse, memory hierarchy, and interconnect cost.
- **Modular pipeline** — the mapping process is a sequence of stages you can configure or extend.
- **Agent-friendly** — a documented public API, typed IR models, and an MCP server let an AI agent drive the framework (see [Using Stream with an AI agent](ai-agents.md)).

---

## 🚀 Get started

```bash
git clone https://github.com/KULeuven-MICAS/stream_aie.git
cd stream_aie
pip install -e .
```

Then run the CO pipeline on a bundled workload, with an auto-generated mapping:

```bash
python scripts/main_stream_co.py \
  --hardware stream/inputs/examples/hardware/tpu_like_quad_core.yaml \
  --workload stream/inputs/testing/workload/2conv_1_8_32_32_16_32_3.onnx
```

See [Installation](installation.md) and [Getting Started](getting-started.md) for details, and the [User Guide](user-guide.md) for the workload, hardware, and mapping input formats.

---

## 📚 Publication

> A. Symons, L. Mei, S. Colleman, P. Houshmand, S. Karl and M. Verhelst,
> *"Stream: Design Space Exploration of Layer-Fused DNNs on Heterogeneous Dataflow Accelerators"*,
> IEEE Transactions on Computers, vol. 74, no. 1, pp. 237–249, Jan. 2025.
> [📄 Read the paper](https://ieeexplore.ieee.org/abstract/document/10713407)

Developed as part of the **TETRA** project at **KU Leuven MICAS**.
