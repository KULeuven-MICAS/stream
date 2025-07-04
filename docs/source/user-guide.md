# User Guide

The Stream framework consists of five major building blocks. The documents listed below explain each component in detail.

If you're looking for implementation details or want to understand the design philosophy behind Stream, refer to the [publications](publications.md) page or dive into the [code](https://github.com/KULeuven-MICAS/stream) on GitHub.

![User Guide Overview](images/user-guide-overview.jpg)

## Components

- [Workload](workload.md): Learn how Stream interprets deep learning workloads and represents them as computation graphs.
- [Hardware](hardware.md): Explore how hardware architectures are modeled and parsed into the framework.
- [Mapping](mapping.md): Understand the process of mapping workloads to multi-core accelerators, including both intra- and inter-core strategies.
- [Stages](stages.md): A modular pipeline structure lets you configure or extend Stream’s behavior with custom stages.
- [Outputs](outputs.md): Discover what results Stream produces—visualizations, performance metrics, and more.
