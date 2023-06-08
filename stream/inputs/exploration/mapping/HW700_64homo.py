mapping = {
    "default": {
        "core_allocation": list(range(64)),
    },
    "Conv": {
        "core_allocation": list(range(64)),
    },
    "Gemm": {
        "core_allocation": list(range(64)),
    },
    "Pool": {
        "core_allocation": 64,
    },
    "MaxPool": {
        "core_allocation": 64,
    },
    "AveragePool": {
        "core_allocation": 64,
    },
    "GlobalAveragePool": {
        "core_allocation": 64,
    },
    "Add": {
        "core_allocation": 65,
    },
}