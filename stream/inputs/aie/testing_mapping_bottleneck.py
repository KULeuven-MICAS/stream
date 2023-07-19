mapping = {
    "Conv1": {
        "core_allocation": [0],
    },
    # "Conv2": {
    #     "core_allocation": [1],
    # },
    "Conv3": {
        "core_allocation": [3],
    },

    "Add": {
        "core_allocation": [0],
    },
    # "Gemm": {
    #     "core_allocation": [0, 1, 2, 3],
    # },
    # "Pool": {
    #     "core_allocation": [4],
    # },
    # "MaxPool": {
    #     "core_allocation": [4],
    # },
    # "AveragePool": {
    #     "core_allocation": [4],
    # },
    # "GlobalAveragePool": {
    #     "core_allocation": [4],
    # },
    
    "default": {
        "core_allocation": [ 1, 2],
    },
}