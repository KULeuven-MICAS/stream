mapping = {
    "layer_on_core_0": {
        "core_allocation": [0],  # First conv layer mapped to digital core
    },
    "Conv": {
        "core_allocation": [1],  # Other conv layers mapped to analog core
    },
    "MaxPool": {
        "core_allocation": [2],
    },
    "Add": {
        "core_allocation": [3],
    },
    "default": {
        "core_allocation": [0],
    },
}
