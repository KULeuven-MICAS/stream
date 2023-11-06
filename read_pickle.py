

import pandas as pd
import pickle
# pickleFile = open("/proj/rdi/staff/gagandee/dse/stream_aie/outputs/saved_cn_hw_cost-aie_col-bottleneck-hintloop_oy_all.pickle","rb")

# obj = pd.read_pickle(pickleFile)
# print (obj)
pickle_filepath="/proj/rdi/staff/gagandee/dse/stream_aie/outputs/saved_cn_hw_cost-aie_col-one_bottleneck_with_bias-hintloop_oy_all-layer_splitting.pickle"

with open(pickle_filepath, "rb") as handle:
        node_hw_performances = pickle.load(handle)

# first_key = next(iter(node_hw_performances))
# node_hw_performances = {first_key: node_hw_performances[first_key]}
print(node_hw_performances)
node_labels = []
cores = []
min_latency_per_node = {}
min_energy_per_node = {}
for node, hw_performances in node_hw_performances.items():
    node_labels.append(f"L{node.id[0]}\nN{node.id[1]}")
    min_latency_per_node[node] = float("inf")
    min_energy_per_node[node] = float("inf")
    for core, cme in hw_performances.items():
        
        if core not in cores:
            cores.append(core)
        if cme.latency_total2 < min_latency_per_node[node]:
            min_latency_per_node[node] = cme.latency_total2
            min_energy_per_node[node] = cme.energy_total
    print(
        node.type,
        node.id,
        {k: cme.temporal_mapping for k, cme in hw_performances.items()},
    )