from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from stream.classes.cost_model.cost_model import StreamCostModelEvaluation

def bar_plot_stream_cost_model_evaluations_breakdown(scmes: List[StreamCostModelEvaluation], fig_path: str):

    barWidth = 0.1

    list_attributes = [x for x in scmes[0][0].__dict__.keys() if 'energy' in x]
    #list_attributes.remove('input_onloading_link_energy_cost')
    #list_attributes.remove('input_onloading_memory_energy_cost')
    #list_attributes.remove('output_offloading_link_energy_cost')
    #list_attributes.remove('output_offloading_memory_energy_cost')
    #list_attributes.remove('eviction_memory_energy_cost')

    list_corrected = []
    for attribute in list_attributes:
        for removal in ['_energy_cost', 'input_', 'output_']:
            attribute = attribute.replace(removal, '')
        list_corrected.append(attribute.replace('_', '-').replace('energy', 'Tot-Energy').replace('total', 'Tot').replace('computation', 'Comp').replace('memory', 'Mem').replace('onloading', 'On-lo').replace('offloading', 'Off-lo').replace('eviction', 'Evi'))
    breakdown = [[accelerator[0].__getattribute__(list_attributes[idx]) for accelerator in scmes]
                 for idx, _ in enumerate(list_attributes)]

    fig, ax = plt.subplots(figsize=(8,4))

    br = np.arange(len(breakdown[0]))
    for i, _ in enumerate(list_corrected):
        ax.bar(br+barWidth*i, breakdown[i],width=barWidth)

    # Buil the x and y of the plot
    x_bar = []
    for i,_ in enumerate(scmes):
        for j,_ in enumerate(list_corrected):
            x_bar.append(j*barWidth+i)

    y_bar = []
    for i,_ in enumerate(scmes):
        y_bar.extend(list_corrected)

    ax.set_xticks(x_bar,y_bar, rotation=90)
    ax.set_ylabel("Energy [pJ]", fontsize=12)

    fig.tight_layout()
    plt.show()
    plt.savefig(fig_path, format="png", bbox_inches='tight')
    print(f"Saved breakdown fig to {fig_path}")
