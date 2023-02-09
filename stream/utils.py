def get_too_large_operands(cme, accelerator, core_id):
    """Create a list of memory operands for which an extra memory level (i.e. offchip) was added.

    Args:
        cme (CostModelEvaluation): The CostModelEvaluation containing information wrt the memory utilization.
        accelerator (Accelerator): The accelerator object containing the different cores.
        core_id (int): The id of the core of which we wish to get the too large operands.
    """
    too_large_operands = []
    core = accelerator.get_core(core_id)
    core_nb_memory_levels = core.memory_hierarchy.nb_levels
    for (layer_operand, l) in cme.mapping.data_elem_per_level.items():
        memory_operand = cme.layer.memory_operand_links[layer_operand]
        if len(l) > core_nb_memory_levels[memory_operand] + 1:  # +1 because of spatial level
            too_large_operands.append(memory_operand)
    return too_large_operands
