workload = {
    0: {
        'operator_type': 'layer_on_core0',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]',
        'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 32, 'OX': 376, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 7, 'OX': 37600, 'FY': 3, 'FX': 3},   ## both input and output cannot
        # 'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 7, 'OX': 12500, 'FY': 3, 'FX': 3}, ## input can fit 1 CN a time, output cannot
        # 'loop_dim_size': {'B': 1, 'K': 64, 'C': 2, 'OY': 7, 'OX': 12500, 'FY': 2, 'FX': 3}, ## input can fit 2 CNs a time, output cannot
        # 'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 7, 'OX': 14562, 'FY': 3, 'FX': 3}, ## input can fit, output cannot (boundary condition)
        # 'loop_dim_size': {'B': 1, 'K': 3, 'C': 64, 'OY': 7, 'OX': 12500, 'FY': 3, 'FX': 3}, ## input cannot fit, output can
        # 'loop_dim_size': {'B': 1, 'K': 640, 'C': 30, 'OY': 7, 'OX': 7, 'FY': 3, 'FX': 3}, ## weight cannot fit, input can fit, output can
        # 'loop_dim_size': {'B': 1, 'K': 640, 'C': 30, 'OY': 7, 'OX': 7, 'FY': 3, 'FX': 3}, ## weight cannot fit, input can fit, output can
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'constant_operands': ['W', 'I'],
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        'padding': {'IY': (0, 0), 'IX': (1, 1)}
    }
    ,
    1: {
        'operator_type': 'layer_on_core1',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]',
        'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 8, 'C': 64, 'OY': 30, 'OX': 376, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 8, 'C': 64, 'OY': 5, 'OX': 37600, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 8, 'C': 64, 'OY': 5, 'OX': 12500, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 8, 'C': 64, 'OY': 5, 'OX': 12500, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 8, 'C': 64, 'OY': 5, 'OX': 14562, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 8, 'C': 3, 'OY': 5, 'OX': 12500, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 80, 'C': 640, 'OY': 5, 'OX': 7, 'FY': 3, 'FX': 3},  # weight cannot fit, input can, output can
        # 'loop_dim_size': {'B': 1, 'K': 8, 'C': 640, 'OY': 5, 'OX': 7, 'FY': 3, 'FX': 3},  # weight cannot fit, input can, output can
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        'padding': {'IY': (0, 0), 'IX': (1, 1)}
    }
    ,
    # 2: {
    #     'operator_type': 'Conv',
    #     'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]',
    #     'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
    #     'loop_dim_size': {'B': 1, 'K': 48, 'C': 16, 'OY': 3, 'OX': 376, 'FY': 1, 'FX': 1},
    #     'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
    #     'operand_source': {'W': [], 'I': [1]},
    #     'constant_operands': ['W'],
    #     'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
    #     'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
    #     'padding': {'IY': (0, 0), 'IX': (0, 0)}
    # }
}