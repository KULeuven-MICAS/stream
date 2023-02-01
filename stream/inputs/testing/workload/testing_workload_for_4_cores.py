workload = {
    0: {
        'operator_type': 'layer_on_core0',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]',
        'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 3, 'OY': 7, 'OX': 376, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 64, 'C': 3, 'OY': 7, 'OX': 37600, 'FY': 3, 'FX': 3},  # bug exists
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
        'loop_dim_size': {'B': 1, 'K': 8, 'C': 16, 'OY': 5, 'OX': 376, 'FY': 3, 'FX': 3},
        # 'loop_dim_size': {'B': 1, 'K': 8, 'C': 64, 'OY': 5, 'OX': 37600, 'FY': 3, 'FX': 3},  # bug exists
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        'padding': {'IY': (0, 0), 'IX': (1, 1)}
    }
    ,
    2: {
        'operator_type': 'layer_on_core2',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]',
        'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 48, 'C': 8, 'OY': 5, 'OX': 376, 'FY': 1, 'FX': 1},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [1]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        'padding': {'IY': (0, 0), 'IX': (0, 0)}
    }
    ,
    3: {
        'operator_type': 'layer_on_core3',
        'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]',
        'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
        'loop_dim_size': {'B': 1, 'K': 24, 'C': 48, 'OY': 5, 'OX': 376, 'FY': 1, 'FX': 1},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [2]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
        'padding': {'IY': (0, 0), 'IX': (0, 0)}
    },
    # 4: {
    #     'operator_type': 'layer_on_core0',
    #     'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]',
    #     'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
    #     'loop_dim_size': {'B': 1, 'K': 3, 'C': 24, 'OY': 5, 'OX': 376, 'FY': 3, 'FX': 3},
    #     'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
    #     'operand_source': {'W': [], 'I': [3]},
    #     'constant_operands': ['W'],
    #     'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
    #     'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
    #     'padding': {'IY': (1, 1), 'IX': (1, 1)}
    # },
    # 5: {
    #     'operator_type': 'layer_on_core1',
    #     'equation': 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]',
    #     'dimension_relations': ['ix=1*ox+1*fx', 'iy=1*oy+1*fy'],
    #     'loop_dim_size': {'B': 1, 'K': 12, 'C': 3, 'OY': 3, 'OX': 376, 'FY': 3, 'FX': 3},
    #     'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
    #     'operand_source': {'W': [], 'I': [4]},
    #     'constant_operands': ['W'],
    #     'operand_source_dimension_mapping': {'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}},
    #     'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'},
    #     'padding': {'IY': (0, 0), 'IX': (1, 1)}
    # }
}