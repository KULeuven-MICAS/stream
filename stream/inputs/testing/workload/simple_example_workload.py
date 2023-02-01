workload = {
    0: {
        'operator_type': 'Conv_pointwise',
        'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][oy][ox]',
        'loop_dim_size': {'B': 1, 'K': 16, 'C': 8, 'OY': 5, 'OX': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': []},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'C': 'K'}},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    1: {
        'operator_type': 'Conv_pointwise',
        'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][oy][ox]',
        'loop_dim_size': {'B': 1, 'K': 4, 'C': 16, 'OY': 5, 'OX': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'C': 'K'}},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
        }
    ,
    2: {
        'operator_type': 'Conv_pointwise',
        'equation': 'O[b][k][oy][ox]+=W[k][c]*I[b][c][oy][ox]',
        'loop_dim_size': {'B': 1, 'K': 4, 'C': 16, 'OY': 5, 'OX': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'W': 8, 'I': 8},
        'operand_source': {'W': [], 'I': [0]},
        'constant_operands': ['W'],
        'operand_source_dimension_mapping': {'I': {'OX': 'OX', 'OY': 'OY', 'C': 'K'}},
        'memory_operand_links': {'O': 'O', 'W': 'I2', 'I': 'I1'}
    }
    ,
    3: {  # Addition of layer 22 (residual connection) and layer 24 (main path)
        'operator_type': 'Add',
        'equation': 'O[b][g][oy][ox]=X[b][g][oy][ox]+Y[b][g][oy][ox]',
        'dimension_relations': [],
        'loop_dim_size': {'B': 1, 'G': 4, 'OY': 5, 'OX': 5},
        'operand_precision': {'O': 16, 'O_final': 8, 'X': 8, 'Y': 8},
        'operand_source': {'X': [1], 'Y': [2]},
        'constant_operands': [],
        'operand_source_dimension_mapping': {'X': {'OX': 'OX', 'OY': 'OY', 'G': 'G'}, 'Y': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}},
        'memory_operand_links': {'O': 'O', 'X': 'I2', 'Y': 'I1'}
    },
}
