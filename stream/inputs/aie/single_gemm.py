workload = {
    0: {
        "operator_type": "Gemm",
        "equation": "O[B][K]+=I[B][C]*W[C][K]",
        "loop_dim_size": {
            "B": 32,
            "K": 512,
            "C": 512,
        },
        "operand_precision": {"O": 8, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["W", "I"],
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},

    }
    # 3: {  # Addition of layer 1 (residual path) and layer 3 (main path)
    #     'operator_type': 'Add',
    #     'equation': 'O[b][g][oy][ox]=X[b][g][oy][ox]+Y[b][g][oy][ox]',
    #     'dimension_relations': [],
    #     'loop_dim_size': {'B': 1, 'G': 256, 'OY': 112, 'OX': 112},
    #     'operand_precision': {'O': 8, 'O_final': 8, 'X': 8, 'Y': 8},
    #     'operand_source': {'X': [0], 'Y': [2]},
    #     'constant_operands': [],
    #     'operand_source_dimension_mapping': {'X': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}, 'Y': {'OX': 'OX', 'OY': 'OY', 'G': 'K'}},
    #     'memory_operand_links': {'O': 'O', 'X': 'I2', 'Y': 'I1'}
    # }
    # 3: {
    #     "operator_type": "Add",
    #     "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
    #     "dimension_relations": ["ix=2*ox+1*fx", "iy=2*oy+1*fy"],
    #     "loop_dim_size": {
    #         "B": 1,
    #         "K": 64,
    #         "C": 3,
    #         "OY": 112,
    #         "OX": 112,
    #         "FY": 7,
    #         "FX": 7,
    #     },
    #     "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
    #     "operand_source": {"W": [], "I": [2]},
    #     "constant_operands": ["W"],
    #     "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    #     "padding": {"IY": (3, 2), "IX": (3, 2)},
    # },
}