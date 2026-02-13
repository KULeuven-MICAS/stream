swiglu:
    python main_swiglu.py --embedding_dim 32 --hidden_dim 32 --seq_len 32 --m 1 --k 32 --n 32 --in_dtype bf16 --out_dtype bf16 --trace_size 1048576 --rows 4 --cols 1 --npu 2 --line_size 32

swiglu2:
    python main_swiglu.py --seq_len 32 --embedding_dim 32 --hidden_dim 32 --line_size 32 --m 1 --k 32 --n 32 --in_dtype bf16 --out_dtype bf16 --trace_size 1048576 --rows 4 --cols 4 --npu npu2 --mapping_version 2

swiglu3:
    python main_swiglu.py --seq_len 256 --embedding_dim 512 --hidden_dim 2048 --in_dtype bf16 --out_dtype bf16 --trace_size 1048576 --rows 4 --cols 4 --npu npu2 --embedding_tile_size 32 --hidden_tile_size 64
