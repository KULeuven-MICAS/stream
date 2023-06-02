from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance

test_memory = MemoryInstance(
    name="sram_64KB",
    size=8192 * 8 * 8,
    r_bw=64 * 8,
    w_bw=64 * 8,
    r_cost=3.32 * 8,
    w_cost=3.85 * 8,
    area=0,
    r_port=1,
    w_port=1,
    rw_port=0,
    latency=1,
    min_r_granularity=64,
    min_w_granularity=64,
)
