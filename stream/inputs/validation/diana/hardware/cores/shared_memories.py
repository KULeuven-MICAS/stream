from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance

shared_l1 = MemoryInstance(
    name="sram_256KB",
    size=8192 * 32 * 8,
    r_bw=64 * 8,
    w_bw=64 * 8,
    r_cost=33.2 * 8,
    w_cost=38.5 * 8,
    area=0,
    r_port=0,
    w_port=0,
    rw_port=2,
    latency=1,
    min_r_granularity=64,
    min_w_granularity=64,
)
