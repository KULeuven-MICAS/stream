from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance

# Common parameters across shared memories
name = "sram_256KB"
size = 8192 * 32 * 8
r_bw = 256 * 8
w_bw = 256 * 8
r_cost = 33.2 * 8
w_cost = 38.5 * 8
area = 0
r_port = 0
w_port = 0
rw_port = 2
latency = 1
min_r_granularity = 64
min_w_granularity = 64

shared_c0_c1 = MemoryInstance(
    name=name,
    size=size,
    r_bw=r_bw,
    w_bw=w_bw,
    r_cost=r_cost,
    w_cost=w_cost,
    area=area,
    r_port=r_port,
    w_port=w_port,
    rw_port=rw_port,
    latency=latency,
    min_r_granularity=min_r_granularity,
    min_w_granularity=min_w_granularity,
)

shared_c1_c2 = MemoryInstance(
    name=name,
    size=size,
    r_bw=r_bw,
    w_bw=w_bw,
    r_cost=r_cost,
    w_cost=w_cost,
    area=area,
    r_port=r_port,
    w_port=w_port,
    rw_port=rw_port,
    latency=latency,
    min_r_granularity=min_r_granularity,
    min_w_granularity=min_w_granularity,
)

shared_c2_c3 = MemoryInstance(
    name=name,
    size=size,
    r_bw=r_bw,
    w_bw=w_bw,
    r_cost=r_cost,
    w_cost=w_cost,
    area=area,
    r_port=r_port,
    w_port=w_port,
    rw_port=rw_port,
    latency=latency,
    min_r_granularity=min_r_granularity,
    min_w_granularity=min_w_granularity,
)

shared_c3_c4 = MemoryInstance(
    name=name,
    size=size,
    r_bw=r_bw,
    w_bw=w_bw,
    r_cost=r_cost,
    w_cost=w_cost,
    area=area,
    r_port=r_port,
    w_port=w_port,
    rw_port=rw_port,
    latency=latency,
    min_r_granularity=min_r_granularity,
    min_w_granularity=min_w_granularity,
)