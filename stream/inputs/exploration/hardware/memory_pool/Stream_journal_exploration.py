from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance

def reg_64B_1r1w_8b():
    reg_64B_1r1w_8b = MemoryInstance(
        name="reg_64B_1r1w_8b",
        size=64 * 8,
        r_bw=8,
        w_bw=8,
        r_cost=0.06,
        w_cost=0.06,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )
    return reg_64B_1r1w_8b

def reg_64B_2r2w_16b():
    reg_64B_2r2w_16b = MemoryInstance(
        name="reg_64B_2r2w_16b",
        size=64 * 8,
        r_bw=8,
        w_bw=8,
        r_cost=0.1,
        w_cost=0.1,
        area=0,
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )
    return reg_64B_2r2w_16b

def reg_256B_2r2w_16b():
    reg_256B_2r2w_16b = MemoryInstance(
        name="reg_256B_2r2w_16b",
        size=256 * 8,
        r_bw=16,
        w_bw=16,
        r_cost=0.258,
        w_cost=0.356,
        area=0,
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )
    return reg_256B_2r2w_16b

def reg_256B_1r1w_8b():
    reg_256B_1r1w_8b = MemoryInstance(
        name="reg_256B_1r1w_8b",
        size=256 * 8,
        r_bw=8,
        w_bw=8,
        r_cost=0.205,
        w_cost=0.217,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )
    return reg_256B_1r1w_8b

def reg_512B_1r1w_8b():
    reg_512B_1r1w_8b = MemoryInstance(
        name="reg_512B_1r1w_8b",
        size=512 * 8,
        r_bw=8,
        w_bw=8,
        r_cost=0.296,
        w_cost=0.326,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
    )
    return reg_512B_1r1w_8b

def reg_512B_2r2w_16b():
    reg_512B_2r2w_16b = MemoryInstance(
        name="reg_512B_2r2w_16b",
        size=512 * 8,
        r_bw=16,
        w_bw=16,
        r_cost=0.454,
        w_cost=0.485,
        area=0,
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )
    return reg_512B_2r2w_16b

def reg_1KB_2r2w_16b():
    reg_1KB_2r2w_16b = MemoryInstance(
        name="reg_1KB_2r2w_16b",
        size=1024 * 8,
        r_bw=16,
        w_bw=16,
        r_cost=0.625,
        w_cost=0.7,
        area=0,
        r_port=2,
        w_port=2,
        rw_port=0,
        latency=1,
    )
    return reg_1KB_2r2w_16b

def sram_32KB_2_16KB_1r1w_64b():
    sram_32KB_2_16KB_1r1w_64b = MemoryInstance(
        name="sram_32KB_2_16KB_1r1w_64b",
        size=2 * 16 * 1024 * 8,
        r_bw=2 * 64,
        w_bw=2 * 64,
        r_cost=2 * 6.4,
        w_cost=2 * 5.22,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=64,
        min_w_granularity=64,
    )
    return sram_32KB_2_16KB_1r1w_64b

def sram_32KB_2_16KB_1r1w_128b():
    sram_32KB_2_16KB_1r1w_128b = MemoryInstance(
        name="sram_32KB_2_16KB_1r1w_128b",
        size=2 * 16 * 1024 * 8,
        r_bw=2 * 128,
        w_bw=2 * 128,
        r_cost=2 * 6.08,
        w_cost=2 * 11.67,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=128,
        min_w_granularity=128,
    )
    return sram_32KB_2_16KB_1r1w_128b

def sram_32KB_2_16KB_1r1w_224b():
    sram_32KB_2_16KB_1r1w_224b = MemoryInstance(
        name="sram_32KB_2_16KB_1r1w_224b",
        size=2 * 16 * 1024 * 8,
        r_bw=2 * 224,
        w_bw=2 * 224,
        r_cost=2 * 10.17,
        w_cost=2 * 15.73,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=224,
        min_w_granularity=224,
    )
    return sram_32KB_2_16KB_1r1w_224b

def sram_128KB_8_16KB_1r1w_64b():
    sram_128KB_8_16KB_1r1w_64b = MemoryInstance(
        name="sram_128KB_8_16KB_1r1w_64b",
        size=8 * 16 * 1024 * 8,
        r_bw=8 * 64,
        w_bw=8 * 64,
        r_cost=8 * 6.4,
        w_cost=8 * 5.22,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=64,
        min_w_granularity=64,
    )
    return sram_128KB_8_16KB_1r1w_64b

def sram_256KB_2_128KB_1r1w_64b():
    sram_256KB_2_128KB_1r1w_64b = MemoryInstance(
        name="sram_256KB_2_128KB_1r1w_64b",
        size=2 * 128 * 1024 * 8,
        r_bw=2 * 64,
        w_bw=2 * 64,
        r_cost=2 * 17.96,
        w_cost=2 * 13.38,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=64,
        min_w_granularity=64,
    )
    return sram_256KB_2_128KB_1r1w_64b

def sram_256KB_2_128KB_1r1w_36b():
    sram_256KB_2_128KB_1r1w_36b = MemoryInstance(
        name="sram_256KB_2_128KB_1r1w_36b",
        size=2 * 128 * 1024 * 8,
        r_bw=2 * 36,
        w_bw=2 * 36,
        r_cost=2 * 13.19,
        w_cost=2 * 10.89,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=36,
        min_w_granularity=36,
    )
    return sram_256KB_2_128KB_1r1w_36b

def sram_256KB_2_128KB_1r1w_128b():
    sram_256KB_2_128KB_1r1w_128b = MemoryInstance(
        name="sram_256KB_2_128KB_1r1w_128b",
        size=2 * 128 * 1024 * 8,
        r_bw=2 * 128,
        w_bw=2 * 128,
        r_cost=2 * 26,
        w_cost=2 * 23.65,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=128,
        min_w_granularity=128,
    )
    return sram_256KB_2_128KB_1r1w_128b

def sram_256KB_2_128KB_1r1w_256b():
    sram_256KB_2_128KB_1r1w_256b = MemoryInstance(
        name="sram_256KB_2_128KB_1r1w_256b",
        size=2 * 128 * 1024 * 8,
        r_bw=2 * 128,
        w_bw=2 * 128,
        r_cost=2 * 41.54,
        w_cost=2 * 45.95,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=256,
        min_w_granularity=256,
    )
    return sram_256KB_2_128KB_1r1w_256b

def sram_512KB_32_16KB_1r1w_32b():
    sram_512KB_32_16KB_1r1w_32b = MemoryInstance(
        name="sram_512KB_32_16KB_1r1w_32b",
        size=32 * 16 * 1024 * 8,
        r_bw=32 * 32,
        w_bw=32 * 32,
        r_cost=32 * 5.47,
        w_cost=32 * 3.17,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=32,
        min_w_granularity=32,
    )
    return sram_512KB_32_16KB_1r1w_32b

def sram_1MB_8_128KB_1r1w_32b():
    sram_1MB_8_128KB_1r1w_32b = MemoryInstance(
        name="sram_1MB_8_128KB_1r1w_32b",
        size=8 * 128 * 1024 * 8,
        r_bw=8 * 32,
        w_bw=8 * 32,
        r_cost=8 * 12.22,
        w_cost=8 * 9.21,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=32,
        min_w_granularity=32,
    )
    return sram_1MB_8_128KB_1r1w_32b

def sram_1MB_8_128KB_1r1w_64b():
    sram_1MB_8_128KB_1r1w_64b = MemoryInstance(
        name="sram_1MB_8_128KB_1r1w_64b",
        size=8 * 128 * 1024 * 8,
        r_bw=8 * 64,
        w_bw=8 * 64,
        r_cost=8 * 17.96,
        w_cost=8 * 13.38,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=64,
        min_w_granularity=64,
    )
    return sram_1MB_8_128KB_1r1w_64b

def sram_2MB_128_16KB_1r1w_16b():
    sram_2MB_128_16KB_1r1w_16b = MemoryInstance(
        name="sram_2MB_128_16KB_1r1w_16b",
        size=128 * 16 * 1024 * 8,
        r_bw=128 * 16,
        w_bw=128 * 16,
        r_cost=128 * 3.31,
        w_cost=128 * 2.16,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=16,
        min_w_granularity=16,
    )
    return sram_2MB_128_16KB_1r1w_16b

def sram_4MB_32_128KB_1r1w_16b():
    sram_4MB_32_128KB_1r1w_16b = MemoryInstance(
        name="sram_4MB_32_128KB_1r1w_16b",
        size=32 * 128 * 1024 * 8,
        r_bw=32 * 16,
        w_bw=32 * 16,
        r_cost=32 * 10.12,
        w_cost=32 * 6.88,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=16,
        min_w_granularity=16,
    )
    return sram_4MB_32_128KB_1r1w_16b

def sram_4MB_32_128KB_1r1w_32b():
    sram_4MB_32_128KB_1r1w_32b = MemoryInstance(
        name="sram_4MB_32_128KB_1r1w_32b",
        size=32 * 128 * 1024 * 8,
        r_bw=32 * 32,
        w_bw=32 * 32,
        r_cost=32 * 12.22,
        w_cost=32 * 9.21,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=32,
        min_w_granularity=32,
    )
    return sram_4MB_32_128KB_1r1w_32b

def sram_16MB_128_128KB_1r1w_16b():
    sram_16MB_128_128KB_1r1w_16b = MemoryInstance(
        name="sram_16MB_128_128KB_1r1w_16b",
        size=128 * 128 * 1024 * 8,
        r_bw=128 * 16,
        w_bw=128 * 16,
        r_cost=128 * 10.12,
        w_cost=128 * 6.88,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=16,
        min_w_granularity=16,
    )
    return sram_16MB_128_128KB_1r1w_16b

def sram_16MB_128_128KB_1r1w_8b():
    sram_16MB_128_128KB_1r1w_8b = MemoryInstance(
        name="sram_16MB_128_128KB_1r1w_8b",
        size=128 * 128 * 1024 * 8,
        r_bw=128 * 8,
        w_bw=128 * 8,
        r_cost=128 * 9.19,
        w_cost=128 * 5.8,
        area=0,
        r_port=1,
        w_port=1,
        rw_port=0,
        latency=1,
        min_r_granularity=8,
        min_w_granularity=8,
    )
    return sram_16MB_128_128KB_1r1w_8b

dram_single_port = MemoryInstance(
    name="dram_sp",
    size=10000000000,
    r_bw=64,
    w_bw=64,
    r_cost=650,
    w_cost=650,
    area=0,
    r_port=0,
    w_port=0,
    rw_port=1,
    latency=1,
    min_r_granularity=64,
    min_w_granularity=64,
)

dram_dual_port = MemoryInstance(
    name="dram_dp",
    size=10000000000,
    r_bw=64,
    w_bw=64,
    r_cost=700,
    w_cost=700,
    area=0,
    r_port=1,
    w_port=1,
    rw_port=0,
    latency=1,
    min_r_granularity=64,
    min_w_granularity=64,
)
