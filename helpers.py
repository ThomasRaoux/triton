from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import hopper
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

@gluon.jit
def dot_accumulate(a, b, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=ttgl.float32,
        _semantic=None):
    # TODO: check if MMAv5 cannot be used and fallback to mmav2
    # Shapes (constexpr)
    M: ttgl.constexpr = a.type.shape[0]
    N: ttgl.constexpr = b.type.shape[1]
    K: ttgl.constexpr = a.type.shape[1]
    # Shared memory layouts for inputs (simple default)
    nvmma_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False, element_bitwidth=16,
                                                          rank=2)
    # Allocate shared memory and initialize with values
    a_smem = ttgl.allocate_shared_memory(a.dtype, [M, K], nvmma_layout, a)
    b_smem = ttgl.allocate_shared_memory(b.dtype, [K, N], nvmma_layout, b)
    # Allocate TMEM accumulator initialized with current acc
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
    tmem_reg_layout: ttgl.constexpr = get_tmem_32x32b_reg_layout(M, N, [M, N], ttgl.num_warps())
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem = allocate_tensor_memory(acc_temp.dtype, [M, N], acc_tmem_layout, acc_temp)
    # Barrier for commit
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    # MMA into TMEM, accumulating into existing TMEM contents
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    # Load back from TMEM using a register layout and convert to acc layout
    out = acc_tmem.load(tmem_reg_layout)
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)
    return out


@gluon.constexpr_function
def default_blocked_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr) -> ttgl.constexpr:
    # shape: list of positive ints (constexpr)
    rank = len(shape)
    # 1 element per thread for all dimensions
    size_per_thread = [1 for _ in range(rank)]
    # Distribute 32 threads per warp across dimensions (simple heuristic: last-fastest)
    threads_per_warp = [1 for _ in range(rank)]
    remaining_threads = 32
    for dim in range(rank - 1, -1, -1):
        threads_per_warp[dim] = remaining_threads
        remaining_threads = 1
        break
    # Use provided num_warps to distribute warps per CTA (put all on first dim)
    warps_per_cta = [1 for _ in range(rank)]
    warps_per_cta[0] = num_warps
    # Natural order [0, 1, ..., rank-1]
    order = [i for i in range(rank)]
    return ttgl.BlockedLayout(size_per_thread=size_per_thread, threads_per_warp=threads_per_warp,
                              warps_per_cta=warps_per_cta, order=order)

@gluon.jit
def descriptor_store(desc, offsets, value):
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    tma.async_copy_shared_to_global(desc, offsets, alloc)
    tma.store_wait(0)
    alloc._keep_alive()


@gluon.jit
def descriptor_load(desc, offsets):
    # Allocate shared memory tile matching descriptor block
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    # Allocate and initialize an mbarrier for the async TMA load
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    # Issue async copy from global (descriptor) to shared memory and wait for completion
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_copy_global_to_shared(desc, offsets, bar, smem)
    mbarrier.wait(bar, phase=0)
    # Load from shared memory into a register tensor using a reasonable default layout
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    out = smem.load(ret_layout)
    return out


@gluon.jit
def tl_arange(start, stop=None, step=None):
    # Normalize signature: tl.arange(N) -> (0, N)
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    # Derive default 1D layout when not provided
    layout: ttgl.constexpr = default_blocked_layout([(stop - start) // step], ttgl.num_warps())
    return ttgl.arange(start, stop, layout=layout)


@gluon.jit
def tl_full(shape, value, dtype=None):
    layout: ttgl.constexpr = default_blocked_layout(shape, ttgl.num_warps())
    return ttgl.full(shape, value, dtype, layout=layout)
