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


@gluon.jit
def dot_accumulate(a, b, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=ttgl.float32,
        _semantic=None):
    # TODO: check if MMAv5 cannot be used and fallback to mmav2
    # Shapes (constexpr)
    M: ttgl.constexpr = acc.type.shape[0]
    N: ttgl.constexpr = acc.type.shape[1]
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
    acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    acc_tmem = allocate_tensor_memory(acc.dtype, [M, N], acc_tmem_layout, acc_temp)
    # Barrier for commit
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    # MMA into TMEM, accumulating into existing TMEM contents
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    # Load back from TMEM using a register layout and convert to acc layout
    out = acc_tmem.load(tmem_reg_layout)
    out = ttgl.convert_layout(out, acc.type.layout)
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
