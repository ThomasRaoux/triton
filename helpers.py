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
from triton.experimental.gluon.language.nvidia.blackwell import tma as tma_blackwell
from typing import List, Tuple

@gluon.jit
def tl_dot(a, b, acc=None, input_precision=None, allow_tf32=None, 
           max_num_imprecise_acc=None, out_dtype=ttgl.float32):
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

@gluon.jit
def tl_dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format,
                  acc=None, fast_math=False, lhs_k_pack=True,
                  rhs_k_pack=True, out_dtype=ttgl.float32):
    return acc
    #ttgl.static_assert(False, "TODO: implement scaled dot in gluon")
    #return None

from triton.experimental.gluon.language._core import builtin

@builtin
def tl_make_tensor_descriptor(base, shape: List[ttgl.tensor], strides: List[ttgl.tensor], block_shape: List[ttgl.constexpr],
                              padding_option: ttgl.constexpr="zero", _semantic=None):
    ttgl.static_assert(False, "TODO: implement make_tensor_descriptor in gluon", _semantic=_semantic)
    return None


@gluon.constexpr_function
def get_num_threads_per_warp() -> ttgl.constexpr:
    return ttgl.constexpr(32)

@gluon.constexpr_function
def default_blocked_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr) -> ttgl.constexpr:
    # shape: list of positive ints (constexpr)
    rank = len(shape)
    # 1 element per thread for all dimensions
    size_per_thread = [1 for _ in range(rank)]
    # Distribute 32 threads per warp across dimensions (simple heuristic: last-fastest)
    threads_per_warp = [1 for _ in range(rank)]
    remaining_threads = get_num_threads_per_warp()
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
def tl_obj_store(obj, offsets, value):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_store_tensor_descriptor(obj, offsets, value)
    else:
        return obj.store(offsets, value)
@gluon.jit
def tl_obj_load(obj, offsets):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_load_tensor_descriptor(obj, offsets)
    else:
        return obj.load(offsets)        

@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        desc = obj
        alloc = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
        bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        tma_blackwell.async_gather(desc, x_offsets, y_offset, bar, alloc)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        # Load from shared memory into a register tensor using a reasonable default layout
        ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
        out = alloc.load(ret_layout)
        return out
    else:
        return obj.gather(x_offsets, y_offset)                

@gluon.jit
def tl_store_tensor_descriptor(desc, offsets, value):
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    tma.async_copy_shared_to_global(desc, offsets, alloc)
    tma.store_wait(0)
    alloc._keep_alive()


@gluon.jit
def tl_load_tensor_descriptor(desc, offsets):
    # Allocate shared memory tile matching descriptor block
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    # Allocate and initialize an mbarrier for the async TMA load
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    # Issue async copy from global (descriptor) to shared memory and wait for completion
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_copy_global_to_shared(desc, offsets, bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
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

@gluon.jit
def reset_to_default_layout(value):
    ty: ttgl.constexpr = value.type
    if isinstance(ty, ttgl.tuple_type):
      layout: ttgl.constexpr = default_blocked_layout(value[0].type.shape, ttgl.num_warps())
      return (ttgl.convert_layout(value[0], layout=layout), ttgl.convert_layout(value[1], layout=layout))
    else:
      layout: ttgl.constexpr = default_blocked_layout(ty.shape, ttgl.num_warps())
      return ttgl.convert_layout(value, layout=layout)


def current_target():
    from triton.runtime import driver
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()


current_target.__triton_builtin__ = True