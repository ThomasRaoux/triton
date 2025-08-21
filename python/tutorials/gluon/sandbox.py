
import itertools
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
    fence_async_shared,
    tcgen05_cp_smem_to_tmem,
)


@gluon.jit
def tmem_example_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr, num_warps: gl.constexpr):
    global_memory_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    # Get the register layout needed to access the tensor memory using a helper.
    tmem_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(
        M=128,
        N=128,
        shape=[M, N],
        num_warps=num_warps,
    )


    offs_m = gl.arange(0, M, gl.SliceLayout(1, tmem_reg_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, tmem_reg_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]

    input = gl.load(in_ptr + offs)

    # Allocate some tensor memory.
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(128, 128),
        unpacked=True,
    )

    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    smem = gl.allocate_shared_memory(in_ptr.dtype.element_ty, [M, N], layout=smem_layout)    

    smem.store(input)


    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )


    #input = gl.convert_layout(input, tmem_reg_layout)
    #tmem.store(input)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    tcgen05_cp_smem_to_tmem(smem, tmem)

    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)

    output = tmem.load(tmem_reg_layout)

    gl.store(out_ptr + offs, output)


def exp_tmem():
    M = 128
    N = 128
    num_warps = 4
   # input = torch.randn(M, N, dtype=torch.float32, device="cuda")
    input = torch.arange(M * N, device="cuda").reshape(M, N).to(torch.int32)
    output = torch.empty_like(input)

    tmem_example_kernel[(1, )](input, output, M, N, num_warps=num_warps)
    torch.set_printoptions(threshold=1000000)
    print(output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
    print("pass")

exp_tmem()