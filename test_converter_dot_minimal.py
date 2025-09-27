import sys
import importlib.util
import torch
import triton
import triton.language as tl

import pytest

from sandbox import convert_triton_to_gluon


@triton.jit
def matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = 0
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m * K + (k + tl.arange(0, BLOCK_K))[None, :])
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N + offs_n)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m * N + offs_n, acc)


def test_triton_to_gluon_dot_minimal(tmp_path):
    # Convert directly from the Triton kernel object
    converted = convert_triton_to_gluon(matmul_tile_kernel.fn)
    # Write converted kernel to a file so @gluon.jit can retrieve source
    mod_path = tmp_path / "converted_dot_kernel.py"
    mod_path.write_text(converted)

    spec = importlib.util.spec_from_file_location("converted_dot_kernel", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["converted_dot_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    kernel = getattr(module, "matmul_tile_kernel")

    # Prepare inputs
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    M = BLOCK_M
    N = BLOCK_N
    K = BLOCK_K

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    # Launch converted kernel
    grid = (1, )
    kernel[grid](a, b, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4)

    # Reference
    ref = (a.float() @ b.float())
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)





@triton.jit
def matmul_kernel(  #
        a_ptr, b_ptr, output_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        A_TRANS: tl.constexpr = False):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    if not A_TRANS:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    else:
        a_ptrs = a_ptr + (offs_k[:, None] * stride_ak + offs_am[None, :] * stride_am)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        if A_TRANS:
            a = a.T
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=output_ptr.dtype.element_ty)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(output_ptrs, accumulator)


def get_src_element_ty_size(dtype_str):
    if dtype_str == "float8e5":
        return 1
    if dtype_str == "float16":
        return 2
    if dtype_str == "float32" or dtype_str == "tensorfloat32":
        return 4
    if dtype_str == "float64":
        return 8
    raise ValueError(f"Unknown dtype {dtype_str}")


@pytest.mark.parametrize("dtype_src_str", ["float16"])
@pytest.mark.parametrize("dtype_dst_str", ["float32"])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES", [(128, 128, 64, 1)])
@pytest.mark.parametrize("NUM_WARPS", [4])
def test_simple_matmul(dtype_src_str, dtype_dst_str, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS, tmp_path):
    device = "cuda"
    M, N, K = 1024, 512, 256
    torch.manual_seed(42)
    precision = "ieee"
    dtype_src_str = "float32" if dtype_src_str == "tensorfloat32" else dtype_src_str
    dtype_src = getattr(torch, dtype_src_str)

    # Convert directly from the Triton kernel object
    converted = convert_triton_to_gluon(matmul_kernel.fn)
    # Write converted kernel to a file so @gluon.jit can retrieve source
    mod_path = tmp_path / "converted_dot_kernel.py"
    mod_path.write_text(converted)

    spec = importlib.util.spec_from_file_location("converted_dot_kernel", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["converted_dot_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    kernel = getattr(module, "matmul_kernel")

    a = torch.randn(M, K, dtype=dtype_src, device=device)
    b = torch.randn(K, N, dtype=dtype_src, device=device)
    A = a
    B = b
    dtype_dst = getattr(torch, dtype_dst_str)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    k = kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                            output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K, num_warps=NUM_WARPS)
    ref_out = torch.matmul(A, B).to(torch.float32)
    output = output.to(torch.float32)
    atol = 0.001
    rtol = 0.001
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol)