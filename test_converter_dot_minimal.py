import sys
import importlib.util
import torch
import triton
import triton.language as tl

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

    print(converted)
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
