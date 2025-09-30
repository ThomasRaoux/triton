import sys
import importlib.util
import torch
import triton
import triton.language as tl

import pytest
from sandbox import convert_triton_to_gluon

@triton.jit
def impl_matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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


@triton.jit
def matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    impl_matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
