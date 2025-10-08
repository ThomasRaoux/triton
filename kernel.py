import sys
import importlib.util
import torch
import triton
import triton.language as tl

import pytest
from sandbox import convert_triton_to_gluon
import kernel_impl as impl


@triton.jit
def matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    impl.impl_matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
