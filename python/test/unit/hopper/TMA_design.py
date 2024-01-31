
import itertools

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
from triton.runtime import driver


# current nvidia kernel.
@triton.jit
def full_static_persistent_matmul_kernel(a_ptr, b_ptr, w_ptr, bias_ptr, z_ptr,  #
                                         M, N, K,  #
                                         stride_am, stride_ak,  #
                                         stride_bk, stride_bn,  #
                                         stride_wm, stride_wn,  #
                                         stride_zm, stride_zn,  #
                                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                         GROUP_SIZE_M: tl.constexpr,  #
                                         out_dtype: tl.constexpr, USE_TMA_STORE: tl.constexpr,  #
                                         ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,  #
                                         DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,  #
                                         A_ORDER_0: tl.constexpr, A_ORDER_1: tl.constexpr,  #
                                         B_ORDER_0: tl.constexpr, B_ORDER_1: tl.constexpr,  #
                                         NUM_SMS: tl.constexpr  #
                                         ):
    start_pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = start_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pre_pid_m = first_pid_m + ((start_pid % num_pid_in_group) % group_size_m)
    pre_pid_n = (start_pid % num_pid_in_group) // group_size_m

    pre_block_offset_m = pre_pid_m * BLOCK_M
    pre_block_offset_n = pre_pid_n * BLOCK_N
    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   offsets=(pre_block_offset_m, 0), block_shape=(BLOCK_M, BLOCK_K),
                                   order=(A_ORDER_0, A_ORDER_1))
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                   offsets=(0, pre_block_offset_n), block_shape=(BLOCK_K, BLOCK_N),
                                   order=(B_ORDER_0, B_ORDER_1))
    w_tile_ptr = tl.make_block_ptr(base=w_ptr, shape=(N, N), strides=(stride_wm, stride_wn),
                                   offsets=(0, pre_block_offset_n), block_shape=(BLOCK_N, BLOCK_N), order=(0, 1))

    if USE_TMA_STORE:
        z_block_ptr = tl.make_block_ptr(base=z_ptr, shape=(M, N), strides=(stride_zm, stride_zn),
                                        offsets=(pre_block_offset_m, pre_block_offset_n),
                                        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        block_offset_m = pid_m * BLOCK_M
        block_offset_n = pid_n * BLOCK_N

        offs_m = block_offset_m + tl.arange(0, BLOCK_M)
        offs_n = block_offset_n + tl.arange(0, BLOCK_N)
        z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
        bias_ptrs = bias_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
        mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

        # TODO: lib/Dialect/TritonGPU/Transforms/RewriteTensorPointer.cpp does not support scf.if yet.
        # if tile_id >= NUM_SMS:
        #     a_tile_ptr = tl.advance(a_tile_ptr, [(pid_m - pre_pid_m) * BLOCK_M, -tl.cdiv(K, BLOCK_K) * BLOCK_K])
        #     b_tile_ptr = tl.advance(b_tile_ptr, [-tl.cdiv(K, BLOCK_K) * BLOCK_K, (pid_n - pre_pid_n) * BLOCK_N])

        a_tile_ptr = tl.advance(a_tile_ptr, [(pid_m - pre_pid_m) * BLOCK_M, 0])
        b_tile_ptr = tl.advance(b_tile_ptr, [0, (pid_n - pre_pid_n) * BLOCK_N])
        z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_tile_ptr, boundary_check=(0, 1))
            b = tl.load(b_tile_ptr, boundary_check=(0, 1))
            z += tl.dot(a, b)
            a_tile_ptr = tl.advance(a_tile_ptr, [0, BLOCK_K])
            b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_K, 0])
        a_tile_ptr = tl.advance(a_tile_ptr, [0, -tl.cdiv(K, BLOCK_K) * BLOCK_K])
        b_tile_ptr = tl.advance(b_tile_ptr, [-tl.cdiv(K, BLOCK_K) * BLOCK_K, 0])

        z = z.to(tl.float16)
        tl.store(z_ptrs, z, mask=mask)

        pre_pid_m = pid_m
        pre_pid_n = pid_n




# current nvidia kernel.
@triton.jit
def full_static_persistent_matmul_kernel(desc_ptr,
                                        a_ptr, b_ptr, w_ptr, bias_ptr, z_ptr,  #
                                         M, N, K,  #
                                         stride_am, stride_ak,  #
                                         stride_bk, stride_bn,  #
                                         stride_wm, stride_wn,  #
                                         stride_zm, stride_zn,  #
                                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                         GROUP_SIZE_M: tl.constexpr,  #
                                         out_dtype: tl.constexpr, USE_TMA_STORE: tl.constexpr,  #
                                         ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,  #
                                         DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,  #
                                         A_ORDER_0: tl.constexpr, A_ORDER_1: tl.constexpr,  #
                                         B_ORDER_0: tl.constexpr, B_ORDER_1: tl.constexpr,  #
                                         NUM_SMS: tl.constexpr  #
                                         ):
    start_pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = start_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pre_pid_m = first_pid_m + ((start_pid % num_pid_in_group) % group_size_m)
    pre_pid_n = (start_pid % num_pid_in_group) // group_size_m

    a_ptr = a_ptr + pre_pid_m * BLOCK_M
    b_ptr = b_ptr +  pre_pid_n * BLOCK_N
    desc_a = desc_ptr + start_pid * 2
    desc_b = desc_ptr + start_pid * 2 + 1
    tl.update_descriptor(desc_a, base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak))
    tl.update_descriptor(desc_b, base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn))

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        block_offset_m = pid_m * BLOCK_M
        block_offset_n = pid_n * BLOCK_N

        offs_m = block_offset_m + tl.arange(0, BLOCK_M)
        offs_n = block_offset_n + tl.arange(0, BLOCK_N)
        z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
        mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]

        # TODO: lib/Dialect/TritonGPU/Transforms/RewriteTensorPointer.cpp does not support scf.if yet.
        # if tile_id >= NUM_SMS:
        #     a_tile_ptr = tl.advance(a_tile_ptr, [(pid_m - pre_pid_m) * BLOCK_M, -tl.cdiv(K, BLOCK_K) * BLOCK_K])
        #     b_tile_ptr = tl.advance(b_tile_ptr, [-tl.cdiv(K, BLOCK_K) * BLOCK_K, (pid_n - pre_pid_n) * BLOCK_N])
        offset_m = (pid_m - pre_pid_m) * BLOCK_M
        offset_n = (pid_m - pre_pid_m) * BLOCK_M
        offset_k = 0
        z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(desc_a, [offset_m, offset_k])
            b = tl.load(desc_b, [offset_k, offset_n])
            z += tl.dot(a, b)
            offset_k += BLOCK_K

        z = z.to(tl.float16)
        tl.store(z_ptrs, z, mask=mask)

        pre_pid_m = pid_m
        pre_pid_n = pid_n