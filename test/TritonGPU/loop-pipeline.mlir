// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 | FileCheck %s --check-prefix=CHECK-NOCANON

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
//   CHECK-LABEL: @matmul_tma
//     CHECK-DAG:   triton_gpu.local_alloc  : () -> !tt.memdesc<3x128x64xf16, #{{.+}}, mutable>
//     CHECK-DAG:   triton_gpu.local_alloc  : () -> !tt.memdesc<3xi64, #{{.+}}, mutable>
//         CHECK:   triton_nvidia_gpu.init_barrier
//     CHECK-DAG:   triton_gpu.local_alloc  : () -> !tt.memdesc<3x64x256xf16, #{{.+}}, mutable>
//     CHECK-DAG:   triton_gpu.local_alloc  : () -> !tt.memdesc<3xi64, #{{.+}}, mutable>
// CHECK-COUNT-3:   triton_nvidia_gpu.init_barrier
// CHECK-COUNT-4:   triton_nvidia_gpu.async_tma_copy_global_to_local
//         CHECK:   scf.for
// CHECK-COUNT-2:     triton_nvidia_gpu.wait_barrier
// CHECK-COUNT-2:     triton_nvidia_gpu.async_tma_copy_global_to_local
//         CHECK:     scf.yield
  tt.func public @matmul_tma(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<128x256xf32, #mma> {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0:2 = scf.for %arg3 = %c0_i32 to %c256_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
      %1 = tt.experimental_descriptor_load %arg0[%c0_i32, %arg5] : !tt.ptr<i8> -> tensor<128x64xf16, #blocked>
      %2 = triton_gpu.local_alloc %1 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared>
      %3 = tt.experimental_descriptor_load %arg1[%arg5, %c0_i32] : !tt.ptr<i8> -> tensor<64x256xf16, #blocked1>
      %4 = triton_gpu.local_alloc %3 : (tensor<64x256xf16, #blocked1>) -> !tt.memdesc<64x256xf16, #shared>
      %5 = tt.dot %2, %4, %arg4, inputPrecision = tf32 : !tt.memdesc<128x64xf16, #shared> * !tt.memdesc<64x256xf16, #shared> -> tensor<128x256xf32, #mma>
      %6 = arith.addi %arg5, %c64_i32 : i32
      scf.yield %5, %6 : tensor<128x256xf32, #mma>, i32
    }
    tt.return %0#0 : tensor<128x256xf32, #mma>
  }
}
