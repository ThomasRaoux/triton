// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s


#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: init_barrier
  tt.func @init_barrier(%alloc: !tt.memdesc<1xi64, #shared0>) {
    // CHECK: "mbarrier.init.shared::cta.b64 $0, 0x1;", "r" %{{.*}} : (!llvm.ptr<3>) -> !llvm.void
    triton_nvidia_gpu.init_barrier %alloc, 1 : !tt.memdesc<1xi64, #shared0>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier
  tt.func @wait_barrier(%alloc: !tt.memdesc<1xi64, #shared0>, %phase: i32) {
    // CHECK: waitLoop:
    // CHECK: mbarrier.try_wait.parity.shared.b64
    // CHECK: !@P1 bra.uni waitLoop
    triton_nvidia_gpu.wait_barrier %alloc, %phase : !tt.memdesc<1xi64, #shared0>
    tt.return
  }
}
