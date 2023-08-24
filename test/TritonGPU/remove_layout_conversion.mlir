#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [4, 1, 8], warpsPerCTA = [4, 1, 1], order = [1, 2, 0], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [1, 0, 2]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [0, 2, 1], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [0, 1, 2]}>
#blocked6 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked7 = #triton_gpu.blocked<{sizePerThread = [8, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 4], order = [1, 0, 2], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [1, 0, 2]}>
#blocked8 = #triton_gpu.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 4], order = [0, 1, 2], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [0, 1, 2]}>
#blocked9 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @attention_fw(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg18: i32, %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg21: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked1>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked2>
    %cst_3 = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = arith.muli %1, %arg10 : i32
    %4 = tt.addptr %arg0, %2 : !tt.ptr<f16, 1>, i32
    %5 = arith.muli %0, %c128_i32 : i32
    %6 = arith.extsi %arg8 : i32 to i64
    %7 = arith.extsi %5 : i32 to i64
    %8 = tt.addptr %arg1, %3 : !tt.ptr<f16, 1>, i32
    %9 = arith.addi %arg20, %arg21 : i32
    %10 = arith.extsi %arg11 : i32 to i64
    %11 = tt.addptr %arg2, %3 : !tt.ptr<f16, 1>, i32
    %12 = arith.extsi %arg14 : i32 to i64
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %14 = tt.splat %5 : (i32) -> tensor<128xi32, #blocked1>
    %15 = arith.addi %14, %13 : tensor<128xi32, #blocked1>
    %16 = arith.mulf %arg3, %cst_3 : f32
    %17 = tt.splat %4 : (!tt.ptr<f16, 1>) -> tensor<128x64x!tt.ptr<f16, 1>, #blocked3>
    %18 = tt.splat %7 : (i64) -> tensor<128xi64, #blocked3>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3>
    %20 = arith.extsi %19 : tensor<128xi32, #blocked3> to tensor<128xi64, #blocked3>
    %21 = arith.addi %18, %20 : tensor<128xi64, #blocked3>
    %22 = triton_gpu.convert_layout %21 : (tensor<128xi64, #blocked3>) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %23 = tt.expand_dims %22 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<128x1xi64, #blocked4>
    %24 = tt.splat %6 : (i64) -> tensor<128x1xi64, #blocked4>
    %25 = arith.muli %23, %24 : tensor<128x1xi64, #blocked4>
    %26 = tt.broadcast %25 : (tensor<128x1xi64, #blocked4>) -> tensor<128x64xi64, #blocked4>
    %27 = triton_gpu.convert_layout %26 : (tensor<128x64xi64, #blocked4>) -> tensor<128x64xi64, #blocked3>
    %28 = tt.addptr %17, %27 : tensor<128x64x!tt.ptr<f16, 1>, #blocked3>, tensor<128x64xi64, #blocked3>
    %29 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %30 = arith.extsi %29 : tensor<64xi32, #blocked3> to tensor<64xi64, #blocked3>
    %31 = triton_gpu.convert_layout %30 : (tensor<64xi64, #blocked3>) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked5}>>
    %32 = tt.expand_dims %31 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked5}>>) -> tensor<1x64xi64, #blocked5>
    %33 = tt.broadcast %32 : (tensor<1x64xi64, #blocked5>) -> tensor<128x64xi64, #blocked5>
    %34 = triton_gpu.convert_layout %33 : (tensor<128x64xi64, #blocked5>) -> tensor<128x64xi64, #blocked3>
    %35 = tt.addptr %28, %34 : tensor<128x64x!tt.ptr<f16, 1>, #blocked3>, tensor<128x64xi64, #blocked3>
    %36 = tt.load %35 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked3>
    %37 = triton_gpu.convert_layout %36 : (tensor<128x64xf16, #blocked3>) -> tensor<128x64xf16, #blocked2>
    %38 = tt.splat %16 : (f32) -> tensor<128x64xf32, #blocked2>
    %39 = arith.extf %37 : tensor<128x64xf16, #blocked2> to tensor<128x64xf32, #blocked2>
    %40 = arith.mulf %39, %38 : tensor<128x64xf32, #blocked2>
    %41 = arith.truncf %40 : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
    %42:5 = scf.for %arg22 = %c0_i32 to %9 step %c64_i32 iter_args(%arg23 = %cst_2, %arg24 = %cst_1, %arg25 = %cst_0, %arg26 = %c0_i64, %arg27 = %c0_i64) -> (tensor<128x64xf32, #blocked2>, tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>, i64, i64)  : i32 {
      %78 = tt.splat %8 : (!tt.ptr<f16, 1>) -> tensor<64x64x!tt.ptr<f16, 1>, #blocked6>
      %79 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked6>
      %80 = arith.extsi %79 : tensor<64xi32, #blocked6> to tensor<64xi64, #blocked6>
      %81 = triton_gpu.convert_layout %80 : (tensor<64xi64, #blocked6>) -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>
      %82 = tt.expand_dims %81 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked7}>>) -> tensor<64x1xi64, #blocked7>
      %83 = tt.broadcast %82 : (tensor<64x1xi64, #blocked7>) -> tensor<64x64xi64, #blocked7>
      %84 = triton_gpu.convert_layout %83 : (tensor<64x64xi64, #blocked7>) -> tensor<64x64xi64, #blocked6>
      %85 = tt.addptr %78, %84 : tensor<64x64x!tt.ptr<f16, 1>, #blocked6>, tensor<64x64xi64, #blocked6>
      %86 = tt.splat %arg26 : (i64) -> tensor<64xi64, #blocked6>
      %87 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked6>
      %88 = arith.extsi %87 : tensor<64xi32, #blocked6> to tensor<64xi64, #blocked6>
      %89 = arith.addi %86, %88 : tensor<64xi64, #blocked6>
      %90 = triton_gpu.convert_layout %89 : (tensor<64xi64, #blocked6>) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked8}>>
      %91 = tt.expand_dims %90 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked8}>>) -> tensor<1x64xi64, #blocked8>
      %92 = tt.splat %10 : (i64) -> tensor<1x64xi64, #blocked8>
      %93 = arith.muli %91, %92 : tensor<1x64xi64, #blocked8>
      %94 = tt.broadcast %93 : (tensor<1x64xi64, #blocked8>) -> tensor<64x64xi64, #blocked8>
      %95 = triton_gpu.convert_layout %94 : (tensor<64x64xi64, #blocked8>) -> tensor<64x64xi64, #blocked6>
      %96 = tt.addptr %85, %95 : tensor<64x64x!tt.ptr<f16, 1>, #blocked6>, tensor<64x64xi64, #blocked6>
      %97 = tt.load %96 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf16, #blocked6>
      %98 = tt.splat %11 : (!tt.ptr<f16, 1>) -> tensor<64x64x!tt.ptr<f16, 1>, #blocked3>
      %99 = tt.splat %arg27 : (i64) -> tensor<64xi64, #blocked3>
      %100 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
      %101 = arith.extsi %100 : tensor<64xi32, #blocked3> to tensor<64xi64, #blocked3>
      %102 = arith.addi %99, %101 : tensor<64xi64, #blocked3>
      %103 = triton_gpu.convert_layout %102 : (tensor<64xi64, #blocked3>) -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
      %104 = tt.expand_dims %103 {axis = 1 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi64, #blocked4>
      %105 = tt.splat %12 : (i64) -> tensor<64x1xi64, #blocked4>
      %106 = arith.muli %104, %105 : tensor<64x1xi64, #blocked4>
      %107 = tt.broadcast %106 : (tensor<64x1xi64, #blocked4>) -> tensor<64x64xi64, #blocked4>
      %108 = triton_gpu.convert_layout %107 : (tensor<64x64xi64, #blocked4>) -> tensor<64x64xi64, #blocked3>
      %109 = tt.addptr %98, %108 : tensor<64x64x!tt.ptr<f16, 1>, #blocked3>, tensor<64x64xi64, #blocked3>
      %110 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
      %111 = arith.extsi %110 : tensor<64xi32, #blocked3> to tensor<64xi64, #blocked3>
      %112 = triton_gpu.convert_layout %111 : (tensor<64xi64, #blocked3>) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked5}>>
      %113 = tt.expand_dims %112 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked5}>>) -> tensor<1x64xi64, #blocked5>
      %114 = tt.broadcast %113 : (tensor<1x64xi64, #blocked5>) -> tensor<64x64xi64, #blocked5>
      %115 = triton_gpu.convert_layout %114 : (tensor<64x64xi64, #blocked5>) -> tensor<64x64xi64, #blocked3>
      %116 = tt.addptr %109, %115 : tensor<64x64x!tt.ptr<f16, 1>, #blocked3>, tensor<64x64xi64, #blocked3>
      %117 = tt.load %116 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf16, #blocked3>
      %118 = triton_gpu.convert_layout %41 : (tensor<128x64xf16, #blocked2>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
      %119 = triton_gpu.convert_layout %97 : (tensor<64x64xf16, #blocked6>) -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
      %120 = tt.dot %118, %119, %cst {allowTF32 = true} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xf16, #blocked>
      %121 = triton_gpu.convert_layout %120 : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #blocked2>
      %122 = arith.extf %121 : tensor<128x64xf16, #blocked2> to tensor<128x64xf32, #blocked2>
      %123 = "tt.reduce"(%122) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32, %arg29: f32):
        %153 = arith.maxf %arg28, %arg29 : f32
        tt.reduce.return %153 : f32
      }) : (tensor<128x64xf32, #blocked2>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %124 = triton_gpu.convert_layout %123 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128xf32, #blocked1>
      %125 = arith.maxf %arg25, %124 : tensor<128xf32, #blocked1>
      %126 = arith.subf %arg25, %125 : tensor<128xf32, #blocked1>
      %127 = tt.pure_extern_elementwise %126 {libname = "libdevice", libpath = "/root/.pyenv/versions/3.9.9/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_exp2f"} : (tensor<128xf32, #blocked1>) -> tensor<128xf32, #blocked1>
      %128 = triton_gpu.convert_layout %125 : (tensor<128xf32, #blocked1>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
      %129 = tt.expand_dims %128 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>) -> tensor<128x1xf32, #blocked9>
      %130 = triton_gpu.convert_layout %129 : (tensor<128x1xf32, #blocked9>) -> tensor<128x1xf32, #blocked2>
      %131 = tt.broadcast %130 : (tensor<128x1xf32, #blocked2>) -> tensor<128x64xf32, #blocked2>
      %132 = arith.subf %122, %131 : tensor<128x64xf32, #blocked2>
      %133 = tt.pure_extern_elementwise %132 {libname = "libdevice", libpath = "/root/.pyenv/versions/3.9.9/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_exp2f"} : (tensor<128x64xf32, #blocked2>) -> tensor<128x64xf32, #blocked2>
      %134 = arith.mulf %arg24, %cst_1 : tensor<128xf32, #blocked1>
      %135 = arith.addf %134, %127 : tensor<128xf32, #blocked1>
      %136 = triton_gpu.convert_layout %135 : (tensor<128xf32, #blocked1>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
      %137 = tt.expand_dims %136 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>) -> tensor<128x1xf32, #blocked9>
      %138 = triton_gpu.convert_layout %137 : (tensor<128x1xf32, #blocked9>) -> tensor<128x1xf32, #blocked2>
      %139 = tt.broadcast %138 : (tensor<128x1xf32, #blocked2>) -> tensor<128x64xf32, #blocked2>
      %140 = arith.mulf %arg23, %139 : tensor<128x64xf32, #blocked2>
      %141 = arith.truncf %133 : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
      %142 = triton_gpu.convert_layout %141 : (tensor<128x64xf16, #blocked2>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
      %143 = triton_gpu.convert_layout %117 : (tensor<64x64xf16, #blocked3>) -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
      %144 = triton_gpu.convert_layout %140 : (tensor<128x64xf32, #blocked2>) -> tensor<128x64xf32, #blocked>
      %145 = tt.dot %142, %143, %144 {allowTF32 = true} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x64xf32, #blocked>
      %146 = triton_gpu.convert_layout %145 : (tensor<128x64xf32, #blocked>) -> tensor<128x64xf32, #blocked2>
      %147 = arith.mulf %arg24, %127 : tensor<128xf32, #blocked1>
      %148 = "tt.reduce"(%133) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32, %arg29: f32):
        %153 = arith.addf %arg28, %arg29 : f32
        tt.reduce.return %153 : f32
      }) : (tensor<128x64xf32, #blocked2>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %149 = triton_gpu.convert_layout %148 : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128xf32, #blocked1>
      %150 = arith.addf %147, %149 : tensor<128xf32, #blocked1>
      %151 = arith.addi %arg26, %c64_i64 : i64
      %152 = arith.addi %arg27, %c64_i64 : i64
      scf.yield %146, %150, %125, %151, %152 : tensor<128x64xf32, #blocked2>, tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>, i64, i64
    }
    %43 = triton_gpu.convert_layout %42#1 : (tensor<128xf32, #blocked1>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>
    %44 = tt.expand_dims %43 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked9}>>) -> tensor<128x1xf32, #blocked9>
    %45 = triton_gpu.convert_layout %44 : (tensor<128x1xf32, #blocked9>) -> tensor<128x1xf32, #blocked2>
    %46 = tt.broadcast %45 : (tensor<128x1xf32, #blocked2>) -> tensor<128x64xf32, #blocked2>
    %47 = arith.divf %42#0, %46 : tensor<128x64xf32, #blocked2>
    %48 = arith.muli %1, %arg20 : i32
    %49 = tt.addptr %arg4, %48 : !tt.ptr<f32, 1>, i32
    %50 = tt.splat %49 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked1>
    %51 = tt.addptr %50, %15 : tensor<128x!tt.ptr<f32, 1>, #blocked1>, tensor<128xi32, #blocked1>
    %52 = tt.pure_extern_elementwise %42#1 {libname = "libdevice", libpath = "/root/.pyenv/versions/3.9.9/lib/python3.9/site-packages/triton/language/../third_party/cuda/lib/libdevice.10.bc", symbol = "__nv_log2f"} : (tensor<128xf32, #blocked1>) -> tensor<128xf32, #blocked1>
    %53 = arith.addf %42#2, %52 : tensor<128xf32, #blocked1>
    tt.store %51, %53 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked1>
    %54 = tt.addptr %arg5, %2 : !tt.ptr<f16, 1>, i32
    %55 = arith.extsi %arg17 : i32 to i64
    %56 = arith.extsi %5 : i32 to i64
    %57 = arith.truncf %47 : tensor<128x64xf32, #blocked2> to tensor<128x64xf16, #blocked2>
    %58 = triton_gpu.convert_layout %57 : (tensor<128x64xf16, #blocked2>) -> tensor<128x64xf16, #blocked3>
    %59 = tt.splat %54 : (!tt.ptr<f16, 1>) -> tensor<128x64x!tt.ptr<f16, 1>, #blocked3>
    %60 = tt.splat %56 : (i64) -> tensor<128xi64, #blocked3>
    %61 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3>
    %62 = arith.extsi %61 : tensor<128xi32, #blocked3> to tensor<128xi64, #blocked3>
    %63 = arith.addi %60, %62 : tensor<128xi64, #blocked3>
    %64 = triton_gpu.convert_layout %63 : (tensor<128xi64, #blocked3>) -> tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : (tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<128x1xi64, #blocked4>
    %66 = tt.splat %55 : (i64) -> tensor<128x1xi64, #blocked4>
    %67 = arith.muli %65, %66 : tensor<128x1xi64, #blocked4>
    %68 = tt.broadcast %67 : (tensor<128x1xi64, #blocked4>) -> tensor<128x64xi64, #blocked4>
    %69 = triton_gpu.convert_layout %68 : (tensor<128x64xi64, #blocked4>) -> tensor<128x64xi64, #blocked3>
    %70 = tt.addptr %59, %69 : tensor<128x64x!tt.ptr<f16, 1>, #blocked3>, tensor<128x64xi64, #blocked3>
    %71 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %72 = arith.extsi %71 : tensor<64xi32, #blocked3> to tensor<64xi64, #blocked3>
    %73 = triton_gpu.convert_layout %72 : (tensor<64xi64, #blocked3>) -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked5}>>
    %74 = tt.expand_dims %73 {axis = 0 : i32} : (tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked5}>>) -> tensor<1x64xi64, #blocked5>
    %75 = tt.broadcast %74 : (tensor<1x64xi64, #blocked5>) -> tensor<128x64xi64, #blocked5>
    %76 = triton_gpu.convert_layout %75 : (tensor<128x64xi64, #blocked5>) -> tensor<128x64xi64, #blocked3>
    %77 = tt.addptr %70, %76 : tensor<128x64x!tt.ptr<f16, 1>, #blocked3>, tensor<128x64xi64, #blocked3>
    tt.store %77, %58 {cache = 1 : i32, evict = 1 : i32} : tensor<128x64xf16, #blocked3>
    tt.return
  }
}
