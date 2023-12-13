/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "TensorPtrOpsToLLVM.h"
using namespace mlir;
using namespace mlir::triton;

static CUtensorMapDataType getCUtensorMapDataType(Type ty) {
  if (ty.isF16()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if (ty.isBF16()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (ty.isF32()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if (ty.getIntOrFloatBitWidth() == 8) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else {
    llvm::report_fatal_error("Unsupported elemTy for InsertSliceAsyncV2Op");
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  }
}

struct MakeTensorPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::MakeTensorPtrOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::MakeTensorPtrOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // struct { offset0, offset1, descriptor_shared_memory_address };
    auto offsets = adaptor.getOffsets();
    auto shapes = adaptor.getShape();
    auto strides = adaptor.getStrides();
    auto base = adaptor.getBase();
    auto result = op.getResult();

    Value smemBase = getSharedMemoryBase(op.getLoc(), rewriter, op.getResult());
    // Set descriptor
    PTXBuilder ptxBuilder;
    SmallVector<PTXBuilder::Operand *> operands;
    std::string asmStr;

    std::vector<uint32_t> boxDims;
    auto tensorTy = result.getType()
                        .cast<triton::PointerType>()
                        .getPointeeType()
                        .cast<RankedTensorType>();
    auto tensorShape = tensorTy.getShape();
    auto inOrder = triton::gpu::getOrder(tensorTy.getEncoding());

    auto elTy = tensorTy.getElementType();
    int rank = shapes.size();
    auto shapePerCTA = mlir::triton::gpu::getShapePerCTA(tensorTy);
    auto getDimOfOrder = [](ArrayRef<unsigned> order, int32_t i) {
      auto it = std::find(order.begin(), order.end(), i);
      assert(it != order.end());
      return std::distance(order.begin(), it);
    };
    for (size_t i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(inOrder, i);
      boxDims.emplace_back(shapePerCTA[dim]);
    }

    operands.push_back(ptxBuilder.newOperand(smemBase, "r"));
    operands.push_back(ptxBuilder.newOperand(base, "l"));
    asmStr.append("tensormap.replace.mode.global_address.shared::cta.b1024.b64 "
                  "[$0], $1;");
    asmStr.append("tensormap.replace.mode.rank.shared::cta.b1024.b32 [$0], " +
                  std::to_string(rank - 1) + ";");
    for (size_t i = 0; i < rank; ++i) {
      operands.push_back(ptxBuilder.newOperand(shapes[i], "r"));
      asmStr.append(
          "tensormap.replace.mode.global_dim.shared::cta.b1024.b32 [$0], " +
          std::to_string(i) + ", $" + std::to_string(operands.size() - 1) +
          ";");
      operands.push_back(ptxBuilder.newOperand(strides[i], "r"));
      asmStr.append(
          "tensormap.replace.mode.global_stride.shared::cta.b1024.b64 [$0], " +
          std::to_string(i) + ", $" + std::to_string(operands.size() - 1) +
          ";");
      asmStr.append(
          "tensormap.replace.mode.box_dim.shared::cta.b1024.b32 [$0], " +
          std::to_string(i) + ", " + std::to_string(boxDims[i]) + ";");
      asmStr.append("tensormap.replace.mode.element_stride.shared::cta.b1024."
                    "type [$0], " +
                    std::to_string(i) + ", 1;");
    }
    asmStr.append(
        "tensormap.replace.mode.elemtype.shared::cta.b1024.b32 [$0], " +
        std::to_string(getCUtensorMapDataType(elTy)) + ";");
    asmStr.append(
        "tensormap.replace.mode.interleave_layout.shared::cta.b1024.b32 [$0], 0;");
    asmStr.append(
        "tensormap.replace.mode.swizzle_mode.shared::cta.b1024.b32 [$0], 0;");
    asmStr.append(
        "tensormap.replace.mode.fill_mode.shared::cta.b1024.b32 [$0], 0;");        

    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(asmStr);
    ptxInstr(operands, /*onlyAttachMLIRArgs=*/true);
    auto retTy = void_ty(op.getContext());
    auto res = ptxBuilder.launch(rewriter, op.getLoc(), retTy,
                                 /*hasSideEffects*/ true);
    SmallVector<Value> elems;
    for (auto offset : offsets)
      elems.push_back(offset);
    elems.push_back(smemBase);
    auto newValue = getTypeConverter()->packLLElements(
        op.getLoc(), elems, rewriter, result.getType());
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

struct AdvanceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AdvanceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AdvanceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // struct { offset0, offset1, descriptor_shared_memory_address };
    auto loc = op.getLoc();
    auto ptrType = op.getPtr().getType();
    auto tensorPtr = adaptor.getPtr();

    auto offsets = adaptor.getOffsets();
    auto elems =
        getTypeConverter()->unpackLLElements(loc, tensorPtr, rewriter, ptrType);

    SmallVector<Value, 2> newOffsets;

    for (auto [offset, oldOffset] : llvm::zip_first(offsets, elems)) {
      newOffsets.push_back((add(offset, oldOffset)));
    }

    for (size_t i = 0; i < newOffsets.size(); ++i) {
      elems[i] = newOffsets[i];
    }

    auto newValue = getTypeConverter()->packLLElements(op.getLoc(), elems,
                                                       rewriter, ptrType);
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

void populateTensorPtrOpsToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation, PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  patterns.add<AdvanceOpConversion>(typeConverter, benefit);
  return;
}
