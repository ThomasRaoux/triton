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
    //PTXBuilder ptxBuilder;
    //SmallVector<PTXBuilder::Operand *> outputsAndOperands;
    //  outputsAndOperands.append(ptxOperands.begin(), ptxOperands.end());
//    std::string asmStr = R("
//    tensormap.replace.mode.global_address.shared::cta.b1024.type [$0], $1;
//    tensormap.replace.mode.global_address.shared::cta.b1024.type [$0], " + shapes.size() - 1 + ";
//    tensormap.replace.mode.global_address.shared::cta.b1024.type [$0], $3;
//    ");
//    auto &ptxInstr = *ptxBuilder.create<PTXInstr>(asmStr);
//    ptxInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
    //auto retTy = void_ty(op.getContext());
    //.auto res = ptxBuilder.launch(rewriter, op.getLoc(), retTy,
     //                            /*hasSideEffects*/ true);
    SmallVector<Value> elems;
    for (auto offset : offsets)
      elems.push_back(offset);

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
