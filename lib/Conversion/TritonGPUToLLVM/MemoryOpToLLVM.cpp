#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// blocked -> shared.
// Swizzling in shared memory to avoid bank conflict. Normally used for
// A/B operands of dots.
void lowerDistributedToShared(AllocOp op, AllocOpAdaptor adaptor,
                                       const LLVMTypeConverter *typeConverter,
                                       ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto srcTy = op.getInit().getType();
  auto dstTy = op.getType();
  auto dstShapePerCTA = triton::gpu::getShapePerCTA(dstTy);
  auto srcLayout = srcTy.getEncoding();
  auto outOrd = dstTy.getEncoding().cast<SharedEncodingAttr>().getOrder();
  assert(srcTy.getShape().size() == 2 ||
         (srcTy.getShape().size() <= 3 && outOrd[2] == 0) &&
             "Unexpected rank of ConvertLayout(blocked->shared)");
  Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
  auto elemTy = typeConverter->convertType(srcTy.getElementType());
  auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
  smemBase = bitcast(smemBase, elemPtrTy);

  int32_t elemSize = elemTy.getIntOrFloatBitWidth();
  auto mmaLayout = srcLayout.dyn_cast<NvidiaMmaEncodingAttr>();
  unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
  auto dstStrides =
      LLVM::getStridesFromShapeAndOrder(dstShapePerCTA, outOrd, loc, rewriter);
  auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy, false);
  auto inVals = unpackLLElements(loc, adaptor.getInit(), rewriter);
  storeDistributedToShared(op.getInit(), inVals, dstStrides, srcIndices,
                           op.getResult(), smemBase, elemTy, loc, rewriter);
}

struct AllocOpConversion : public ConvertOpToLLVMPattern<triton::gpu::AllocOp> {
  using ConvertOpToLLVMPattern<triton::gpu::AllocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto resultTy = op.getType().cast<MemDescType>();
    auto typeConverter = getTypeConverter();
    auto sharedLayout =
        resultTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto order = sharedLayout.getOrder();
    // Workaround for 3D tensors
    // TODO: we need to modify the pipeline pass to give a proper shared
    // encoding to 3D tensors
    SmallVector<unsigned> newOrder;
    if (resultTy.getShape().size() != order.size()) {
      for (auto i = 0; i < order.size(); ++i)
        newOrder.push_back(order[i] + 1);
      newOrder.push_back(0);
    } else {
      newOrder = SmallVector<unsigned>(order.begin(), order.end());
    }

    // If there is an initial tensor, store it into the shared memory.
    if (op.getInit()) {
      lowerDistributedToShared(op, adaptor, typeConverter, rewriter);
    }

    auto llvmElemTy = typeConverter->convertType(resultTy.getElementType());
    auto shapePerCTA = getShapePerCTA(sharedLayout, resultTy.getShape());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, shapePerCTA,
                                      newOrder, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct DeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::DeallocOp> {
  using ConvertOpToLLVMPattern<triton::gpu::DeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<AllocOpConversion>(typeConverter, benefit);
  patterns.add<DeallocOpConversion>(typeConverter, benefit);
}
