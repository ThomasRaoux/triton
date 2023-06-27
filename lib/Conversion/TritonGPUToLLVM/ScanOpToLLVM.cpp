#include "ScanOpToLLVM.h"
#include "triton/Analysis/Utility.h"
#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::shflUpSync;

static void accumulate(ConversionPatternRewriter &rewriter, Region &combineOp,
                       Value &acc, Value cur) {
  if (!acc) {
    acc = cur;
    return;
  }
  // Create a new copy of the reduce block, and inline it
  Block *currentBlock = rewriter.getBlock();
  Region &parent = *currentBlock->getParent();
  rewriter.cloneRegionBefore(combineOp, &parent.front());
  auto &newScan = parent.front();
  auto returnOp = dyn_cast<triton::ScanReturnOp>(newScan.getTerminator());
  llvm::SmallVector<Value> combineArgs = {acc, cur};
  rewriter.inlineBlockBefore(&newScan, &*rewriter.getInsertionPoint(),
                             combineArgs);
  auto results = returnOp.getResult();
  acc = results[0];
  // Delete the terminator, which is no longer used
  rewriter.eraseOp(returnOp);
}

// Scan a contiguous elements within a thread and update `srcValues` in place.
static void ScanThreadContiguousElements(SmallVector<Value> &srcValues,
                                         ConversionPatternRewriter &rewriter,
                                         triton::ScanOp op,
                                         BlockedEncodingAttr blockLayout,
                                         unsigned axis) {
  Value acc;
  unsigned contiguousElementsPerThreads = blockLayout.getSizePerThread()[axis];
  for (unsigned i = 0; i < contiguousElementsPerThreads; ++i) {
    accumulate(rewriter, op.getCombineOp(), acc, srcValues[i]);
    srcValues[i] = acc;
  }
}

static unsigned getIntraWarpSizeWithUniqueData(triton::ScanOp op) {
  // TODO: implement helper.
  return 32;
  //auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  //return std::min(srcReduceDimSize,
  //                triton::gpu::getThreadsPerWarpWithUniqueData(
  //                    getSrcLayout(), getSrcShape())[axis]);
}

// Apply a scan acorss threads of the warp for the last element of each
// contiguous group of elements.
static void warpScan(SmallVector<Value> &srcValues,
                     ConversionPatternRewriter &rewriter, triton::ScanOp op,
                     BlockedEncodingAttr blockLayout, unsigned axis,
                     Value laneId) {
  Location loc = op.getLoc();                      
  unsigned contiguousElementsPerThreads = blockLayout.getSizePerThread()[axis];
  Value lastElement = srcValues[contiguousElementsPerThreads - 1];
  Value acc = lastElement;
  unsigned sizeIntraWarps = getIntraWarpSizeWithUniqueData(op);
  // Reduce within warps
  for (unsigned i = 1; i <= sizeIntraWarps / 2; i = i << 1) {
    Value shfl = shflUpSync(loc, rewriter, acc, i);
    accumulate(rewriter, op.getCombineOp(), acc, shfl);
    Value mask = icmp_slt(laneId, i32_val(i));
    lastElement = select(mask, lastElement, acc);
  }
  srcValues[contiguousElementsPerThreads - 1] = lastElement;
}

namespace {
struct ScanOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::ScanOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::ScanOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (succeeded(emitFastScan(op, adaptor, rewriter)))
      return success();
    return failure();
  }
private:
  LogicalResult emitFastScan(triton::ScanOp op,
                                  triton::ScanOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const;
};

// Naive lowering of the scan op as a fallback for cases that we don't know
// how to generate with warp shuffle ops.
LogicalResult
ScanOpConversion::emitFastScan(triton::ScanOp op, triton::ScanOpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto input = adaptor.getOperands()[0];
  unsigned axis = op.getAxis();
  auto type = input.getType().cast<RankedTensorType>();
  auto srcEncoding = type.getEncoding();
  auto blockLayout = srcEncoding.dyn_cast<BlockedEncodingAttr>();
  if (axis != triton::gpu::getOrder(srcEncoding)[0] || !blockLayout)
    return failure();

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(32);
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);
  Value result = input;

  SmallVector<Value> srcValues =
      getTypeConverter()->unpackLLElements(loc, input, rewriter, type);

  // Scan contigous elements in a thread and update `srcValues`.
  ScanThreadContiguousElements(srcValues, rewriter, op, blockLayout, axis);
  warpScan(srcValues, rewriter, op, blockLayout, axis, laneId);
  barrier();
  // AccumulateOffsets(rewriter, op, input, axis, type);
  // AddOffsets(rewriter, op, input, axis, type);
  // UpdateWithLastContiguousElement(rewriter, op, input, axis, type);

  return success();
}
} // namespace

void populateScanOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, allocation, indexCacheInfo,
                                 benefit);
}
