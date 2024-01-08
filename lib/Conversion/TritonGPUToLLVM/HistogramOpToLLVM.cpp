#include "HistogramOpToLLVM.h"
#include "TritonGPUToLLVMBase.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton;

static int log2Int(int64_t num) { return (num > 1) ? 1 + log2Int(num / 2) : 0; }

// Compute a histogram within a warp. This uses an algorithm by @apgoucher
// that does the following:
// Create a ballot for each bit of the bin index (there
// are only log2(num_bins) of these) and then apply bitwise operations to get
// the indicator functions for the bins owned by this particular thread, and
// only popcount those.
static SmallVector<Value>
computeWarpLevelHistogram(Location loc, SmallVector<Value> srcValues,
                          int numBins, int numThreadPerWarp, Value threadId,
                          ConversionPatternRewriter &rewriter) {
  assert(numBins % numThreadPerWarp == 0 &&
         "numBins must be divisible by numThreadPerWarp");
  Value zero = i32_val(0);
  int numBits = log2Int(numBins);
  int numBitsLaneId = log2Int(numThreadPerWarp);
  SmallVector<Value> warpLevelHistogram(numBins / numThreadPerWarp, zero);
  for (Value value : srcValues) {
    SmallVector<Value> ballotBits;
    for (int i = 0; i < numBits; ++i) {
      Value bitSet = and_(value, i32_val(1 << i));
      Value threadMask = i32_val(-1);
      Value bit = rewriter.create<NVVM::VoteBallotOp>(loc, i32_ty, threadMask,
                                                      icmp_ne(bitSet, zero));
      ballotBits.push_back(bit);
    }
    Value fullMask = i32_val(0xFFFFFFFF);
    Value mask = fullMask;
    for (int i = 0; i < numBitsLaneId; i++) {
      Value updateMask = select(icmp_ne(and_(threadId, i32_val(1 << i)), zero),
                                zero, fullMask);
      mask =
          and_(mask, xor_(ballotBits[i + numBits - numBitsLaneId], updateMask));
    }
    // at this point, 'mask' tells you which elements are in a bin owned by this thread.
    for (int k =0; k < warpLevelHistogram.size(); k++) {
        Value binMask = mask;
        for (int j = 0; j < numBits - numBitsLaneId; j++) {
            Value updateMask = i32_val(((k & (1 << j)) ? 0 : 0xffffffff));
            binMask = and_(binMask, xor_(ballotBits[j], updateMask));
        }
        // at this point, 'bin_mask' tells you which elements are in the kth bin
        // owned by this thread.
        Value bitCount = rewriter.create<LLVM::CtPopOp>(loc, i32_ty, binMask);
        warpLevelHistogram[k] = add(warpLevelHistogram[k], bitCount);
    }
  }
  return warpLevelHistogram;
}

namespace {
struct HistogramOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::HistogramOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::HistogramOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    SmallVector<Value> srcValues =
        getTypeConverter()->unpackLLElements(loc, input, rewriter);
    int numBins =
        op.getResult().getType().cast<RankedTensorType>().getDimSize(0);
    int numThreadsPerWarp = 32;
    Value threadId = getThreadId(rewriter, loc);
    // First compute a warp local histogram based on values owned by each warps.
    SmallVector<Value> warpLevelHistogram = computeWarpLevelHistogram(
        loc, srcValues, numBins, numThreadsPerWarp, threadId, rewriter);
    
    // Then use atomic to update the histogram in shared memory.

    Value results = getTypeConverter()->packLLElements(
        loc, warpLevelHistogram, rewriter, op.getResult().getType());
    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

void populateHistogramOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<HistogramOpConversion>(typeConverter, allocation, indexCacheInfo,
                                      benefit);
}
