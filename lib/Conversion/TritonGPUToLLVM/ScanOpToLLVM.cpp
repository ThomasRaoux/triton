#include "ScanOpToLLVM.h"
#include "triton/Analysis/Utility.h"
#include "TritonGPUToLLVMBase.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::shflUpSync;
using ::mlir::LLVM::storeShared;

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
static void scanThreadContiguousElements(SmallVector<Value> &srcValues,
                                         ConversionPatternRewriter &rewriter,
                                         triton::ScanOp op,
                                         BlockedEncodingAttr blockLayout,
                                         unsigned axis) {

  unsigned contiguousElementsPerThreads = blockLayout.getSizePerThread()[axis];
  // Loop through the chunks of contiguous elements.
  for (unsigned j = 0; j < srcValues.size();
       j += contiguousElementsPerThreads) {
    Value acc;
    // Loop through the contiguous elements.
    for (unsigned i = 0; i < contiguousElementsPerThreads; ++i) {
      accumulate(rewriter, op.getCombineOp(), acc, srcValues[i + j]);
      srcValues[i + j] = acc;
    }
  }
}

static unsigned getScanDimSizePerWarp(triton::ScanOp op) {
  unsigned axis = op.getAxis();
  auto type = op.getOperand(0).getType().cast<RankedTensorType>();
  auto srcEncoding = type.getEncoding();
  return triton::gpu::getThreadsPerWarpWithUniqueData(
                      srcEncoding, type.getShape())[axis];
}

// Apply a scan acorss threads of the warp for the last element of each
// contiguous group of elements.
static void warpScan(SmallVector<Value> &srcValues,
                     ConversionPatternRewriter &rewriter, triton::ScanOp op,
                     BlockedEncodingAttr blockLayout, unsigned axis,
                     Value laneId) {
  Location loc = op.getLoc();
  unsigned contiguousElementsPerThreads = blockLayout.getSizePerThread()[axis];
  for (unsigned j = contiguousElementsPerThreads - 1; j < srcValues.size();
       j += contiguousElementsPerThreads) {
    Value acc = srcValues[j];
    unsigned scanDim = getScanDimSizePerWarp(op);
    // Reduce within warps
    for (unsigned i = 1; i <= scanDim / 2; i = i << 1) {
      Value shfl = shflUpSync(loc, rewriter, acc, i);
      Value tempAcc = acc;
      accumulate(rewriter, op.getCombineOp(), tempAcc, shfl);
      Value mask = icmp_slt(laneId, i32_val(i));
      acc = select(mask, acc, tempAcc);
    }
    srcValues[j] = acc;
  }
}

static void storeWarpAccumulator(SmallVector<Value> &srcValues,
                                 ConversionPatternRewriter &rewriter,
                                 triton::ScanOp op,
                                 BlockedEncodingAttr blockLayout, unsigned axis,
                                 Value laneId, Value warpId, unsigned numWarps,
                                 Value baseSharedMemPtr) {
  Location loc = op.getLoc();
  unsigned contiguousElementsPerThreads = blockLayout.getSizePerThread()[axis];
  unsigned scanDim = getScanDimSizePerWarp(op);
  unsigned chunkId = 0;
  for (unsigned j = contiguousElementsPerThreads - 1; j < srcValues.size();
       j += contiguousElementsPerThreads, ++chunkId) {
    Value lastElement = srcValues[j];
    Value mask = icmp_eq(laneId, i32_val(scanDim - 1));
    Value index = add(warpId, i32_val(chunkId * numWarps));
    Value writePtr = gep(baseSharedMemPtr.getType(), baseSharedMemPtr, index);
    storeShared(rewriter, loc, writePtr, lastElement, mask);
  }
}

static void AddPartialReduce(SmallVector<Value> &srcValues,
                                           ConversionPatternRewriter &rewriter,
                                           triton::ScanOp op,
                                           BlockedEncodingAttr blockLayout,
                                           unsigned axis, Value sharedMemoryPtr,
                                           Value warpId, unsigned numWarps, Value laneId) {
  Location loc = op.getLoc();
  Value acc;
  Value maskedAcc;
  unsigned contiguousElementsPerThreads = blockLayout.getSizePerThread()[axis];
  Value maskFirstWarp = icmp_eq(warpId, i32_val(0));
  Value maskFirstLane = icmp_eq(laneId, i32_val(0));
  Value maskFirstThread = and_(maskFirstWarp, maskFirstLane);
  unsigned chunkId = 0;
  for (unsigned j = contiguousElementsPerThreads - 1; j < srcValues.size();
       j += contiguousElementsPerThreads, ++chunkId) {
    for (unsigned i = 0; i < numWarps; ++i) {
      Value ptr = gep(sharedMemoryPtr.getType(), sharedMemoryPtr,
                      i32_val(i + chunkId * numWarps));
      Value partialReduce = load(ptr);
      if (!acc) {
        acc = partialReduce;
        maskedAcc = partialReduce;
        continue;
      }
      accumulate(rewriter, op.getCombineOp(), acc, partialReduce);
      Value mask = icmp_slt(warpId, i32_val(i+1));
      maskedAcc = select(mask, maskedAcc, acc);
    }
    Value temp = srcValues[j];
    accumulate(rewriter, op.getCombineOp(), temp, maskedAcc);
    if (chunkId == 0) {
      // For the first warp and first chunk we don't have anything to
      // accumulate.
      temp = select(maskFirstWarp, srcValues[j], temp);
    }
    srcValues[j] = temp;

    // Update the rest of the contiguous elements.
    Value lastElement = shflUpSync(loc, rewriter, srcValues[j], 1);
    lastElement = select(maskFirstLane, maskedAcc, lastElement);
    for(unsigned i = 1; i < contiguousElementsPerThreads; ++i) {
      Value laneValue = srcValues[j - i];
      accumulate(rewriter, op.getCombineOp(), laneValue, lastElement);
      if (chunkId == 0) {
      // For the first warp and first chunk we don't have anything to
      // accumulate.
        laneValue = select(maskFirstThread, srcValues[j - i], laneValue);
      }
      srcValues[j - i] = laneValue;
    }
    // For the next chunk start back from the value containing the accumulated
    // value of all the warps.
    maskedAcc = acc;
  }
}

static void UpdateWithLastContiguousElement(SmallVector<Value> &srcValues,
                                            ConversionPatternRewriter &rewriter,
                                            triton::ScanOp op,
                                            BlockedEncodingAttr blockLayout,
                                            unsigned axis, Value laneId) {
  Location loc = op.getLoc();
  unsigned contiguousElementsPerThreads = blockLayout.getSizePerThread()[axis];
  for (unsigned j = contiguousElementsPerThreads - 1; j < srcValues.size();
       j += contiguousElementsPerThreads) {
    Value tmp = shflUpSync(loc, rewriter, srcValues[j], 1);
    for (unsigned i = 1; i < contiguousElementsPerThreads; ++i) {
      Value acc = srcValues[j - i];
      accumulate(rewriter, op.getCombineOp(), acc, tmp);
      Value mask = icmp_eq(laneId, i32_val(0));
      srcValues[j - i] = select(mask, srcValues[j - i], acc);
    }
  }
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
  auto type = op.getOperand(0).getType().cast<RankedTensorType>();
  auto llvmType = input.getType();
  auto srcEncoding = type.getEncoding();
  auto blockLayout = srcEncoding.dyn_cast<BlockedEncodingAttr>();
  if (axis != triton::gpu::getOrder(srcEncoding)[0] || !blockLayout)
    return failure();

  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(32);
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
  auto order = triton::gpu::getOrder(srcEncoding);
  SmallVector<Value> multiDimLaneId =
      delinearize(rewriter, loc, laneId, threadsPerWarp, order);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  Value laneIdAxis = multiDimLaneId[axis];
  Value warpIdAxis = multiDimWarpId[axis];

  Value result = input;

  SmallVector<Value> srcValues =
      getTypeConverter()->unpackLLElements(loc, input, rewriter, type);

  // Scan contigous elements in a thread and update `srcValues`.
  scanThreadContiguousElements(srcValues, rewriter, op, blockLayout, axis);
  warpScan(srcValues, rewriter, op, blockLayout, axis, laneIdAxis);
  Type elemPtrTys = LLVM::LLVMPointerType::get(srcValues[0].getType(), 3);
  Value baseSharedMemPtr = bitcast(
      getSharedMemoryBase(loc, rewriter, op.getOperation()), elemPtrTys);
  storeWarpAccumulator(srcValues, rewriter, op, blockLayout, axis, laneIdAxis,
                       warpIdAxis, warpsPerCTA[axis], baseSharedMemPtr);
  barrier();
  AddPartialReduce(srcValues,
      rewriter, op, blockLayout, axis, baseSharedMemPtr, warpIdAxis, warpsPerCTA[axis], laneIdAxis);
  //UpdateWithLastContiguousElement(srcValues, rewriter, op, blockLayout, axis, laneIdAxis);

  Value results = getTypeConverter()->packLLElements(loc, srcValues, rewriter, llvmType);
  rewriter.replaceOp(op, results);
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
