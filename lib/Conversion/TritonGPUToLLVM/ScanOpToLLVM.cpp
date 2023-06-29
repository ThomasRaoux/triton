#include "ScanOpToLLVM.h"
#include "TritonGPUToLLVMBase.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::shflUpSync;
using ::mlir::LLVM::storeShared;

namespace {
class ScanLoweringHelper {
public:
  explicit ScanLoweringHelper(triton::ScanOp op) : scanOp(op) {}
  Location getLoc() { return scanOp.getLoc(); }
  unsigned getAxis() { return scanOp.getAxis(); }
  unsigned scanElementsPerThreads() {
    return getEncoding().getSizePerThread()[getAxis()];
  }
  unsigned parallelElementsPerThread() {
    SmallVector<unsigned> sizePerThreads(getEncoding().getSizePerThread().begin(),
                                         getEncoding().getSizePerThread().end());
    sizePerThreads[getAxis()] = 1;
    return product<unsigned>(sizePerThreads);
  }
  Region &getCombineOp() { return scanOp.getCombineOp(); }
  unsigned getScanDimSizePerWarp() {
    auto type = scanOp.getOperand(0).getType().cast<RankedTensorType>();
    return triton::gpu::getThreadsPerWarpWithUniqueData(
        getEncoding(), type.getShape())[getAxis()];
  }

  unsigned getNumParrallelThreadsPerWarp() {
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(getEncoding());
    threadsPerWarp[getAxis()] = 1;
    return product<unsigned>(threadsPerWarp);
  }

  // Return the flat numbers of threads computing independent scan results.
  unsigned getNumParrallelThreadsPerCTA() {
    unsigned numParallelThreadsPerWarp = getNumParrallelThreadsPerWarp();
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(getEncoding());
    warpsPerCTA[getAxis()] = 1;
    unsigned numParallelWarpsPerCTA = product<unsigned>(warpsPerCTA);
    return numParallelThreadsPerWarp * numParallelWarpsPerCTA;
  }
  unsigned getScanNumWarps() {
    auto type = scanOp.getOperand(0).getType().cast<RankedTensorType>();
    auto srcEncoding = type.getEncoding();
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
    return warpsPerCTA[getAxis()];
  }

  unsigned getNumScanBlocks() {
    auto type = scanOp.getOperand(0).getType().cast<RankedTensorType>();
    auto srcEncoding = type.getEncoding();
    auto sizePerThreads = triton::gpu::getSizePerThread(srcEncoding);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
    unsigned axis = getAxis();
    return type.getShape()[axis] /
           (sizePerThreads[axis] * threadsPerWarp[axis] * warpsPerCTA[axis]);
  }

  unsigned getNumParallelBlocks() {
    auto type = scanOp.getOperand(0).getType().cast<RankedTensorType>();
    auto srcEncoding = type.getEncoding();
    auto sizePerThreads = triton::gpu::getSizePerThread(srcEncoding);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcEncoding);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(srcEncoding);
    unsigned axis = getAxis();
    unsigned numBlocks = 1;
    for(unsigned i =0; i < sizePerThreads.size(); i++) {
      if(i == axis) continue;
      numBlocks *= type.getShape()[i] /
           (sizePerThreads[i] * threadsPerWarp[i] * warpsPerCTA[i]);
    }
    return numBlocks;
  }

  bool isSupported() {
    auto type = scanOp.getOperand(0).getType().cast<RankedTensorType>();
    auto srcEncoding = type.getEncoding();
    if (getAxis() != triton::gpu::getOrder(srcEncoding)[0] ||
        !isa<BlockedEncodingAttr>(srcEncoding))
      return false;
    return true;
  }

private:
  BlockedEncodingAttr getEncoding() {
    auto type = scanOp.getOperand(0).getType().cast<RankedTensorType>();
    auto srcEncoding = type.getEncoding();
    return srcEncoding.cast<BlockedEncodingAttr>();
  }
  triton::ScanOp scanOp;
};
} // namespace

// Apply the region of the scan op to the acc and cur values and update acc
// inplace with the result.
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
                                         ScanLoweringHelper &helper) {
  // TODO: this assumes that axis is the fastest moving dimension. We should
  // relax that.
  unsigned scanElementsPerThreads = helper.scanElementsPerThreads();
  // Loop through the blocks of contiguous elements.
  for (unsigned j = 0; j < srcValues.size(); j += scanElementsPerThreads) {
    // Reset the accumulator at the beginning of each block of contiguous
    // elements.
    Value acc;
    // Loop through the contiguous elements.
    for (unsigned i = 0; i < scanElementsPerThreads; ++i) {
      accumulate(rewriter, helper.getCombineOp(), acc, srcValues[i + j]);
      srcValues[i + j] = acc;
    }
  }
}

// Apply a scan across threads of the warp for the last element of each
// contiguous group of elements.
static void warpScan(SmallVector<Value> &srcValues,
                     ConversionPatternRewriter &rewriter,
                     ScanLoweringHelper &helper, Value laneId) {
  Location loc = helper.getLoc();
  unsigned scanElementsPerThreads = helper.scanElementsPerThreads();
  for (unsigned j = scanElementsPerThreads - 1; j < srcValues.size();
       j += scanElementsPerThreads) {
    Value acc = srcValues[j];
    unsigned scanDim = helper.getScanDimSizePerWarp();
    // Reduce within warps.
    for (unsigned i = 1; i <= scanDim / 2; i = i << 1) {
      Value shfl = shflUpSync(loc, rewriter, acc, i);
      Value tempAcc = acc;
      accumulate(rewriter, helper.getCombineOp(), tempAcc, shfl);
      Value mask = icmp_slt(laneId, i32_val(i));
      acc = select(mask, acc, tempAcc);
    }
    srcValues[j] = acc;
  }
}

// Shared memory will contain the partial reduction value for each parallel
// scans and for each warp for each contiguous elements within a thread.
//          -----------------------------------------------------------------
// chunk 0: | scan 0 warp 0 | scan 1 warp 0 | scan 0 warp 1 | scan 1 warp 1 |
// chunk 1: | scan 0 warp 0 | scan 1 warp 0 | scan 0 warp 1 | scan 1 warp 1 |
static void storeWarpAccumulator(SmallVector<Value> &srcValues,
                                 ConversionPatternRewriter &rewriter,
                                 ScanLoweringHelper &helper, Value laneId,
                                 Value warpId, Value baseSharedMemPtr,
                                 Value parallelLaneId) {
  Location loc = helper.getLoc();
  unsigned scanElementsPerThreads = helper.scanElementsPerThreads();
  unsigned scanDim = helper.getScanDimSizePerWarp();
  unsigned numParallelLane = helper.getNumParrallelThreadsPerCTA();
  llvm::dbgs() << "numParallelLane: " << numParallelLane << "\n";
  unsigned numWarps = helper.getScanNumWarps();
  unsigned chunkId = 0;
  for (unsigned j = scanElementsPerThreads - 1; j < srcValues.size();
       j += scanElementsPerThreads, ++chunkId) {
    Value lastElement = srcValues[j];
    Value mask = icmp_eq(laneId, i32_val(scanDim - 1));
    Value index = add(parallelLaneId, mul(warpId, i32_val(numParallelLane)));
    index = add(index, i32_val(chunkId * numParallelLane * numWarps));
    Value writePtr = gep(baseSharedMemPtr.getType(), baseSharedMemPtr, index);
    storeShared(rewriter, loc, writePtr, lastElement, mask);
  }
}

// Read the partial reductions from shared memory from each warp and chunks and
// combine them with the right elements.
static void AddPartialReduce(SmallVector<Value> &srcValues,
                             ConversionPatternRewriter &rewriter,
                             ScanLoweringHelper &helper, Value sharedMemoryPtr,
                             Value warpId, Value laneId, Value parallelLaneId) {
  Location loc = helper.getLoc();
  unsigned numParallelLane = helper.getNumParrallelThreadsPerCTA();
  unsigned numWarps = helper.getScanNumWarps();
  unsigned scanElementsPerThreads = helper.scanElementsPerThreads();
  unsigned parallelElementsPerThread = helper.parallelElementsPerThread();
  Value maskFirstWarp = icmp_eq(warpId, i32_val(0));
  Value maskFirstLane = icmp_eq(laneId, i32_val(0));
  Value maskFirstThread = and_(maskFirstWarp, maskFirstLane);
  struct Accumulator {
    Value acc;
    Value maskedAcc;
  };
  unsigned numScanBlocks = helper.getNumScanBlocks();
  unsigned numParallelBlocks = helper.getNumParallelBlocks();
  assert(numScanBlocks * numParallelBlocks * parallelElementsPerThread * scanElementsPerThreads == srcValues.size());
  SmallVector<Accumulator> accumulators(numParallelBlocks * parallelElementsPerThread);
  unsigned chunkId = 0;
  for(unsigned parallelBlockId = 0; parallelBlockId < numParallelBlocks; ++parallelBlockId ) {
    for(unsigned scanBlockId = 0; scanBlockId < numScanBlocks; ++scanBlockId) {
      for(unsigned parallelElementId = 0; parallelElementId < parallelElementsPerThread; ++parallelElementId) {
        unsigned accumulatorIndex = parallelElementId + parallelBlockId * parallelElementsPerThread;
        Accumulator &accumulator = accumulators[accumulatorIndex];
        for (unsigned i = 0; i < numWarps; ++i) {
          Value index = add(parallelLaneId, i32_val(numParallelLane *
                                                    (i + chunkId * numWarps)));
          Value ptr = gep(sharedMemoryPtr.getType(), sharedMemoryPtr, index);
          Value partialReduce = load(ptr);
          if (!accumulator.acc) {
            accumulator.acc = partialReduce;
            accumulator.maskedAcc = partialReduce;
            continue;
          }
          accumulate(rewriter, helper.getCombineOp(), accumulator.acc,
                     partialReduce);
          Value mask = icmp_slt(warpId, i32_val(i + 1));
          accumulator.maskedAcc =
              select(mask, accumulator.maskedAcc, accumulator.acc);
        }
        unsigned lastElementIndex =
            chunkId * scanElementsPerThreads + scanElementsPerThreads - 1;
        Value temp = srcValues[lastElementIndex];
        accumulate(rewriter, helper.getCombineOp(), temp,
                   accumulator.maskedAcc);
        if (scanBlockId == 0) {
          // For the first warp and first chunk we don't have anything to
          // accumulate.
          temp = select(maskFirstWarp, srcValues[lastElementIndex], temp);
        }
        srcValues[lastElementIndex] = temp;

        // Update the rest of the contiguous elements.
        Value lastElement =
            shflUpSync(loc, rewriter, srcValues[lastElementIndex], 1);
        lastElement = select(maskFirstLane, accumulator.maskedAcc, lastElement);
        for (unsigned i = 1; i < scanElementsPerThreads; ++i) {
          Value laneValue = srcValues[lastElementIndex - i];
          accumulate(rewriter, helper.getCombineOp(), laneValue, lastElement);
          if (scanBlockId == 0) {
            // For the first warp and first chunk we don't have anything to
            // accumulate.
            laneValue = select(maskFirstThread, srcValues[lastElementIndex - i],
                               laneValue);
          }
          srcValues[lastElementIndex - i] = laneValue;
        }
        // For the next chunk start back from the value containing the
        // accumulated value of all the warps.
        accumulator.maskedAcc = accumulator.acc;
        chunkId++;
      }
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
  LogicalResult emitFastScan(triton::ScanOp op, triton::ScanOpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const;
};

// Naive lowering of the scan op as a fallback for cases that we don't know
// how to generate with warp shuffle ops.
LogicalResult
ScanOpConversion::emitFastScan(triton::ScanOp op, triton::ScanOpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  ScanLoweringHelper helper(op);
  auto loc = op.getLoc();
  auto input = adaptor.getOperands()[0];
  unsigned axis = op.getAxis();
  auto type = op.getOperand(0).getType().cast<RankedTensorType>();
  auto srcEncoding = type.getEncoding();

  if (!helper.isSupported())
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

  multiDimLaneId[axis] = i32_val(0);
  threadsPerWarp[axis] = 1;
  Value laneIdParallel =
      linearize(rewriter, loc, multiDimLaneId, threadsPerWarp, order);
  multiDimWarpId[axis] = i32_val(0);
  warpsPerCTA[axis] = 1;
  Value warpIdParallel =
      linearize(rewriter, loc, multiDimWarpId, warpsPerCTA, order);
  Value flatIdParallel =
      add(laneIdParallel,
          mul(warpIdParallel, i32_val(helper.getNumParrallelThreadsPerWarp())));

  SmallVector<Value> srcValues =
      getTypeConverter()->unpackLLElements(loc, input, rewriter, type);

  // Scan contigous elements in a thread and update `srcValues`.
  scanThreadContiguousElements(srcValues, rewriter, helper);
  // Apply warp level scan to the last element of each chunk of contiguous
  // elements.
  warpScan(srcValues, rewriter, helper, laneIdAxis);

  // Store the partial reducing for each warp into shared memory.
  Type elemPtrTys = LLVM::LLVMPointerType::get(srcValues[0].getType(), 3);
  Value baseSharedMemPtr = bitcast(
      getSharedMemoryBase(loc, rewriter, op.getOperation()), elemPtrTys);
  storeWarpAccumulator(srcValues, rewriter, helper, laneIdAxis, warpIdAxis,
                       baseSharedMemPtr, flatIdParallel);
  barrier();
  // Read back the partial reduction of each warp and accumulate them based on
  // warpId. Then update each chunk of contiguous elements by adding the
  // accumulated value from the previous lane.
  AddPartialReduce(srcValues, rewriter, helper, baseSharedMemPtr, warpIdAxis,
                   laneIdAxis, flatIdParallel);

  Value results =
      getTypeConverter()->packLLElements(loc, srcValues, rewriter, input.getType());
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
