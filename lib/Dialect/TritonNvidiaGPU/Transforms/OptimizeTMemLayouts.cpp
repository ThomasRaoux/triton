#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include <cstdint>

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZETMEMLAYOUTSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// clang-format off
// Converts:
//  %l  = ttng.tmem_load  %o : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
//                               -> tensor<128x256xf32, #blocked>
//  %r  = tt.reshape %l  : tensor<128x256xf32, #blocked>
//                               -> tensor<128x2x128xf32, #blocked4>
//  %t  = tt.trans   %r  {order = array<i32: 0, 2, 1>}
//                               -> tensor<128x128x2xf32, #blocked5>
//  %lhs, %rhs = tt.split %t
//
// becomes
//  %o0   = ttng.tmem_subslice %o { N = 0   }
//  %lhs  = ttng.tmem_load     %o0
//  %o1   = ttng.tmem_subslice %o { N = 128 }
//  %rhs  = ttng.tmem_load     %o1
//
// and if %lhs / %rhs are split again through the same reshape->trans->split
// pattern, the transformation is can match again so that each further
// split is materialised as an independent `ttng.tmem_subslice` / `ttng.tmem_load`
// pair.  Consequently, a chain such as
//
//   acc0, acc1  = split(permute(reshape(acc , ...)))
//   acc00, acc01 = split(permute(reshape(acc0, ...)))
//   acc10, acc11 = split(permute(reshape(acc1, ...)))
//
// is lowered to four independent TMEM loads operating on four disjoint
// subslices.
//
// clang-format on
// Strip away all intermediate ttg.convert_layout ops to reach the true
// producer.
static Value stripConvertLayout(Value v) {
  while (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

class TMemSplitLoadPattern : public OpRewritePattern<SplitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    // -----------------------------------------------------------------------
    // Match the pattern:
    //      splitOp
    //        ^  |
    //        |  +-- transOp(order = [0, 2, 1])
    //        |       ^  |
    //        |       |  +-- reshapeOp
    //        |       |        ^  |
    //        |       |        |  +-- (maybe convert_layout)
    //        |       |        +-- tmemLoad
    // -----------------------------------------------------------------------

    // Starting from the split source, peel off convert_layouts if any.
    Value src = stripConvertLayout(splitOp.getSrc());
    auto transOp = src.getDefiningOp<TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto reshapeOp = transOp.getSrc().getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();

    // Peel off convert_layouts *below* the reshape as well.  This is required
    // for the recursive case where the producer of the reshape is the result
    // of an earlier optimisation pass (i.e. a convert_layout of a previous
    // tmem_load).
    Value reshapeSrc = stripConvertLayout(reshapeOp.getSrc());
    auto tmemLoad = reshapeSrc.getDefiningOp<TMEMLoadOp>();
    if (!tmemLoad)
      return failure();

    auto shape = reshapeOp.getResult().getType().getShape();
    // Ensure M dimension is preserved by the reshape.
    if (shape[0] != cast<RankedTensorType>(reshapeSrc.getType()).getShape()[0])
      return failure();
    int mDim = getShapePerCTA(tmemLoad.getSrc().getType())[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    // Create the two TMEM subslices and their corresponding loads.
    Value tmem = tmemLoad.getSrc(); // Could itself be a subslice.
    int numWarps = ttg::lookupNumWarps(tmemLoad);
    rewriter.setInsertionPoint(tmemLoad);

    auto createSliceLoad =
        [&](int64_t nOffset) -> std::pair<TMEMLoadOp, ttg::ConvertLayoutOp> {
      // Generate the subslice op.
      Value subSlice = rewriter.create<TMEMSubSliceOp>(tmemLoad.getLoc(), tmem,
                                                       nOffset, splitNSize);

      // Choose a layout compatible with the slice size.
      Attribute distLayout = getTmemCompatibleLayout(
          mDim, splitNSize, splitOp.getOutLHS().getType(), numWarps);

      RankedTensorType newLoadType =
          splitOp.getOutLHS().getType().cloneWithEncoding(distLayout);

      // Generate the load and convert_layout back to the original layout.
      auto load =
          rewriter.create<TMEMLoadOp>(tmemLoad.getLoc(), newLoadType, subSlice);
      auto cvt = rewriter.create<ttg::ConvertLayoutOp>(
          tmemLoad.getLoc(), splitOp.getOutLHS().getType(), load);

      return {load, cvt};
    };

    auto [load0, cvt0] = createSliceLoad(/*nOffset=*/0);
    auto [load1, cvt1] = createSliceLoad(/*nOffset=*/splitNSize);
    rewriter.replaceOp(splitOp, {cvt0, cvt1});
    return success();
  }
};

class TMemStoreJoinPattern : public OpRewritePattern<TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMStoreOp storeOp,
                                PatternRewriter &b) const override {
    // Look through layout conversions.
    Value src = storeOp.getSrc();
    while (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>()) {
      src = cvt.getSrc();
    }

    // Only support joinin N dimension on the outer most.
    auto reshapeOp = src.getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto shape = reshapeOp.getSrc().getType().getShape();
    if (reshapeOp.getType().getShape().front() != shape[0])
      return failure();
    auto transOp = reshapeOp.getSrc().getDefiningOp<TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto joinOp = transOp.getSrc().getDefiningOp<JoinOp>();
    if (!joinOp)
      return failure();

    // We found a tmem_store that is joined on the N dimension. We can split it
    // into multiple tmem_stores.
    int mDim = getShapePerCTA(storeOp.getDst().getType())[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    Location loc = storeOp.getLoc();
    Value tmem = storeOp.getDst();
    int numWarps = ttg::lookupNumWarps(storeOp);
    Value truePred = b.create<arith::ConstantOp>(loc, b.getBoolAttr(true));

    Attribute distLayout = getTmemCompatibleLayout(
        mDim, splitNSize, joinOp.getLhs().getType(), numWarps);
    auto newStoreType = joinOp.getLhs().getType().cloneWithEncoding(distLayout);

    // First slice.
    auto subSlice0 = b.create<TMEMSubSliceOp>(loc, tmem, 0, splitNSize);
    auto cvt0 =
        b.create<ttg::ConvertLayoutOp>(loc, newStoreType, joinOp.getLhs());
    auto store0 =
        b.create<TMEMStoreOp>(loc, subSlice0, cvt0.getResult(), truePred);
    // Second slice.
    auto subSlice1 =
        b.create<TMEMSubSliceOp>(loc, tmem, splitNSize, splitNSize);
    auto cvt1 =
        b.create<ttg::ConvertLayoutOp>(loc, newStoreType, joinOp.getRhs());
    auto store1 =
        b.create<TMEMStoreOp>(loc, subSlice1, cvt1.getResult(), truePred);
    b.eraseOp(storeOp);
    return success();
  }
};

// Pick an optimized tmem load layout based on its users. When there are
// multiple warpgroups tmem_load results can be distirbuted along M or N across
// the warpgroups. By default distribute along N but when there is a reduction
// along N dimension we want to distribute along M instead to avoid having to
// reduce across warps.
class TMemLoadReducePattern : public OpRewritePattern<TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMLoadOp tmemLoadOp,
                                PatternRewriter &rewriter) const override {
    int numWarps = ttg::lookupNumWarps(tmemLoadOp);
    // If there is only 1 warpgroup there is nothing to optimize as the layout
    // is already reduction friendly.
    if (numWarps != 8)
      return failure();
    auto tmemEnc = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
        tmemLoadOp.getSrc().getType().getEncoding());
    if (!tmemEnc)
      return failure();
    int M = tmemEnc.getBlockM();
    int N = tmemEnc.getBlockN();
    if (M != 128)
      return failure();
    bool foundReductionAlongN = false;
    auto filter = [&](Operation *op) {
      if (isa<ttg::ConvertLayoutOp>(op) || op->hasTrait<OpTrait::Elementwise>())
        return true;
      if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
        foundReductionAlongN = reduce.getAxis() == 1;
      }
      return false;
    };
    ForwardSliceOptions fwdOpt;
    fwdOpt.filter = filter;
    SetVector<mlir::Operation *> fwdSlices;
    getForwardSlice(tmemLoadOp.getResult(), &fwdSlices, fwdOpt);
    if (!foundReductionAlongN)
      return failure();
    // Try to split along M dimension but follow the restrictions of TMEM:
    // warp0 get M = 0, warp 1 gets M = 32, warp 2 gets M = 64, warp 3 gets
    // M = 96 warp 4 gets M = 16, warp 5 gets M = 48, warp 6 gets M = 80,
    // warp 7 gets M = 112
    RankedTensorType oldType = tmemLoadOp.getType();
    Attribute newLayout = ttg::LinearEncodingAttr::get(
        tmemLoadOp.getContext(),
        ttg::getTmemLoadLayoutSplitLongM(M, N, oldType, numWarps));
    if (newLayout == oldType.getEncoding())
      return failure();

    auto newType = oldType.cloneWithEncoding(newLayout);
    tmemLoadOp.getResult().setType(newType);
    OpBuilder builder(tmemLoadOp);
    builder.setInsertionPointAfter(tmemLoadOp);
    auto cvt = builder.create<ttg::ConvertLayoutOp>(
        tmemLoadOp.getLoc(), oldType, tmemLoadOp.getResult());
    tmemLoadOp.getResult().replaceAllUsesExcept(cvt.getResult(), cvt);
    return success();
  }
};

// TODO: Smem spliting could be fully generalized using linear layout encoding
// for smem memdesc. This is a stop gap solution to help use smem and tmem
// together but should be replaced by a generic solution.
class SMemSplitLoadPattern : public OpRewritePattern<SplitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    // -----------------------------------------------------------------------
    // Match the pattern:
    //      splitOp
    //        ^  |
    //        |  +-- transOp(order = [0, 2, 1])
    //        |       ^  |
    //        |       |  +-- reshapeOp
    //        |       |        ^  |
    //        |       |        |  +-- (maybe convert_layout)
    //        |       |        +-- localLoadOp
    // -----------------------------------------------------------------------

    // Starting from the split source, peel off convert_layouts if any.
    Value src = stripConvertLayout(splitOp.getSrc());
    auto transOp = src.getDefiningOp<TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto reshapeOp = transOp.getSrc().getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();

    // Peel off convert_layouts *below* the reshape as well.  This is required
    // for the recursive case where the producer of the reshape is the result
    // of an earlier optimisation pass (i.e. a convert_layout of a previous
    // tmem_load).
    Value reshapeSrc = stripConvertLayout(reshapeOp.getSrc());
    auto smemLoad = reshapeSrc.getDefiningOp<gpu::LocalLoadOp>();
    if (!smemLoad)
      return failure();

    auto shape = reshapeOp.getResult().getType().getShape();
    // Ensure M dimension is preserved by the reshape.
    if (shape[0] != cast<RankedTensorType>(reshapeSrc.getType()).getShape()[0] &&
        !(shape[0] == 1 && smemLoad.getType().getRank() == 1))
      return failure();
    int splitNSize = shape[2];

    // Create the two SMEM subslices and their corresponding loads.
    Value smem = smemLoad.getSrc(); // Could itself be a subslice.
    rewriter.setInsertionPoint(smemLoad);
    triton::gpu::MemDescType smemType = smemLoad.getSrc().getType();
    auto createSliceLoad = [&](int32_t nOffset) -> Value {
      SmallVector<int64_t> splitShape;
      SmallVector<int32_t> offsets;
      if (smemLoad.getType().getRank() > 1) {
        splitShape.push_back(shape[0]);
        offsets.push_back(0);
      }
      offsets.push_back(nOffset);
      splitShape.push_back(splitNSize);
      // Generate the subslice op.
      auto dstType = triton::gpu::MemDescType::get(
          splitShape, smemType.getElementType(),
          smemType.getEncoding(), smemType.getMemorySpace(),
          smemType.getMutableMemory(), smemType.getAllocShape());
      Value subslice = rewriter.create<triton::gpu::MemDescSubsliceOp>(
          smemLoad->getLoc(), dstType, smem, offsets);
      RankedTensorType distributedType = splitOp.getOutLHS().getType();
      if (smemLoad.getType().getRank() == 1) {
        auto distributedLayout = triton::gpu::SliceEncodingAttr::get(
            rewriter.getContext(), 0,
            cast<ttg::DistributedEncodingTrait>(distributedType.getEncoding()));
        distributedType = RankedTensorType::get(
            splitShape, distributedType.getElementType(), distributedLayout);
      }
      Value subSliceTensor = rewriter.create<gpu::LocalLoadOp>(
          smemLoad.getLoc(), distributedType, subslice);
      if (smemLoad.getType().getRank() == 1) {
        subSliceTensor = rewriter.create<triton::ExpandDimsOp>(
            smemLoad.getLoc(), subSliceTensor, 0);
      }
      return subSliceTensor;
    };

    Value load0 = createSliceLoad(/*nOffset=*/0);
    Value load1 = createSliceLoad(/*nOffset=*/splitNSize);
    rewriter.replaceOp(splitOp, {load0, load1});
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUOptimizeTMemLayoutsPass
    : public impl::TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
          TritonNvidiaGPUOptimizeTMemLayoutsPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
      TritonNvidiaGPUOptimizeTMemLayoutsPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMemSplitLoadPattern, TMemStoreJoinPattern,
                 TMemLoadReducePattern, SMemSplitLoadPattern>(context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
