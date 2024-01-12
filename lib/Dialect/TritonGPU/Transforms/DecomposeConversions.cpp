#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

class TritonGPUDecomposeConversionsPass
    : public TritonGPUDecomposeConversionsBase<
          TritonGPUDecomposeConversionsPass> {
public:
  TritonGPUDecomposeConversionsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcEncoding = srcType.getEncoding();
      if (srcEncoding.isa<triton::gpu::SharedEncodingAttr>())
        return;
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (!dstDotOp)
        return;
      if (auto srcMmaEncoding =
              srcEncoding.dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>()) {

        if (srcMmaEncoding.getVersionMajor() == 1 ||
            (srcMmaEncoding.getWarpsPerCTA()[1] == 1 &&
             dstDotOp.getParent() == srcMmaEncoding))
          return;
      }
      auto tmpType = RankedTensorType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(),
              triton::gpu::getOrder(srcEncoding),
              triton::gpu::getCTALayout(srcEncoding),
              srcType.getElementType()));
      auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getOperand());
      auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
          cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });

    SmallVector<Operation *> toErase;
    mod.walk([&](triton::StoreOp storeOp) -> void {
      OpBuilder builder(storeOp);
      auto cvtSrc =
          storeOp.getValue().getDefiningOp<triton::gpu::ConvertLayoutOp>();
      if (!cvtSrc)
        return;
      auto storeTy = storeOp.getValue().getType().cast<RankedTensorType>();
     // if (storeTy.getElementType().getIntOrFloatBitWidth() != 16)
     //   return;
      Location loc = storeOp.getLoc();
      auto encoding =
          cvtSrc.getSrc().getType().cast<RankedTensorType>().getEncoding();
      auto order = triton::gpu::getOrder(encoding);
      auto ctaLayout = triton::gpu::getCTALayout(encoding);
      auto sharedEncoding = triton::gpu::SharedEncodingAttr::get(
          mod.getContext(), 1, 1, 1, order, ctaLayout);
      auto tmpTy = RankedTensorType::get(
          storeTy.getShape(), storeTy.getElementType(), sharedEncoding);
      Value cvt = builder.create<triton::gpu::ConvertLayoutOp>(
          loc, tmpTy, cvtSrc.getOperand());
      builder.create<triton::nvidia_gpu::StoreAsyncOp>(loc, storeOp.getPtr(),
                                                       cvt);
      builder.create<mlir::triton::gpu::AsyncBulkCommitGroupOp>(loc);
      builder.create<mlir::triton::gpu::AsyncBulkWaitOp>(loc, 0);
      toErase.push_back(storeOp);
    });
    for (auto op : toErase)
      op->erase();
  }
};

std::unique_ptr<Pass> mlir::triton::gpu::createDecomposeConversionsPass() {
  return std::make_unique<TritonGPUDecomposeConversionsPass>();
}
