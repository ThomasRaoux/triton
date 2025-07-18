#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "mlir/Analysis/Liveness.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;

namespace {

struct PressureAnnotationPass
    : public PassWrapper<PressureAnnotationPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PressureAnnotationPass);

  StringRef getArgument() const final { return "print-pressure-annotation"; }
  StringRef getDescription() const final {
    return "print the pressure annotation";
  }

  static int getValueSize(Value v) {
    if (v.getDefiningOp<arith::ConstantOp>())
      return 0;
    if (auto tensorTy = dyn_cast<RankedTensorType>(v.getType())) {
      int numElems = triton::gpu::getTotalElemsPerThread(tensorTy);
      int typeSize = isa<triton::PointerType>(v.getType())
                         ? 8
                         : tensorTy.getElementTypeBitWidth() / 8;
      return numElems * typeSize;
    } else {
      if (isa<triton::PointerType, triton::TensorDescType>(v.getType())) {
        return 8;
      } else if (isa<triton::gpu::MemDescType>(v.getType())) {
        return 4;
      } else {
        return v.getType().getIntOrFloatBitWidth() / 8;
      }
    }
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    Liveness liveness(moduleOp);
    OpBuilder builder(moduleOp);
    moduleOp.walk([&](Operation *op) {
      Block *block = op->getBlock();
      const LivenessBlockInfo *info = liveness.getLiveness(block);
      if (!info)
        return;

      auto liveNow = info->currentlyLiveValues(op);

     /* llvm::errs() << "LIVE @ ";
      op->print(llvm::errs());
      llvm::errs() << " : {";
      bool first = true;
      for (Value v : liveNow) {
        if (!first)
          llvm::errs() << ", ";
        first = false;
        v.print(llvm::errs());
      }
      llvm::errs() << "}\n";*/

      int pressure = 0;
      for (Value v : liveNow) {
        if (v.getDefiningOp() == op)
          continue;
        pressure += getValueSize(v);
      }
      op->setAttr("pressure", builder.getI32IntegerAttr(pressure));
      int dstSize = 0;
      for (auto result : op->getResults()) {
        dstSize += getValueSize(result);
      }
      op->setAttr("dst_size", builder.getI32IntegerAttr(dstSize));
      //llvm::errs() << "Pressure: " << pressure << " dstSize: " << dstSize << "\n";
   /*   llvm::errs() << "Pressure: " << pressure << "\n";
      llvm::errs() << "Op: ";
      op->print(llvm::errs());
      llvm::errs() << "\n";
      for (Value v : liveNow) {
        llvm::errs() << "Value: ";
        v.print(llvm::errs());
        llvm::errs() << " Size: " << getValueSize(v) << "\n";
        llvm::errs() << "\n";
      }*/
    });
  }
};

} // namespace

namespace mlir {
namespace tools {
void registerPressureAnnotationPass() {
  PassRegistration<PressureAnnotationPass>();
}
} // namespace tools
} // namespace mlir
