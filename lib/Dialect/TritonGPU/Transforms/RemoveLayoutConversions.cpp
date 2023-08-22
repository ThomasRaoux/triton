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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <memory>

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SliceEncodingAttr;

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// convert(blocked, dot_operand) ->
// convert(blocked, mma) + convert(mma,  dot_operand)
// if this value is itself the result of a dot operation
// this is a heuristic to accommodate some pattern seen in fused attention
// kernels.
// TODO: replace this by something more generic, i.e. layout-aware CSE
class DecomposeDotOperand : public mlir::RewritePattern {

public:
  explicit DecomposeDotOperand(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    auto srcType = convert.getOperand().getType().cast<RankedTensorType>();
    auto dstType = convert.getType().cast<RankedTensorType>();
    if (srcType.getEncoding().isa<triton::gpu::BlockedEncodingAttr>() &&
        dstType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>()) {
      auto dstDotOperand =
          dstType.getEncoding().cast<triton::gpu::DotOperandEncodingAttr>();
      auto dstParent = dstDotOperand.getParent();
      if (dstDotOperand.getOpIdx() == 1 ||
          !dstParent.isa<triton::gpu::MmaEncodingAttr>())
        return mlir::failure();
      auto dstParentMma = dstParent.cast<triton::gpu::MmaEncodingAttr>();
      if (dstParentMma.isVolta() || dstParentMma.getWarpsPerCTA()[1] > 1)
        return mlir::failure();
      SetVector<Operation *> bwdSlices;
      mlir::getBackwardSlice(convert.getResult(), &bwdSlices);
      if (llvm::find_if(bwdSlices, [](Operation *op) {
            return isa<triton::DotOp>(op);
          }) == bwdSlices.end())
        return mlir::failure();

      auto tmpType = RankedTensorType::get(
          dstType.getShape(), dstType.getElementType(), dstParentMma);
      auto tmp = rewriter.create<triton::gpu::ConvertLayoutOp>(
          convert.getLoc(), tmpType, convert.getOperand());
      rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(op, dstType,
                                                                tmp);
      return mlir::success();
    }
    return mlir::failure();
  }
};

// Layout conversions are expensive. They require going through
// shared memory, which is orders of magnitude slower than
// other non-i/o operations in the dialect.
// It therefore makes sense to remove them whenever possible,
// even if it means rematerializing all values whose definitions
// are reachable from it without passing through any memory operation.
class RematerializeBackward : public mlir::RewritePattern {
public:
  explicit RematerializeBackward(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             3, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvt,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(cvt))
      return mlir::failure();
    // we don't touch block arguments
    Operation *op = cvt->getOperand(0).getDefiningOp();
    if (!op)
      return mlir::failure();
    // we don't want to rematerialize any conversion to/from shared
    if (triton::gpu::isSharedEncoding(cvt->getResults()[0]) ||
        triton::gpu::isSharedEncoding(cvt->getOperand(0)))
      return mlir::failure();
    // we don't handle conversions to DotOperandEncodingAttr
    // this is a heuristics to accommodate fused attention
    auto targetType = cvt->getResultTypes()[0].cast<RankedTensorType>();
    if (targetType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>())
      return mlir::failure();
    // DFS
    SetVector<Operation *> processed;
    SetVector<Attribute> layout;
    llvm::MapVector<Value, Attribute> toConvert;
    if (simulateBackwardRematerialization(cvt, processed, layout, toConvert,
                                          targetType.getEncoding()) > 0)
      return mlir::failure();

    IRMapping mapping;
    rematerializeConversionChain(toConvert, rewriter, processed, mapping);
    rewriter.replaceOp(cvt, mapping.lookup(cvt->getOperand(0)));

    return mlir::success();
  }
};

//
class ConvertDotConvert : public mlir::RewritePattern {
public:
  ConvertDotConvert(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstOp = cast<triton::gpu::ConvertLayoutOp>(op);
    auto dotOp =
        dyn_cast_or_null<triton::DotOp>(dstOp.getSrc().getDefiningOp());
    if (!dotOp)
      return mlir::failure();
    if (std::distance(dstOp->user_begin(), dstOp->user_end()) != 1 ||
        std::distance(dotOp->user_begin(), dotOp->user_end()) != 1)
      return mlir::failure();
    auto cvtOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        dotOp.getOperand(2).getDefiningOp());
    if (!cvtOp)
      return mlir::failure();
    auto loadOp =
        dyn_cast_or_null<triton::LoadOp>(cvtOp.getSrc().getDefiningOp());
    if (!loadOp)
      return mlir::failure();
    auto dstTy = dstOp.getResult().getType().cast<RankedTensorType>();
    auto srcTy = cvtOp.getOperand().getType().cast<RankedTensorType>();
    if (dstTy != srcTy)
      return mlir::failure();

    // TODO: int tensor cores
    auto out_dtype = dstTy.getElementType().cast<FloatType>();
    APFloat value(0.0f);
    if (out_dtype.isBF16())
      value = APFloat(APFloat::IEEEhalf(), APInt(16, 0));
    else if (out_dtype.isF16())
      value = APFloat(APFloat::IEEEhalf(), APInt(16, 0));
    else if (out_dtype.isF32())
      value = APFloat(0.0f);
    else
      llvm_unreachable("unsupported data type");

    auto _0f =
        rewriter.create<arith::ConstantFloatOp>(op->getLoc(), value, out_dtype);
    auto _0 = rewriter.create<triton::SplatOp>(
        op->getLoc(), dotOp.getResult().getType(), _0f);
    auto newDot = rewriter.create<triton::DotOp>(
        op->getLoc(), dotOp.getResult().getType(), dotOp.getOperand(0),
        dotOp.getOperand(1), _0, dotOp.getAllowTF32());
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op->getLoc(), dstTy, newDot.getResult());
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, newCvt, cvtOp.getOperand());
    return mlir::success();
  }
};

} // namespace

static bool isLayoutAnchor(Operation *op) {
  if (isa<triton::LoadOp, triton::StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  if (isa<triton::DotOp>(op))
    return true;
  return false;
}

namespace {
class LayoutPropagation {
public:
  struct LayoutInfo {
    LayoutInfo(Attribute encoding) { encodings = {encoding}; }
    LayoutInfo() {}
    llvm::SmallDenseSet<Attribute> encodings;
  };
  LayoutPropagation(ModuleOp m) : module(m) {}
  void initAnchorLayout();
  void propagateLayout();
  SmallVector<Value> propagateToUsers(Value value, LayoutInfo &info);
  void setEncoding(ValueRange values, LayoutInfo &info,
                   SmallVector<Value> &changed, Operation *op);
  void dump();

  void rewrite();
  void rewriteRegion(Region &R, IRMapping &mapping);
  Operation *rewriteOp(Operation *op, IRMapping &mapping);
  Operation *rewriteForOp(scf::ForOp forOp, IRMapping &mapping);

private:
  llvm::MapVector<Value, LayoutInfo> layouts;
  std::vector<Operation *> opToDelete;
  ModuleOp module;
};
} // namespace

void LayoutPropagation::initAnchorLayout() {
  module.walk([&](Operation *op) {
    if (isLayoutAnchor(op)) {
      for (auto result : op->getResults()) {
        if (auto tensorType = result.getType().dyn_cast<RankedTensorType>()) {
          layouts.insert({result, tensorType.getEncoding()});
        }
      }
    }
  });
}

static Attribute inferDestEncoding(triton::ReduceOp op, Attribute encoding) {
  return SliceEncodingAttr::get(op->getContext(), op.getAxis(), encoding);
}

static Attribute inferDestEncoding(triton::ExpandDimsOp op,
                                   Attribute encoding) {
  auto sliceEncoding = encoding.cast<triton::gpu::SliceEncodingAttr>();
  assert(op.getAxis() == sliceEncoding.getDim());
  return sliceEncoding.getParent();
}

void LayoutPropagation::setEncoding(ValueRange values, LayoutInfo &info,
                                    SmallVector<Value> &changed,
                                    Operation *op) {
  for (Value value : values) {
    bool hasChanged = false;
    SmallVector<Attribute> encodings(info.encodings.begin(),
                                     info.encodings.end());
    for (auto encoding : encodings) {
      if (auto reduceOp = dyn_cast<triton::ReduceOp>(op))
        encoding = inferDestEncoding(reduceOp, encoding);
      else if (auto expand = dyn_cast<triton::ExpandDimsOp>(op))
        encoding = inferDestEncoding(expand, encoding);
      hasChanged |= layouts[value].encodings.insert(encoding).second;
    }
    if (hasChanged)
      changed.push_back(value);
  }
}

SmallVector<Value> LayoutPropagation::propagateToUsers(Value value,
                                                       LayoutInfo &info) {
  SmallVector<Value> changed;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getRegionIterArgForOpOperand(use);
      Value result = forOp.getResultForOpOperand(use);
      setEncoding({arg, result}, info, changed, user);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto parent = yieldOp->getParentOp();
      SmallVector<Value> valuesToPropagate = {
          parent->getResult(use.getOperandNumber())};
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        valuesToPropagate.push_back(
            forOp.getRegionIterArg(use.getOperandNumber()));
      }
      setEncoding({valuesToPropagate}, info, changed, user);
      continue;
    }
    if (user->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
        user->hasTrait<mlir::OpTrait::Elementwise>() ||
        isa<triton::ReduceOp, triton::ExpandDimsOp,
            triton::gpu::ConvertLayoutOp>(user)) {
      setEncoding(user->getResults(), info, changed, user);
      continue;
    }
  }
  return changed;
}

void LayoutPropagation::propagateLayout() {
  std::vector<Value> queue;
  for (auto it : layouts) {
    queue.push_back(it.first);
  }
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo &info = layouts[currentValue];
    queue.pop_back();
    SmallVector<Value> changed = propagateToUsers(currentValue, info);
    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

void LayoutPropagation::dump() {
  for (auto it : layouts) {
    llvm::errs() << "Value: ";
    OpPrintingFlags flags;
    flags.skipRegions();
    it.first.print(llvm::errs(), flags);
    llvm::errs() << " \n encoding:\n";
    for (auto encoding : it.second.encodings) {
      encoding.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "--\n";
  }
}

void LayoutPropagation::rewrite() {
  module.walk([&](triton::FuncOp func) {
    IRMapping mapping;
    rewriteRegion(func->getRegion(0), mapping);
  });
}

void LayoutPropagation::rewriteRegion(Region &region, IRMapping &mapping) {
  SmallVector<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.back();
    queue.pop_back();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = false;
      for (Value result : op.getResults()) {
        auto it = layouts.find(result);
        // If we haven't mapped this value skip.
        if (it == layouts.end())
          continue;
        LayoutInfo &info = it->second;
        assert(info.encodings.size() == 1 &&
               "we should have resolved to a single encoding");
        auto encoding = result.getType().cast<RankedTensorType>().getEncoding();
        // If the encoding is already what we want skip.
        if (encoding == *info.encodings.begin())
          continue;
        needRewrite = true;
      }
      if (needRewrite) {
        Operation *newOp = rewriteOp(&op, mapping);
        for (Region &R : newOp->getRegions())
          queue.push_back(&R);
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
        for (OpOperand &operand : op.getOpOperands()) {
          if (mapping.contains(operand.get())) {
            Value newOperand = mapping.lookup(operand.get());
            OpBuilder builder(&op);
            // TODO: fix this hack
            if (!isa<scf::YieldOp>(op))
              newOperand = builder.create<triton::gpu::ConvertLayoutOp>(
                  op.getLoc(), operand.get().getType(), newOperand);
            op.setOperand(operand.getOperandNumber(), newOperand);
          }
        }
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      }
    }
  }
  for(Operation *op : llvm::reverse(opToDelete))
    op->erase();
}

// Force the given operand to be added to the value mapping. If the operand is
// not in the map it means that it wasn't part of the operations we are
// rewriting and we need to add a convert. Since the value is not in the layout
// it means it is not recheable from an anchor op so it will most likely be
// folded the backward rematerialization pattern.
static Value createMappingOperand(Value operand, Attribute encoding,
                                  IRMapping &mapping) {
  auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
  if (!tensorType)
    return operand;
  if (tensorType.getEncoding() == encoding)
    return operand;
  if (mapping.contains(operand))
    return mapping.lookup(operand);
  OpBuilder rewriter(operand.getContext());
  rewriter.setInsertionPointAfterValue(operand);
  auto tmpType = RankedTensorType::get(tensorType.getShape(),
                                       tensorType.getElementType(), encoding);
  Value converted = rewriter.create<triton::gpu::ConvertLayoutOp>(
      operand.getLoc(), tmpType, operand);
  mapping.map(operand, converted);
  return converted;
}

Operation *LayoutPropagation::rewriteForOp(scf::ForOp forOp,
                                           IRMapping &mapping) {
  SmallVector<Value> operands;
  OpBuilder rewriter(forOp);
  for (auto [operand, result] :
       llvm::zip(forOp.getInitArgs(), forOp.getResults())) {
    Value convertedOperand = operand;
    if (layouts.count(result))
      convertedOperand = createMappingOperand(
          operand, *layouts[result].encodings.begin(), mapping);
    operands.push_back(convertedOperand);
  }
  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), operands);

  newForOp.getBody()->getOperations().splice(
      newForOp.getBody()->getOperations().begin(),
      forOp.getBody()->getOperations());

  for (auto [oldResult, newResult] :
       llvm::zip(forOp.getResults(), newForOp.getResults()))
    mapping.map(oldResult, newResult);
  for (auto [oldArg, newArg] : llvm::zip(forOp.getBody()->getArguments(),
                                         newForOp.getBody()->getArguments())) {
    if(oldArg.getType() == newArg.getType()) {
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }
    mapping.map(oldArg, newArg);
  }
  opToDelete.push_back(forOp.getOperation());
  return newForOp.getOperation();
}

Operation *LayoutPropagation::rewriteOp(Operation *op, IRMapping &mapping) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return rewriteForOp(forOp, mapping);
  }
  OpBuilder rewriter(op);
  if (auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
    auto tensorType = convertOp.getType().cast<RankedTensorType>();
    auto newType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), encoding);
    Value operand = mapping.lookupOrDefault(convertOp.getOperand());
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        convertOp.getLoc(), newType, operand);
    mapping.map(convertOp.getResult(), newCvt.getResult());
    opToDelete.push_back(op);
    return newCvt;
  }
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>() ||
      isa<triton::ReduceOp, triton::ExpandDimsOp, triton::gpu::ConvertLayoutOp>(
          op)) {
    Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
    for (Value operand : op->getOperands())
      createMappingOperand(operand, encoding, mapping);
    Operation *newOp = cloneWithInferType(rewriter, op, mapping);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
    opToDelete.push_back(op);
    return newOp;
  }
  assert(false && "unhandled op");
  return nullptr;
}

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPURemoveLayoutConversionsPass
    : public TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
  TritonGPURemoveLayoutConversionsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    {
      mlir::RewritePatternSet patterns(context);
      patterns.add<RematerializeBackward>(context);
      ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);

      if (mlir::applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
        signalPassFailure();
      }
    }

    LayoutPropagation layoutPropagation(m);
    layoutPropagation.initAnchorLayout();
    layoutPropagation.propagateLayout();
    layoutPropagation.rewrite();

    {
      mlir::RewritePatternSet patterns(context);
      patterns.add<RematerializeBackward>(context);
      ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);

      if (mlir::applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
        signalPassFailure();
      }
    }

    {
      mlir::RewritePatternSet patterns(context);
      patterns.add<DecomposeDotOperand>(context);
      patterns.add<ConvertDotConvert>(context);

      if (mlir::applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
        signalPassFailure();
      }
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonGPURemoveLayoutConversionsPass() {
  return std::make_unique<TritonGPURemoveLayoutConversionsPass>();
}
