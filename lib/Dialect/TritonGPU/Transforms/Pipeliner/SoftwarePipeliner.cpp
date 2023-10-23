#include "PipelineExpander.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliner are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue.
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// Return true if the load is transitively used by a dot operand.
static bool isLoadDotOperand(tt::LoadOp loadOp) {
  // We only pipeline loads that have one covert_layout (to dot_op) use
  // TODO: lift this constraint in the future
  bool isCandidate = false;
  if (!loadOp.getResult().hasOneUse())
    return false;

  Operation *use = *loadOp.getResult().getUsers().begin();
  Operation *preUse = nullptr;

  // Advance to the first conversion as long as the use resides in shared
  // memory and it has a single use itself
  while (use) {
    if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
      break;
    auto tensorType = use->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!tensorType.getEncoding().isa<ttg::SharedEncodingAttr>())
      break;
    preUse = use;
    use = *use->getResult(0).getUsers().begin();
  }

  if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
    if (auto tensorType =
            convertLayout.getResult().getType().dyn_cast<RankedTensorType>())
      if (auto dotOpEnc = tensorType.getEncoding()
                              .dyn_cast<ttg::DotOperandEncodingAttr>()) {
        isCandidate = true;
      }
  } else if (preUse && isa<tt::DotOp>(use)) {
    // for MMAv3 whose dot take SharedEncoding as operands directly
    Operation *post = *loadOp.getResult().getUsers().begin();
    auto newOrder = post->getResult(0)
                        .getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<ttg::SharedEncodingAttr>()
                        .getOrder();
    auto ty = loadOp.getType().cast<RankedTensorType>();
    auto oldOrder = ttg::getOrder(ty.getEncoding());
    // The operand of MMAv3 is in SharedEncoding and it's order should not
    // be changed after FuseTranspositions Pass. So we only pipeline the
    // load if the order of the loaded BlockedEncoding is the same as the
    // order of the SharedEncoding it is converted to.
    // TODO: remove this constraint once the LoadOp supports transpose
    // fusion
    if (newOrder[0] == oldOrder[0] || newOrder[1] == oldOrder[1]) {
      isCandidate = true;
    }
  }
  return isCandidate;
}

/// Collect loads to pipeline. Return success if we can pipeline this loop
static void
collectOpsToPipeline(scf::ForOp forOp,
                     SmallVectorImpl<std::pair<Operation *, unsigned>> &ops) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp)
    if (auto loadOp = dyn_cast<tt::LoadOp>(&op)) {
      bool candidate = false;
      if (isLoadFromTensorPtr(loadOp)) {
        // TODO: enable TMA pipelining.
        candidate = false;
      } else {
        auto ptr = loadOp.getPtr();
        unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
        if (auto mask = loadOp.getMask())
          vec =
              std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

        auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
        if (!tensorTy || tensorTy.getRank() < 2)
          continue;
        auto ty =
            tensorTy.getElementType().cast<tt::PointerType>().getPointeeType();
        unsigned width = vec * ty.getIntOrFloatBitWidth();
        // We do not pipeline all loads for the following reasons:
        // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8 and 16.
        // 2. It's likely that pipling small loads won't offer much performance
        //    improvement and may even hurt performance by increasing register
        //    pressure.
        if (width >= 32)
          candidate = true;
      }
      if (!candidate)
        continue;
      if (!isLoadDotOperand(loadOp))
        continue;
      ops.emplace_back(loadOp, 0);
    }
}

// Function to mask operations during scheduling.
static Operation *predicateOp(RewriterBase &rewriter, Operation *op,
                              Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if(isa<ttg::AsyncCommitGroupOp>(op))
    return op;
  if(isa<ttg::AsyncWaitOp>(op))
    return op;    
  if(auto insertOp = dyn_cast<ttg::InsertSliceAsyncOp>(op)) {
    Type maskType = tt::getI1SameShape(insertOp.getSrc().getType());
    Location loc = pred.getLoc();
    rewriter.setInsertionPoint(insertOp);
    Value mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
    if(insertOp.getMask()) {
      mask = rewriter.create<arith::AndIOp>(loc, mask, insertOp.getMask());
    }
    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  
  assert("don't know how to predicate this op" && false);
  return op;
}

static void setWaitNum(Operation *op,
                       mlir::triton::PipeliningOption::PipelinerPart part,
                       unsigned iteration, unsigned numLoadsInStage) {
  if (auto waitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
    waitOp.setNum(numLoadsInStage);
  }
}

static void pipelineLoop(scf::ForOp forOp, int numStages) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  SmallVector<std::pair<Operation *, unsigned>> ops;
  collectOpsToPipeline(forOp, ops);
  if (ops.empty())
    return;

  // 2. Based on the coarse schedule adjust the loop and create a full schedule
  // where each operaiton in the loop is assigned to a stage.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      mlir::triton::createSchedule(forOp, numStages, ops);

  // 3. rewrite the loop using the given schedule, this part of the
  // transformation doesn't take any decision.
  mlir::triton::PipeliningOption options;
  options.getScheduleFn =
      [&schedule](scf::ForOp forOp,
                  std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = predicateOp;
  options.needNumIterationChecks = false;
  unsigned numLoadsInStage = (numStages-2) * ops.size();
  options.annotateFn =
      [numLoadsInStage](Operation *op,
                        mlir::triton::PipeliningOption::PipelinerPart part,
                        unsigned iteration) {
        return setWaitNum(op, part, iteration, numLoadsInStage);
      };
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);
  OpBuilder builder(newForOp->getContext());
  builder.setInsertionPointAfter(newForOp->getOperation());
  builder.create<ttg::AsyncWaitOp>(newForOp->getLoc(), 0);
}

namespace {
struct PipelinePass : public TritonGPUPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int numStages, int numWarps, int numCTAs,
               int computeCapability) {
    this->numStages = numStages;
    this->numWarps = numWarps;
    this->numCTAs = numCTAs;
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    bool mode =
        computeCapability >= 90 && ::triton::tools::getBoolEnv("ENABLE_TMA");
    // TODO: handle TMA, this requires untangling a bunch of pieces.
    if (mode)
      return;
    if (this->numStages <= 1)
      return;
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops) {
      pipelineLoop(forOp, numStages);
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPipelinePass(int numStages,
                                                        int numWarps,
                                                        int numCTAs,
                                                        int computeCapability) {
  return std::make_unique<PipelinePass>(numStages, numWarps, numCTAs,
                                        computeCapability);
}
