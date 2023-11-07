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
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static void peelLoop(scf::ForOp forOp, int numIterations) {
  OpBuilder builder(forOp);
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value newUpperBound = builder.create<arith::SubIOp>(forOp.getLoc(), ub, step);
  forOp.setUpperBound(newUpperBound);
  builder.setInsertionPointAfter(forOp);
  Location loc = forOp.getLoc();
  Type t = lb.getType();
  Value minusOne =
      builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(t, -1));
  // number of iterations = ((ub - 1) - lb) / step
  Value totlaNumIteration = builder.create<arith::DivUIOp>(
      loc,
      builder.create<arith::SubIOp>(
          loc, builder.create<arith::AddIOp>(loc, ub, minusOne), lb),
      step);
  // newLastIter = lb + step * ((((ub - 1) - lb) / step))
  Value newlastIter = builder.create<arith::AddIOp>(
      loc, lb, builder.create<arith::MulIOp>(loc, step, totlaNumIteration));

  SmallVector<int> escapeIndices;
  SmallVector<Type> escapeTypes;
  for (int i = 0; i < forOp.getNumResults(); i++) {
    if (forOp.getResult(i).use_empty())
      continue;
    escapeIndices.push_back(i);
    escapeTypes.push_back(forOp.getResult(i).getType());
  }

  IRMapping irMapping;
  irMapping.map(forOp.getInductionVar(), newlastIter);
  Value loopNotEmpty =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lb, ub);
  auto ifOp = builder.create<scf::IfOp>(loc, escapeTypes, loopNotEmpty,
                                        /*hasElse*/ true);

  for (int i = 0; i < escapeIndices.size(); i++) {
    forOp.getResult(escapeIndices[i])
        .replaceAllUsesWith(ifOp.getResult(escapeIndices[i]));
  }
  builder.setInsertionPointToStart(ifOp.thenBlock());
  for (int i = 0; i < forOp.getNumResults(); i++) {
    irMapping.map(forOp.getBody()->getArgument(i + 1), forOp.getResult(i));
  }
  for (Operation &op : forOp.getBody()->without_terminator()) {
    builder.clone(op, irMapping);
  }
  SmallVector<Value> ifYiledOperands;
  for (int i = 0; i < escapeIndices.size(); i++) {
    ifYiledOperands.push_back(irMapping.lookupOrDefault(
        forOp.getBody()->getTerminator()->getOperand(escapeIndices[i])));
  }
  auto ifYiledOp = builder.create<scf::YieldOp>(loc, ifYiledOperands);

  builder.setInsertionPointToStart(ifOp.elseBlock());
  SmallVector<Value> elseYieldOperands;
  for (int i = 0; i < escapeIndices.size(); i++) {
    elseYieldOperands.push_back(forOp.getResult(escapeIndices[i]));
  }
  builder.create<scf::YieldOp>(loc, elseYieldOperands);
}

static void pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::triton::PipeliningOption options;
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return;

  bool foundSchedule = false;
  foundSchedule = preProcessLoopAndGetSchedule(forOp, numStages, options);

  // TODO: add more pipelines strategy.
  if (!foundSchedule)
    return;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);

  if (succeeded(newForOp)) {
   peelLoop(newForOp.value(), 1);

    mlir::triton::asyncLaunchDots(newForOp.value());
  }
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
