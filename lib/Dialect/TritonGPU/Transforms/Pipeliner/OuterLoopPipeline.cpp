#include "PipelineExpander.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// Combine the current mask with the given predicate.
static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
    bool scalarMask = false;
  if (auto s = currentMask.getDefiningOp<tt::SplatOp>()) {
    mask = rewriter.create<arith::AndIOp>(loc, s.getSrc(), pred);
    scalarMask = true;
  }
  if (maskType.isa<RankedTensorType>()) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }

  if (!scalarMask && currentMask) {
    mask = rewriter.create<arith::AndIOp>(loc, mask, currentMask);
  }
  return mask;
}

// Function to mask operations during scheduling.
static Operation *predicateOp(RewriterBase &rewriter, Operation *op,
                              Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if (isa<ttg::AsyncCommitGroupOp>(op))
    return op;
  if (isa<ttg::AsyncWaitOp>(op))
    return op;
  if (auto insertOp = dyn_cast<ttg::InsertSliceAsyncOp>(op)) {
    rewriter.setInsertionPoint(insertOp);
    Value mask = getPredMask(rewriter, insertOp.getSrc().getType(),
                             insertOp.getMask(), pred);
    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto insertOp = dyn_cast<ttng::InsertSliceTMAOp>(op)) {
//    rewriter.setInsertionPoint(insertOp);
//    Value mask = getPredMask(
//        rewriter,
//        insertOp.getSrc().getType().cast<tt::PointerType>().getPointeeType(),
//        insertOp.getMask(), pred);
//    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto arriveOp = dyn_cast<ttng::MBarrierArriveOp>(op)) {
  //  rewriter.setInsertionPoint(arriveOp);
  //  Value mask = getPredMask(rewriter, rewriter.getIntegerType(1),
  //                           arriveOp.getPred(), pred);
  //  arriveOp.getPredMutable().assign(mask);
    return op;
  }
  if (isa<ttng::MBarrierWaitOp>(op)) {
    return op;
  }
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    rewriter.setInsertionPoint(loadOp);
    Value mask = getPredMask(rewriter, loadOp.getPtr().getType(),
                             loadOp.getMask(), pred);
    loadOp.getMaskMutable().assign(mask);
    return op;
  }

  assert("don't know how to predicate this op" && false);
  return op;
}


/// Helper to recursively add dependencies to the same stage.
static void addDep(Operation *op, DenseSet<Operation *> &deps,
                   bool includeArg = true,
                   DenseSet<Operation *> *filter = nullptr) {
  if (filter && filter->count(op))
    return;
  if (!deps.insert(op).second)
    return;
  for (Value operand : op->getOperands()) {
    Value v = operand;
    llvm::SmallDenseSet<Value> seen;
    while (auto arg = v.dyn_cast<BlockArgument>()) {
      if (!includeArg)
        break;
      if (!seen.insert(v).second)
        break;
      if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
        auto yieldOp = op->getBlock()->getTerminator();
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        continue;
      }
      break;
    }
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      addDep(defOp, deps, includeArg, filter);
    }
  }
}

// Add operations to the schedule with the given stage based on the filter
// function.
static void addOps(scf::ForOp forOp, int stage,
                   std::vector<std::pair<Operation *, unsigned>> &schedule,
                   std::function<bool(Operation *)> filter) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!filter(&op))
      continue;
    schedule.emplace_back(&op, stage);
  }
}

// create the schedule for a matmul loop. This is ad hoc based on how we know
// matmul loops should be pipelined and is not a generic scheduler.
static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages) {
  SmallVector<Operation *> insertOps;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::InsertSliceAsyncOp, ttg::AsyncCommitGroupOp,
            ttng::MBarrierArriveOp, ttng::InsertSliceTMAOp>(op))
      insertOps.emplace_back(&op);
  }
  DenseSet<Operation *> insertAndDeps;
  for (Operation *op : insertOps) {
    addDep(op, insertAndDeps, true);
  }

  DenseSet<Operation *> epilogue;
  bool foundLoop = false;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (insertAndDeps.count(&op))
      continue;
    if (isa<scf::ForOp>(op))
      foundLoop = true;
    if (isa<scf::ForOp, ttg::AsyncWaitOp>(op))
      continue;
    if (foundLoop)
      epilogue.insert(&op);
  }

  std::vector<std::pair<Operation *, unsigned>> schedule;
  // Schedule stage 1 first.
  addOps(forOp, 1, schedule, [&](Operation *op) {
    return insertAndDeps.count(op) == 0 && epilogue.count(op) == 0;
  });

  // Then Schedule stage 0.
  addOps(forOp, 0, schedule,
         [&](Operation *op) { return insertAndDeps.count(op); });

  // Then schedule the epilogue in stage 1
  addOps(forOp, 1, schedule,
         [&](Operation *op) { return epilogue.count(op); });
  return schedule;
}

static void hoistAllocs(scf::ForOp forOp) {
  // 1. pre-process the loop by hosting allocations/deallocation out of the
  // loop.
  SmallVector<Operation *> allocs;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::AllocTensorOp, ttng::AllocMBarrierOp>(op))
      allocs.push_back(&op);
  }
  for (Operation *allocOp : allocs) {
    allocOp->moveBefore(forOp);
    for (Operation *user : allocOp->getUsers()) {
      if (auto dealloc = dyn_cast<ttg::DeallocTensorOp>(user)) {
        dealloc->moveAfter(forOp);
      }
    }
  }
}

static bool preConddition(scf::ForOp forOp) {
  // Check if there is a dependency from the loop to the async copy op. In this
  // case we cannot pipeline the async copy.
  SmallVector<Operation *> insertOps;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::InsertSliceAsyncOp, ttg::AsyncCommitGroupOp,
            ttng::MBarrierArriveOp, ttng::InsertSliceTMAOp>(op))
      insertOps.emplace_back(&op);
  }
  if (insertOps.empty())
    return false;
  DenseSet<Operation *> insertAndDeps;
  for (Operation *op : insertOps) {
    addDep(op, insertAndDeps, true);
  }
  for (Operation *op : insertAndDeps) {
    if (isa<scf::ForOp>(op))
      return false;
  }
  return true;
}

bool mlir::triton::getOuterLoopSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {

  // 1. Check precondition, we cannot have a recurrence involving async cp ops
  if (!preConddition(forOp))
    return false;
  // 2. pre-process the loop by hosting allocations.
  hoistAllocs(forOp);
  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages);

  // 4. Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = predicateOp;
  options.supportDynamicLoops = true;
  return true;
}
