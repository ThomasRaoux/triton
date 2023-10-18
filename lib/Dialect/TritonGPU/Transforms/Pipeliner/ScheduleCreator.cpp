#include "Schedule.h"

using namespace mlir;

/// Helper to recursively add dependencies to the same stage.
static void addDep(Operation *op, unsigned stage,
                   DenseMap<Operation *, unsigned> &opToStage) {
  if (opToStage.count(op))
    return;
  opToStage[op] = stage;
  for (Value operand : op->getOperands()) {
    Value v = operand;
    int distance = 0;
    SmallDenseSet<Value> seen;
    while (auto arg = v.dyn_cast<BlockArgument>()) {
      if (!seen.insert(v).second)
        break;
      if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
        auto yieldOp = op->getBlock()->getTerminator();
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        distance++;
        continue;
      }
      break;
    }
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      addDep(defOp, stage, opToStage);
    }
  }
}

std::vector<std::pair<Operation *, unsigned>> mlir::triton::createSchedule(
    scf::ForOp forOp,
    int numStages,
    ArrayRef<std::pair<Operation *, unsigned>> coarseSchedule) {
  SmallVector<SmallVector<Operation *>> stages(numStages);
  for (auto opStage : coarseSchedule)
    stages[opStage.second].push_back(opStage.first);

  // Assign ops to the first stage where there is a dependency.
  DenseMap<Operation *, unsigned> opToStage;
  for (unsigned i = 0; i < stages.size(); ++i) {
    for (Operation *op : stages[i]) {
      addDep(op, i, opToStage);
    }
  }
  // Move all the remaining operations in the last stage.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!opToStage.count(&op))
      opToStage[&op] = numStages - 1;
  }

  createAsynOps(forOp, opToStage);

  // Schedule stage in decreasing order. This is an arbritrary choice but works
  // well for matmul loops, it can be passed a parameter to this function in the
  // future and decided based on the type of loop.
  std::vector<std::pair<Operation *, unsigned>> schedule;
  for (int i = numStages - 1; i >= 0; i--) {
    for (Operation &op : forOp) {
      if (!opToStage.count(&op))
        continue;
      if (opToStage[&op] == i) {
        schedule.emplace_back(&op, i);
      }
    }
  }
  return schedule;
}