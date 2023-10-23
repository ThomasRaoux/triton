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

#define int_attr(num) builder.getI64IntegerAttr(num)

static Value createAlloc(scf::ForOp &forOp, tt::LoadOp loadOp,
                         unsigned distance) {
  OpBuilder builder(forOp);
  auto ty = loadOp.getType().cast<RankedTensorType>();
  if (!loadOp.getResult().hasOneUse())
    return Value();
  Operation *user = *loadOp.getResult().getUsers().begin();
  Attribute sharedEnc;
  auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(user);
  if (!convertLayout)
    return Value();
  auto tensorType =
      convertLayout.getResult().getType().cast<RankedTensorType>();
  if (auto dotOpEnc =
          tensorType.getEncoding().dyn_cast<ttg::DotOperandEncodingAttr>()) {
    bool needTrans = dyn_cast_or_null<tt::TransOp>(
        convertLayout->getOperand(0).getDefiningOp());
    unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
    auto CTALayout = ttg::getCTALayout(ty.getEncoding());
    sharedEnc = ttg::SharedEncodingAttr::get(
        ty.getContext(), dotOpEnc, ty.getShape(),
        ttg::getOrder(ty.getEncoding()), CTALayout, bitWidth, needTrans);
  }
  // TODO: support async copies for non matmul cases.
  if (!sharedEnc)
    return Value();
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type allocType =
      RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
  Value alloc = builder.create<mlir::triton::gpu::AllocTensorOp>(
      loadOp.getLoc(), allocType);
  return alloc;
}

static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  // Fix up the yield op.
  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands().begin(),
                              yieldOp->getOperands().end());
  operands.append(newOperands.begin(), newOperands.end());
  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

static void createAsyncLoad(scf::ForOp &forOp, tt::LoadOp loadOp,
                            unsigned defStage, unsigned useStage, Value alloc,
                            bool insertWait, Value insertIdx, Value extractIdx,
                            DenseMap<Operation *, unsigned> &opToStage) {
  OpBuilder builder(forOp);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  auto insertOp = builder.create<ttg::InsertSliceAsyncOp>(
      loc, alloc.getType(), loadOp.getPtr(), alloc, insertIdx, loadOp.getMask(),
      loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile(), /*axis*/ 0);
  opToStage[insertOp.getOperation()] = defStage;
  auto commmit = builder.create<ttg::AsyncCommitGroupOp>(loc);
  opToStage[commmit.getOperation()] = defStage;

  // Extract part.
  if (insertWait) {
    auto wait = builder.create<ttg::AsyncWaitOp>(loc, 0);
    opToStage[wait.getOperation()] = useStage - 1;
  }
  auto allocType = alloc.getType().cast<RankedTensorType>();
  RankedTensorType sliceType = RankedTensorType::get(
      {allocType.getShape()[1], allocType.getShape()[2]},
      allocType.getElementType(), allocType.getEncoding());
  auto extract = builder.create<ttg::ExtractSliceOp>(
      loc, sliceType, insertOp.getResult(),
      SmallVector<OpFoldResult>{extractIdx, int_attr(0), int_attr(0)},
      SmallVector<OpFoldResult>{int_attr(1), int_attr(sliceType.getShape()[0]),
                                int_attr(sliceType.getShape()[1])},
      SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});
  Operation *user = *loadOp.getResult().getUsers().begin();
  auto convertLayout = llvm::cast<ttg::ConvertLayoutOp>(user);
  auto newCvt = builder.create<ttg::ConvertLayoutOp>(
      convertLayout->getLoc(), convertLayout.getType(), extract.getResult());
  opToStage[extract.getOperation()] = useStage - 1;
  opToStage[newCvt.getOperation()] = useStage;
  convertLayout->replaceAllUsesWith(newCvt->getResults());
  convertLayout->erase();
  loadOp.erase();
  opToStage.erase(loadOp.getOperation());
  opToStage.erase(convertLayout.getOperation());

  // Fix up the yield op.
  appendToYield(forOp, {insertOp});
}

/// Convert loads where the def and uses are in different stages to async loads.
static void createAsynOps(scf::ForOp &forOp,
                          DenseMap<Operation *, unsigned> &opToStage,
                          int numStages) {
  struct AsyncLoad {
    AsyncLoad(tt::LoadOp loadOp, Value alloc, unsigned defStage,
              unsigned useStage)
        : loadOp(loadOp), alloc(alloc), defStage(defStage), useStage(useStage) {
    }
    tt::LoadOp loadOp;
    Value alloc;
    unsigned defStage;
    unsigned useStage;
  };
  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> newOperands;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto loadOp = dyn_cast<tt::LoadOp>(&op);
    if (!loadOp)
      continue;
    unsigned defStage = opToStage[&op];
    unsigned firstUse = numStages;
    for (auto user : op.getUsers()) {
      firstUse = std::min(firstUse, opToStage[user]);
    }
    if (firstUse == defStage)
      continue;
    Value alloc = createAlloc(forOp, loadOp, firstUse - defStage);
    if (!alloc)
      continue;
    newOperands.push_back(alloc);
    asyncLoads.emplace_back(loadOp, alloc, defStage, firstUse);
  }

  OpBuilder builder(forOp);
  // Create two new counters to index into the allocs.
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  Value insertIdx = zero;
  Value extractIdx = zero;
  Value numBuffers =
      builder.create<arith::ConstantIntOp>(forOp.getLoc(), numStages - 1, 32);
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  for (int i = 0; i < asyncLoads.size(); i++) {
    asyncLoads[i].alloc = newForOp.getBody()->getArgument(newOperandIndex + i);
  }
  builder.setInsertionPoint(asyncLoads.front().loadOp);
  insertIdx =
      newForOp.getBody()->getArgument(newOperandIndex + asyncLoads.size());
  extractIdx =
      newForOp.getBody()->getArgument(newOperandIndex + asyncLoads.size() + 1);
  Location loc = forOp.getLoc();
  
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  opToStage[insertIdx.getDefiningOp()] = 0;
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffers);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);
  opToStage[cndIns.getDefiningOp()] = 0;
  opToStage[insertIdx.getDefiningOp()] = 0;

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  opToStage[extractIdx.getDefiningOp()] = numStages - 2;  
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffers);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);
  opToStage[cndExt.getDefiningOp()] = numStages - 2;
  opToStage[extractIdx.getDefiningOp()] = numStages - 2;

  bool firstLoad = true;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    createAsyncLoad(forOp, asyncLoad.loadOp, asyncLoad.defStage,
                    asyncLoad.useStage, asyncLoad.alloc, firstLoad, insertIdx,
                    extractIdx, opToStage);
    firstLoad = false;
  }
  appendToYield(forOp, {insertIdx, extractIdx});
}

/// Helper to recursively add dependencies to the same stage.
static void addDep(Operation *op, unsigned stage,
                   DenseMap<Operation *, unsigned> &opToStage) {
  if (opToStage.count(op))
    return;
  opToStage[op] = stage;
  for (Value operand : op->getOperands()) {
    Value v = operand;
    int distance = 0;
    llvm::SmallDenseSet<Value> seen;
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

static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp &forOp, int numStages,
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

  createAsynOps(forOp, opToStage, numStages);

  // Schedule stage in decreasing order. This is an arbritrary choice but works
  // well for matmul loops, it can be passed a parameter to this function in the
  // future and decided based on the type of loop.
  std::vector<std::pair<Operation *, unsigned>> schedule;
  SmallVector<int> stagesOrder;
  for (int i = numStages - 1; i >= 0; i--) {
    if (i == numStages - 2)
      continue;
    stagesOrder.push_back(i);
  }
  stagesOrder.push_back(numStages - 2);
  for (int i : stagesOrder) {
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
  if (isa<ttg::AsyncCommitGroupOp>(op))
    return op;
  if (isa<ttg::AsyncWaitOp>(op))
    return op;
  if (auto insertOp = dyn_cast<ttg::InsertSliceAsyncOp>(op)) {
    Type maskType = tt::getI1SameShape(insertOp.getSrc().getType());
    Location loc = pred.getLoc();
    rewriter.setInsertionPoint(insertOp);
    Value mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
    if (insertOp.getMask()) {
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

static void tryMatmulPipeline(scf::ForOp forOp, int numStages) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  SmallVector<std::pair<Operation *, unsigned>> ops;
  collectOpsToPipeline(forOp, ops);
  if (ops.empty())
    return;

  // 2. Based on the coarse schedule adjust the loop and create a full schedule
  // where each operaiton in the loop is assigned to a stage.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages, ops);

  // for(auto it : schedule) {
  //   llvm::dbgs() << "schedule: " << *it.first << " " << it.second << "\n";
  // }

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
  unsigned numLoadsInStage = (numStages - 2) * ops.size();
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

static void pipelineLoop(scf::ForOp forOp, int numStages) {
  tryMatmulPipeline(forOp, numStages);
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
