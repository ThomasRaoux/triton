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

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

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

/// Create an async load equivalent to the given load.
static void createAsyncLoad(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                            bool insertWait, Value insertIdx,
                            Value extractIdx) {
  OpBuilder builder(forOp);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  auto insertOp = builder.create<ttg::InsertSliceAsyncOp>(
      loc, alloc.getType(), loadOp.getPtr(), alloc, insertIdx, loadOp.getMask(),
      loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile(), /*axis*/ 0);
  auto commmit = builder.create<ttg::AsyncCommitGroupOp>(loc);

  // Extract part.
  if (insertWait)
    auto wait = builder.create<ttg::AsyncWaitOp>(loc, 0);
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
  convertLayout->replaceAllUsesWith(newCvt->getResults());
  convertLayout->erase();
  loadOp.erase();

  // Fix up the yield op.
  appendToYield(forOp, {insertOp});
}

// Return the transitive use of the load which is a dot operand.
static Value loadDotOperand(tt::LoadOp loadOp, bool &hasMMAV3) {
  // We only pipeline loads that have one covert_layout (to dot_op) use
  // TODO: lift this constraint in the future
  bool isCandidate = false;
  if (!loadOp.getResult().hasOneUse())
    return Value();

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
        return convertLayout.getResult();
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
      hasMMAV3 = true;
      return preUse->getResult(0);
    }
  }
  return Value();
}

namespace {
struct LoadDotOperand {
    LoadDotOperand(tt::LoadOp load, Value dotOperand)
        : load(load), dotOperand(dotOperand) {}
    tt::LoadOp load;
    Value dotOperand;
};
} // namespace

/// Collect loads to pipeline. Return success if we can pipeline this loop
static void collectOpsToPipeline(scf::ForOp forOp,
                                 SmallVectorImpl<LoadDotOperand> &ops,
                                 bool &hasMMAV3) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp) {
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
      Value dotOperand = loadDotOperand(loadOp, hasMMAV3);
      if (!dotOperand)
        continue;
      ops.emplace_back(loadOp, dotOperand);
    }
  }
}

static Value createAlloc(scf::ForOp &forOp, tt::LoadOp loadOp, Value dotOperand,
                         unsigned distance) {
  OpBuilder builder(forOp);
  auto ty = loadOp.getType().cast<RankedTensorType>();
  if (!loadOp.getResult().hasOneUse())
    return Value();
  Attribute sharedEnc;
  auto CTALayout = ttg::getCTALayout(ty.getEncoding());
  auto tensorType =
      dotOperand.getType().cast<RankedTensorType>();
  if (auto dotOpEnc =
          tensorType.getEncoding().dyn_cast<ttg::DotOperandEncodingAttr>()) {
    auto convertLayout = dotOperand.getDefiningOp<ttg::ConvertLayoutOp>();
    bool needTrans = dyn_cast_or_null<tt::TransOp>(
        convertLayout->getOperand(0).getDefiningOp());
    unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
    sharedEnc = ttg::SharedEncodingAttr::get(
        ty.getContext(), dotOpEnc, ty.getShape(),
        ttg::getOrder(ty.getEncoding()), CTALayout, bitWidth, needTrans);
  } else {
    // MMAv3
    sharedEnc = ttg::SharedEncodingAttr::get(ty.getContext(), ty.getShape(),
                                             ttg::getOrder(ty.getEncoding()),
                                             CTALayout, ty.getElementType());
  }
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type allocType =
      RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
  Value alloc = builder.create<mlir::triton::gpu::AllocTensorOp>(
      loadOp.getLoc(), allocType);
  return alloc;
}

static void createAsynOps(scf::ForOp &forOp, ArrayRef<LoadDotOperand> loads,
                          int numStages, bool hasMMAV3) {
  struct AsyncLoad {
    AsyncLoad(tt::LoadOp loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
    tt::LoadOp loadOp;
    Value alloc;
  };
  int numBuffers = numStages - 1;
  // For MMAv3 we need an extra buffer as this is assumed in the wgmma
  // pipelining post-processing. 
  // TODO: Improve modeling of wgmma pipelining.
  if (hasMMAV3)
    numBuffers++;
  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> newOperands;
  for (const LoadDotOperand& loadOperand : loads) {
    tt::LoadOp loadOp = loadOperand.load;
    Value dotOperand = loadOperand.dotOperand;
    Value alloc = createAlloc(forOp, loadOp, dotOperand, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    newOperands.push_back(alloc);
    asyncLoads.emplace_back(loadOp, alloc);
  }

  OpBuilder builder(forOp);
  // Create two new counters to index into the allocs.
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  Value insertIdx = zero;
  Value extractIdx = zero;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(forOp.getLoc(), numBuffers, 32);
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  for (int i = 0; i < asyncLoads.size(); i++) {
    asyncLoads[i].alloc = newForOp.getBody()->getArgument(newOperandIndex + i);
  }
  insertIdx =
      newForOp.getBody()->getArgument(newOperandIndex + asyncLoads.size());
  extractIdx =
      newForOp.getBody()->getArgument(newOperandIndex + asyncLoads.size() + 1);

  // Create two counters for the insert and extract indices to avoid creating
  // long liverange.
  builder.setInsertionPoint(asyncLoads.front().loadOp);
  Location loc = forOp.getLoc();
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffersVal);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  bool firstLoad = true;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    createAsyncLoad(forOp, asyncLoad.loadOp, asyncLoad.alloc, firstLoad,
                    insertIdx, extractIdx);
    firstLoad = false;
  }
  // Patch the yield with the updated counters.
  appendToYield(forOp, {insertIdx, extractIdx});
}

static Value getPredMask(RewriterBase &rewriter, Value src, Value currentMask,
                         Value pred) {
  Type maskType = tt::getI1SameShape(src.getType());
  Location loc = pred.getLoc();
  Value mask = pred;
  if (maskType.isa<RankedTensorType>()) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }
  if (currentMask) {
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
    Value mask = getPredMask(rewriter, insertOp.getSrc(), insertOp.getMask(),
                             pred);
    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    rewriter.setInsertionPoint(loadOp);
    Value mask = getPredMask(rewriter, loadOp.getPtr(), loadOp.getMask(),
                             pred);
    loadOp.getMaskMutable().assign(mask);
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

/// Helper to recursively add dependencies to the same stage.
static void addDep(Operation *op, DenseSet<Operation *> &deps,
                   DenseSet<Operation *> *filter = nullptr) {
  if (filter && filter->count(op))
    return;
  if (!deps.insert(op).second)
    return;
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
      addDep(defOp, deps, filter);
    }
  }
}

static void addOps(scf::ForOp forOp, int stage,
                   std::vector<std::pair<Operation *, unsigned>> &schedule,
                   std::function<bool(Operation *)> filter) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!filter(&op))
      continue;
    schedule.emplace_back(&op, stage);
  }
}

static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages) {
  SmallVector<Operation *> insertOps;
  SmallVector<Operation *> extractOps;
  // Find the insert/extract ops that will go respectively in stage 0 and stage
  // `numStages - 2`. All the other operations will go in stage `numStages - 1`.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::InsertSliceAsyncOp, ttg::AsyncCommitGroupOp>(op))
      insertOps.emplace_back(&op);
    if (isa<ttg::ExtractSliceOp, ttg::AsyncWaitOp>(op))
      extractOps.emplace_back(&op);
  }
  DenseSet<Operation *> insertAndDeps;
  for (Operation *op : insertOps) {
    addDep(op, insertAndDeps);
  }
  DenseSet<Operation *> extractAndDeps;
  for (Operation *op : extractOps) {
    addDep(op, extractAndDeps, &insertAndDeps);
  }
  std::vector<std::pair<Operation *, unsigned>> schedule;
  // Schedule stage `numStage - 1` first.
  addOps(forOp, numStages - 1, schedule, [&](Operation *op) {
    return insertAndDeps.count(op) == 0 && extractAndDeps.count(op) == 0;
  });

  // Then Schedule stage 0.
  addOps(forOp, 0, schedule,
         [&](Operation *op) { return insertAndDeps.count(op); });

  // Finally schedule the extract ops in stage `numStage - 2` so that they get
  // pre-fetched and play well with pretech pass.
  addOps(forOp, numStages - 2, schedule,
         [&](Operation *op) { return extractAndDeps.count(op); });
  return schedule;         
}

bool mlir::triton::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  SmallVector<LoadDotOperand> loads;
  bool hasMMAV3 = false;
  collectOpsToPipeline(forOp, loads, hasMMAV3);
  if (loads.empty())
    return false;
  // 2. Convert the loads into async loads and create the allocs.
  createAsynOps(forOp, loads, numStages, hasMMAV3);

  // 3. rewrite the loop using the given schedule, this part of the
  // transformation doesn't take any decision.
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
  unsigned numLoadsInStage = (numStages - 2) * loads.size();
  options.annotateFn =
      [numLoadsInStage](Operation *op,
                        mlir::triton::PipeliningOption::PipelinerPart part,
                        unsigned iteration) {
        return setWaitNum(op, part, iteration, numLoadsInStage);
      };
  
  // Insert a wait 0 after the loop
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), 0);

  return true;
}

void mlir::triton::asyncLaunchDots(scf::ForOp forOp) {
  Block *loop = forOp.getBody();

  /// XXX(Keren): Clean up the following duplicate code with checkDotOp
  /// dots to be pipelined
  SmallVector<tt::DotOp> dots;
  SmallVector<unsigned> resultNeedSync;
  for (Operation &op : *loop) {
    if (auto dotOp = dyn_cast<tt::DotOp>(&op)) {
      auto resTy = dotOp.getResult().getType().dyn_cast<RankedTensorType>();
      if (auto resEnc = resTy.getEncoding().dyn_cast<ttg::MmaEncodingAttr>()) {
        if (resEnc && resEnc.isHopper()) {
          // Don't pipeline valid dots that depend on ops other than scf.yield
          // and scf.for
          auto dot = dotOp.getResult();
          bool valid = true;

          // all users of dot should be scf.yield
          if (!dot.hasOneUse())
            valid = false;
          if (!isa<scf::YieldOp>(*dot.getUsers().begin()))
            valid = false;

          // C should be a block argument
          auto CArg = dotOp.getOperand(2).dyn_cast<BlockArgument>();
          if (!CArg || !CArg.hasOneUse())
            valid = false;

          if (valid) {
            dots.push_back(dotOp);
            resultNeedSync.push_back(
                dotOp->getUses().begin()->getOperandNumber());
          }
        }
      }
    }
  }

  // Early stop: no need to continue if there is no valid dot in the loop.
  if (dots.empty())
    return;

  OpBuilder builder(forOp);
  // 0. insert dot_wait after the last dot in the loop as we implicitly pipeline
  // wgmma ops by one stage.
  // This is needed to prevent shared memory inputs to be overriden before the
  // operation is completed.
  // TODO: merge this with the rest of the pipelining transformation and look at
  // a better representation for async dots.
  tt::DotOp lastDot = dots.back();
  builder.setInsertionPointAfter(lastDot);
  auto dotWait = builder.create<tt::nvidia_gpu::DotWaitOp>(
      lastDot.getLoc(), lastDot.getResult(), dots.size());

  // 1. replace Dot with DotAsync
  for (size_t idx = 0; idx < dots.size(); ++idx) {
    tt::DotOp dotOp = dots[idx];
    builder.setInsertionPoint(dotOp);
    auto dotAsync = builder.create<tt::nvidia_gpu::DotAsyncOp>(
        dotOp.getLoc(), dotOp.getA(), dotOp.getB(), dotOp.getC(),
        dotOp.getAllowTF32(), dotOp.getMaxNumImpreciseAcc());
    dotOp.replaceAllUsesWith(dotAsync.getResult());
    dotOp->erase();
  }

  // 2. If there's any outstanding DotAsyncOps, we need to wait for them.
  builder.setInsertionPointAfter(forOp);
  for (unsigned resultIndex : resultNeedSync) {
    Value result = forOp->getResult(resultIndex);
    if (result.use_empty())
      continue;
    auto dotWait =
        builder.create<tt::nvidia_gpu::DotWaitOp>(forOp.getLoc(), result, 0);
    result.replaceAllUsesExcept(dotWait.getResult(), dotWait);
  }
}