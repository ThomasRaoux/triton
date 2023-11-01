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
namespace ttng = mlir::triton::nvidia_gpu;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.

/// Replace the yield with a new one with the given operands appended.
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

static void createAsyncCopy(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
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

static void createTMALoad(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                          bool insertWait, Value insertIdx, Value extractIdx,
                          Value phase) {
  OpBuilder builder(forOp);
  Location loc = loadOp.getLoc();
  auto CTALayout = ttg::CTALayoutAttr::get(loadOp.getContext(),
                                           /*CTAsPerCGA*/ {1},
                                           /*CTASplitNum*/ {1},
                                           /*CTAOrder*/ {0});
  auto sharedEncoding = ttg::SharedEncodingAttr::get(loadOp.getContext(), 1, 1,
                                                     1, {0}, CTALayout, false);
  int64_t numBuffers = alloc.getType().cast<RankedTensorType>().getShape()[0];
  auto mBarriersTy = RankedTensorType::get(
      {numBuffers}, builder.getIntegerType(64), sharedEncoding);
  // Allocate an array of mbarrier objects outside the loop.
  Value barrierArray =
      builder.create<ttng::AllocMBarrierOp>(loc, mBarriersTy, 1);
  // extract the barrier and emit arriver/copy/wait/extract code sequence.
  builder.setInsertionPoint(loadOp);
  auto mBarTy = tt::PointerType::get(builder.getIntegerType(64), 3);
  Value barrier = builder.create<ttng::ExtractMBarrierOp>(
      loc, mBarTy, barrierArray, insertIdx);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value threadId = builder.create<ttng::GetThreadIdOp>(loc);
  Value pred = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             threadId, zero);

  auto loadTy = loadOp.getType().dyn_cast<RankedTensorType>();
  auto loadShape = loadTy.getShape();
  auto CTASplitNum = ttg::getCTASplitNum(loadTy.getEncoding());
  auto shapePerSlice = ttg::getShapePerCTA(CTASplitNum, loadShape);
  auto elemTy = loadTy.getElementType();
  unsigned elems = std::accumulate(shapePerSlice.begin(), shapePerSlice.end(),
                                   1, std::multiplies{});
  elems *= (elemTy.getIntOrFloatBitWidth() / 8);
  builder.create<ttng::MBarrierArriveOp>(loc, barrier, pred,
                                         /*remoteCtaId*/ nullptr,
                                         /*trackAsyncOp*/ false, elems);
  auto allocType = alloc.getType().cast<RankedTensorType>();
  auto insertOp = builder.create<ttng::InsertSliceAsyncV2Op>(
      loc, allocType, loadOp.getPtr(), alloc,
      /*index*/ insertIdx, barrier, loadOp.getMask(), loadOp.getOther(),
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile(),
      /*axis*/ 0);

  RankedTensorType sliceType = RankedTensorType::get(
      {allocType.getShape()[1], allocType.getShape()[2]},
      allocType.getElementType(), allocType.getEncoding());
  auto extract = builder.create<mlir::triton::gpu::ExtractSliceOp>(
      loc, sliceType, insertOp.getResult(),
      SmallVector<OpFoldResult>{extractIdx, int_attr(0), int_attr(0)},
      SmallVector<OpFoldResult>{int_attr(1), int_attr(sliceType.getShape()[0]),
                                int_attr(sliceType.getShape()[1])},
      SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});

  Value barrierWait = builder.create<ttng::ExtractMBarrierOp>(
      loc, mBarTy, barrierArray, extractIdx);
  builder.create<ttng::MBarrierWaitOp>(loc, barrierWait, phase);

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

/// Create an async load equivalent to the given load.
static void createAsyncLoad(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                            bool insertWait, Value insertIdx, Value extractIdx,
                            Value phase) {
  if (isLoadFromTensorPtr(loadOp)) {
    createTMALoad(forOp, loadOp, alloc, insertWait, insertIdx, extractIdx,
                  phase);
  } else {
    createAsyncCopy(forOp, loadOp, alloc, insertWait, insertIdx, extractIdx);
  }
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
            convertLayout.getResult().getType().dyn_cast<RankedTensorType>()) {
      if (auto dotOpEnc = tensorType.getEncoding()
                              .dyn_cast<ttg::DotOperandEncodingAttr>()) {
        return convertLayout.getResult();
      }
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
        // Map to TMA load.
        candidate = true;
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

// Create an allocation that can old distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, tt::LoadOp loadOp, Value dotOperand,
                         unsigned distance) {
  OpBuilder builder(forOp);
  auto ty = loadOp.getType().cast<RankedTensorType>();
  if (!loadOp.getResult().hasOneUse())
    return Value();
  Attribute sharedEnc;
  auto CTALayout = ttg::getCTALayout(ty.getEncoding());
  auto tensorType = dotOperand.getType().cast<RankedTensorType>();
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

// Convert load ops into their asyn version and apply multi-buffering based on
// the number of stages.
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
  bool needsMbarrierPhase = false;
  for (const LoadDotOperand &loadOperand : loads) {
    tt::LoadOp loadOp = loadOperand.load;
    Value dotOperand = loadOperand.dotOperand;
    Value alloc = createAlloc(forOp, loadOp, dotOperand, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    newOperands.push_back(alloc);
    asyncLoads.emplace_back(loadOp, alloc);
    needsMbarrierPhase |= isLoadFromTensorPtr(loadOp);
  }

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  // Create two new counters to index into the allocs.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value insertIdx = minusOne;
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  Value phase;
  if (needsMbarrierPhase) {
    phase = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    newOperands.push_back(phase);
  }
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
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffersVal);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  if (needsMbarrierPhase) {
    phase = newForOp.getBody()->getArgument(newOperandIndex +
                                            asyncLoads.size() + 2);
    Value oneI1 = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, oneI1);
    phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
  }

  bool firstLoad = true;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    createAsyncLoad(forOp, asyncLoad.loadOp, asyncLoad.alloc, firstLoad,
                    insertIdx, extractIdx, phase);
    firstLoad = false;
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  if (needsMbarrierPhase)
    newYieldOperands.push_back(phase);
  // Patch the yield with the updated counters.
  appendToYield(forOp, newYieldOperands);
}

// Combine the current mask with the given predicate.
static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
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
    Value mask = getPredMask(rewriter, insertOp.getSrc().getType(),
                             insertOp.getMask(), pred);
    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto insertOp = dyn_cast<ttng::InsertSliceAsyncV2Op>(op)) {
    rewriter.setInsertionPoint(insertOp);
    Value mask = getPredMask(
        rewriter,
        insertOp.getSrc().getType().cast<tt::PointerType>().getPointeeType(),
        insertOp.getMask(), pred);
    insertOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto arriveOp = dyn_cast<ttng::MBarrierArriveOp>(op)) {
    rewriter.setInsertionPoint(arriveOp);
    Value mask = getPredMask(rewriter, rewriter.getIntegerType(1),
                             arriveOp.getPred(), pred);
    arriveOp.getPredMutable().assign(mask);
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

static void setWaitNum(Operation *op,
                       mlir::triton::PipeliningOption::PipelinerPart part,
                       unsigned iteration, unsigned numLoadsInStage) {
  if (auto waitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
    waitOp.setNum(numLoadsInStage);
  }
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
    int distance = 0;
    llvm::SmallDenseSet<Value> seen;
    while (auto arg = v.dyn_cast<BlockArgument>()) {
      if (!includeArg)
        break;
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
      addDep(defOp, deps, includeArg, filter);
    }
  }
}

// Add operations to the shedule with the given stage based on the filter
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
createSchedule(scf::ForOp forOp, int numStages, bool prefetchExtract) {
  SmallVector<Operation *> insertOps;
  SmallVector<Operation *> extractOps;
  // Find the insert/extract ops that will go respectively in stage 0 and stage
  // `numStages - 2`. All the other operations will go in stage `numStages - 1`.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::InsertSliceAsyncOp, ttg::AsyncCommitGroupOp,
            ttng::MBarrierArriveOp, ttng::InsertSliceAsyncV2Op>(op))
      insertOps.emplace_back(&op);
    if (prefetchExtract) {
      if (isa<ttg::ExtractSliceOp, ttg::AsyncWaitOp>(op))
        extractOps.emplace_back(&op);
    }
  }
  DenseSet<Operation *> insertAndDeps;
  for (Operation *op : insertOps) {
    addDep(op, insertAndDeps, false);
  }

  // Find depenencies with distance of 1.
  SmallVector<Operation *> distance1Arith;
  for (Operation *op : insertAndDeps) {
    for (Value operand : op->getOperands()) {
      if (auto arg = operand.dyn_cast<BlockArgument>()) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
          auto yieldOp = op->getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && insertAndDeps.count(defOp) == 0) {
            distance1Arith.push_back(defOp);
          }
        }
      }
    }
  }
  // Keep loads at a distance of 1 schedule the rest in stage 0.
  for (Operation *op : distance1Arith) {
    if (isa<tt::LoadOp>(op)) {
      addDep(op, insertAndDeps, true);
    }
  }
  // For the rest of the ops we can move then into stage 1 so that they can be
  // closer to their uses.
  DenseSet<Operation *> stage1deps;
  for (Operation *op : distance1Arith) {
    if (!isa<tt::LoadOp>(op)) {
      addDep(op, stage1deps, true, &insertAndDeps);
    }
  }

  DenseSet<Operation *> extractAndDeps;
  for (Operation *op : extractOps) {
    addDep(op, extractAndDeps, true, &insertAndDeps);
  }
  std::vector<std::pair<Operation *, unsigned>> schedule;
  // Schedule stage `numStage - 1` first.
  addOps(forOp, numStages - 1, schedule, [&](Operation *op) {
    return insertAndDeps.count(op) == 0 && stage1deps.count(op) == 0 &&
           extractAndDeps.count(op) == 0;
  });

  // Schedule some dependencies with distance of 1 into stage 1 to reduce
  // pressure.
  addOps(forOp, 1, schedule,
         [&](Operation *op) { return stage1deps.count(op); });

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
  bool hasAsynCp = llvm::any_of(loads, [](LoadDotOperand &load) {
    return !isLoadFromTensorPtr(load.load);
  });
  // 2. Convert the loads into async loads and create the allocs.
  createAsynOps(forOp, loads, numStages, hasMMAV3);

  // 3. rewrite the loop using the given schedule, this part of the
  // transformation doesn't take any decision.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages, /*prefetchExtract=*/!hasMMAV3);

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

  if (hasAsynCp) {
    // Insert a wait 0 after the loop
    OpBuilder builder(forOp);
    builder.setInsertionPointAfter(forOp);
    builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), 0);
  }
  return true;
}

/// MMA V3 post-processing.
static bool selfDepend(tt::DotOp dotOp, scf::ForOp forOp,
                       Operation **firstUse) {
  std::function<bool(Value, int, scf::ForOp)> dependOn =
      [&dependOn](Value v, int argId, scf::ForOp forOp) {
        auto op = v.getDefiningOp();
        if (isa<BlockArgument>(v)) {
          auto iterArgs = forOp.getRegionIterArgs();
          auto iter = std::find(iterArgs.begin(), iterArgs.end(), v);
          if (iter != iterArgs.end())
            return std::distance(iterArgs.begin(), iter) == argId;
        } else {
          if (!op)
            return false;
          for (auto operand : op->getOperands()) {
            if (dependOn(operand, argId, forOp))
              return true;
          }
        }
        return false;
      };
  auto result = dotOp.getResult();
  auto yieldOp = forOp.getBody()->getTerminator();
  int argIdx = -1;
  auto iter = std::find(yieldOp->getOperands().begin(),
                        yieldOp->getOperands().end(), result);
  if (iter != yieldOp->getOperands().end())
    argIdx = std::distance(yieldOp->getOperands().begin(), iter);
  if (argIdx == -1)
    return false;
  for (auto operand : dotOp.getOperands()) {
    if (dependOn(operand, argIdx, forOp)) {
      auto iterArgs = forOp.getRegionIterArgs();
      *firstUse = iterArgs[argIdx].use_begin().getUser();
      return true;
    }
  }
  return false;
}

static void removeExtraWait(tt::nvidia_gpu::DotWaitOp dotWaitOp,
                            bool hasDotWait0) {
  if (hasDotWait0) {
    dotWaitOp->erase();
  }
}

void mlir::triton::asyncLaunchDots(scf::ForOp forOp) {
  Block *loop = forOp.getBody();
  auto getBlockNumInFor = [](Operation *op, scf::ForOp forOp) {
    if (!op)
      return -1l;
    auto lastOp = op;
    while (op->getBlock()->getParentOp() != forOp) {
      lastOp = op;
      op = op->getBlock()->getParentOp();
    }
    return std::distance(lastOp->getBlock()->getParent()->begin(),
                         lastOp->getBlock()->getIterator());
  };
  /// XXX(Keren): Clean up the following duplicate code with checkDotOp
  /// dots to be pipelined
  bool hasSyncDot = false;
  bool hasDotWait0 = false;
  SmallVector<tt::DotOp> allDots;
  SmallVector<tt::DotOp> dots;
  SmallVector<unsigned> resultNeedSync;
  for (Operation &op : *loop) {
    if (auto dotWaitOp = dyn_cast<tt::nvidia_gpu::DotWaitOp>(&op)) {
      auto attr = dotWaitOp->getAttrOfType<IntegerAttr>("pendings");
      auto pendingCount = attr.getInt();
      if (pendingCount == 0)
        hasDotWait0 = true;
    }
    if (auto dotOp = dyn_cast<tt::DotOp>(&op)) {
      allDots.push_back(dotOp);
    }
  }
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

          Operation *firstUse = nullptr;
          selfDepend(dotOp, forOp, &firstUse);
          bool selfDirectDepend = (dotOp == firstUse);
          for (auto tempInAll : allDots) {
            auto iter = std::find(dots.begin(), dots.end(), tempInAll);
            if (iter != dots.end())
              continue;
            auto db = getBlockNumInFor(tempInAll, forOp);
            auto fb = getBlockNumInFor(firstUse, forOp);
            if (db < fb ||
                (db == fb && db >= 0 && tempInAll->isBeforeInBlock(firstUse)))
              hasSyncDot = true;
          }
          auto CArg = dotOp.getOperand(2);
          if (!(selfDirectDepend || (!selfDirectDepend && hasSyncDot)) ||
              !CArg.hasOneUse())
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
  auto loc = lastDot.getLoc();
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

  hasDotWait0 = hasDotWait0 || hasSyncDot;

  // 2. If there's any outstanding DotAsyncOps, we need to wait for them.
  builder.setInsertionPointAfter(forOp);
  SmallVector<Type> resultTypes(resultNeedSync.size());
  SmallVector<Value> yieldThenValues(resultNeedSync.size());
  SmallVector<Value> yieldElseValues(resultNeedSync.size());
  for (int i = 0; i < resultNeedSync.size(); ++i) {
    resultTypes[i] = forOp->getResult(resultNeedSync[i]).getType();
    yieldThenValues[i] = forOp->getResult(resultNeedSync[i]);
    yieldElseValues[i] = forOp->getResult(resultNeedSync[i]);
  }
  Value loopNotEmpty = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, forOp.getLowerBound(),
      forOp.getUpperBound());
  auto ifOp = builder.create<scf::IfOp>(loc, resultTypes, loopNotEmpty,
                                        /*hasElse*/ true);
  builder.setInsertionPointToStart(ifOp.thenBlock());
  for (int i = 0; i < resultNeedSync.size(); ++i) {
    Value result = forOp->getResult(resultNeedSync[i]);
    if (result.use_empty())
      continue;
    auto dotWait =
        builder.create<tt::nvidia_gpu::DotWaitOp>(forOp.getLoc(), result, 0);
    result.replaceAllUsesExcept(ifOp.getResult(i), dotWait);
    yieldThenValues[i] = dotWait.getResult();
  }
  auto yieldOpThen = builder.create<scf::YieldOp>(loc, yieldThenValues);
  builder.setInsertionPointToEnd(ifOp.elseBlock());
  auto yieldOpElse = builder.create<scf::YieldOp>(loc, yieldElseValues);

  // 3. potentially remove redundant dot_wait after dot_async if having mutiple
  // DotOp in the loop
  removeExtraWait(dotWait, hasDotWait0);
}
