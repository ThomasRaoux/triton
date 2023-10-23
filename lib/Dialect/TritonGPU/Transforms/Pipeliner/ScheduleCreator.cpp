#include "Schedule.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define int_attr(num) builder.getI64IntegerAttr(num)

static void createAsyncLoad(scf::ForOp &forOp, tt::LoadOp loadOp,
                            unsigned defStage, unsigned useStage,
                            bool insertWait,
                            DenseMap<Operation *, unsigned> &opToStage) {
  OpBuilder builder(forOp);
  unsigned distance = useStage - defStage;
  auto ty = loadOp.getType().cast<RankedTensorType>();
  if (loadOp.getResult().hasOneUse()) {
    Operation *user = *loadOp.getResult().getUsers().begin();
    Attribute sharedEnc;
    auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(user);
    if (!convertLayout)
      return;
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
      return;
    SmallVector<int64_t> bufferShape(ty.getShape().begin(),
                                     ty.getShape().end());
    bufferShape.insert(bufferShape.begin(), distance);
    Type allocType =
        RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
    RankedTensorType sliceType =
        RankedTensorType::get(ty.getShape(), ty.getElementType(), sharedEnc);        
    Location loc = loadOp.getLoc();
    Value alloc =
        builder.create<triton::gpu::AllocTensorOp>(loadOp.getLoc(), allocType);
    Value ub = builder.create<arith::ConstantIntOp>(loc, distance, 32);

    // Make the alloc a loop argument.
    scf::ForOp newForOp = replaceForOpWithNewSignature(builder, forOp, {alloc});
    forOp.erase();
    forOp = newForOp;
    Value loopCarriedAlloc = forOp.getBody()->getArguments().back();
    // Replace the load with insert/extract slice.
    builder.setInsertionPoint(loadOp);
    Value iv = forOp.getInductionVar();
    if (iv.getType().isIndex())
      iv = builder.create<mlir::arith::IndexCastOp>(iv.getLoc(),
                                                    builder.getI32Type(), iv);
    Value index = builder.create<mlir::arith::RemUIOp>(loc, iv, ub);
    auto insertOp = builder.create<ttg::InsertSliceAsyncOp>(
        loc, loopCarriedAlloc.getType(), loadOp.getPtr(), loopCarriedAlloc,
        index, loadOp.getMask(), loadOp.getOther(), loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
    opToStage[insertOp.getOperation()] = defStage;
    opToStage[index.getDefiningOp()] = defStage;
    opToStage[iv.getDefiningOp()] = defStage;
    auto commmit = builder.create<ttg::AsyncCommitGroupOp>(loc);
    opToStage[commmit.getOperation()] = defStage;

    // Extract part.
    if (insertWait) {
      auto wait = builder.create<ttg::AsyncWaitOp>(loc, 0);
      opToStage[wait.getOperation()] = useStage - 1;
    }
    iv = forOp.getInductionVar();
    if (iv.getType().isIndex())
      iv = builder.create<mlir::arith::IndexCastOp>(iv.getLoc(),
                                                    builder.getI32Type(), iv);
    index = builder.create<mlir::arith::RemUIOp>(loc, iv, ub);
    auto extract = builder.create<ttg::ExtractSliceOp>(
        loc, sliceType, insertOp.getResult(),
        SmallVector<OpFoldResult>{index, int_attr(0), int_attr(0)},
        SmallVector<OpFoldResult>{int_attr(1),
                                  int_attr(sliceType.getShape()[0]),
                                  int_attr(sliceType.getShape()[1])},
        SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});
    auto newCvt = builder.create<ttg::ConvertLayoutOp>(convertLayout->getLoc(), convertLayout.getType(), extract.getResult());
    opToStage[index.getDefiningOp()] = useStage - 1;
    opToStage[iv.getDefiningOp()] = useStage - 1;
    opToStage[extract.getOperation()] = useStage - 1;
    opToStage[newCvt.getOperation()] = useStage;
    convertLayout->replaceAllUsesWith(newCvt->getResults());
    convertLayout->erase();
    loadOp.erase();
    opToStage.erase(loadOp.getOperation());
    opToStage.erase(convertLayout.getOperation());


    // Fix up the yield op.
    Operation *yieldOp = forOp.getBody()->getTerminator();
    SmallVector<Value> operands(yieldOp->getOperands().begin(),
                                yieldOp->getOperands().end());
    operands.push_back(insertOp);
    builder.setInsertionPoint(yieldOp);
    builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
    yieldOp->erase();
  }
}

/// Convert loads where the def and uses are in different stages to async loads.
static void createAsynOps(scf::ForOp &forOp,
                          DenseMap<Operation *, unsigned> &opToStage,
                          int numStages) {
  struct AsyncLoad {
    AsyncLoad(tt::LoadOp loadOp, unsigned defStage, unsigned useStage)
        : loadOp(loadOp), defStage(defStage), useStage(useStage) {}
    tt::LoadOp loadOp;
    unsigned defStage;
    unsigned useStage;
  };
  SmallVector<AsyncLoad> asyncLoads;
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
    asyncLoads.emplace_back(loadOp, defStage, firstUse);
  }
  bool firstLoad = true;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    createAsyncLoad(forOp, asyncLoad.loadOp, asyncLoad.defStage,
                    asyncLoad.useStage, firstLoad, opToStage);
    firstLoad = false;                    
  }
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
      addDep(defOp, stage + distance, opToStage);
    }
  }
}

std::vector<std::pair<Operation *, unsigned>> mlir::triton::createSchedule(
    scf::ForOp& forOp, int numStages,
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
    if(i == numStages - 2)
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