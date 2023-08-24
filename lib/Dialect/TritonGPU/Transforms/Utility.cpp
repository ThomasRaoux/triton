#include "triton/Analysis/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <fstream>

namespace mlir {

namespace {

class FixupLoop : public mlir::RewritePattern {

public:
  explicit FixupLoop(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);

    // Rewrite init argument
    SmallVector<Value, 4> newInitArgs = forOp.getInitArgs();
    bool shouldRematerialize = false;
    for (size_t i = 0; i < newInitArgs.size(); i++) {
      if (newInitArgs[i].getType() != forOp.getRegionIterArgs()[i].getType() ||
          newInitArgs[i].getType() != forOp.getResultTypes()[i]) {
        shouldRematerialize = true;
        break;
      }
    }
    if (!shouldRematerialize)
      return failure();

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->moveBefore(forOp);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    IRMapping mapping;
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    for (Operation &op : forOp.getBody()->getOperations()) {
      rewriter.clone(op, mapping);
    }
    rewriter.replaceOp(forOp, newForOp.getResults());
    return success();
  }
};

} // namespace

LogicalResult fixupLoops(ModuleOp mod) {
  auto *ctx = mod.getContext();
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<FixupLoop>(ctx);
  if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
    return failure();
  return success();
}

SmallVector<unsigned, 3> mmaVersionToInstrShape(int version,
                                                const ArrayRef<int64_t> &shape,
                                                RankedTensorType type) {
  if (version == 1)
    return {16, 16};
  else if (version == 2)
    return {16, 8};
  else if (version == 3) {
    unsigned k = 256 / type.getElementTypeBitWidth();
    if (shape[0] % 64 != 0 || shape[1] % 8 != 0) {
      assert(false && "type not supported");
      return {0, 0, 0};
    }
    auto eltType = type.getElementType();
    SmallVector<unsigned> validN;

    // MMAv3 with larger instruction shape is preferred.
    if (eltType.isFloat8E5M2() || eltType.isFloat8E4M3FN() || eltType.isF16() ||
        eltType.isBF16() || eltType.isF32()) {
      validN.assign({256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176,
                     168, 160, 152, 144, 136, 128, 120, 112, 104, 96,  88,
                     80,  72,  64,  56,  48,  40,  32,  24,  16,  8});
    }

    if (eltType.isInteger(8)) {
      validN.assign({224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32,
                     24, 16, 8});
    }

    for (auto n : validN) {
      if (shape[1] % n == 0) {
        return {16, n, k};
      }
    }

    assert(false && "type not supported");
    return {0, 0, 0};
  } else {
    assert(false && "version not supported");
    return {0, 0};
  }
}

bool isLoadFromTensorPtr(triton::LoadOp op) {
  return mlir::triton::isTensorPointerType(op.getPtr().getType());
}

bool isStoreToTensorPtr(triton::StoreOp op) {
  return mlir::triton::isTensorPointerType(op.getPtr().getType());
}

Operation *getFirstUser(Value v) {
  DenseMap<Operation *, size_t> operationId;
  v.getParentBlock()->walk<WalkOrder::PostOrder>(
      [&](Operation *op) { operationId[op] = operationId.size(); });
  size_t minId = std::numeric_limits<size_t>::max();
  Operation *firstUser = nullptr;
  for (Operation *user : v.getUsers()) {
    assert(operationId.find(user) != operationId.end());
    size_t userId = operationId[user];
    if (userId < minId) {
      minId = userId;
      firstUser = user;
    }
  }
  assert(firstUser);
  return firstUser;
}

triton::gpu::SharedEncodingAttr getSharedEncoding(RankedTensorType tensorTy) {
  auto blockedLayout =
      tensorTy.getEncoding().cast<triton::gpu::BlockedEncodingAttr>();
  return triton::gpu::SharedEncodingAttr::get(
      tensorTy.getContext(), tensorTy.getShape(), blockedLayout.getOrder(),
      blockedLayout.getCTALayout(), tensorTy.getElementType());
}

//===----------------------------------------------------------------------===//
// GraphDumper
//===----------------------------------------------------------------------===//

GraphDumper::NodeInfo GraphDumper::onValue(Value value) const {
  return {{"shape", "box"}, {"style", "filled"}, {"fillcolor", "white"}};
}

GraphDumper::NodeInfo GraphDumper::onOperation(Operation *op) const {
  return {{"shape", "ellipse"}, {"style", "filled"}, {"fillcolor", "white"}};
}

std::string GraphDumper::dump(triton::FuncOp func) const {
  llvm::SetVector<Value> values;
  llvm::SetVector<Operation *> operations;

  func.walk([&](Operation *op) {
    operations.insert(op);
    for (Value operand : op->getOperands())
      values.insert(operand);
    for (Value result : op->getResults())
      values.insert(result);
  });

  std::ostringstream oss;
  oss << "// Generated by Triton GraphDumper\n"
      << "\n"
      << "digraph {\n";

  oss << "    // Value Nodes\n";
  for (Value value : values)
    oss << "    " << emitValueNode(value) << "\n";
  oss << "\n";

  oss << "    // Operation Nodes\n";
  for (Operation *op : operations)
    oss << "    " << emitOperationNode(op) << "\n";
  oss << "\n";

  oss << "    // Edges\n";
  for (Operation *op : operations) {
    for (Value operand : op->getOperands())
      oss << "    " << emitEdge(getUniqueId(operand), getUniqueId(op)) << "\n";
    for (Value result : op->getResults())
      oss << "    " << emitEdge(getUniqueId(op), getUniqueId(result)) << "\n";
  }

  oss << "}\n";
  return oss.str();
}

void GraphDumper::dumpToFile(triton::FuncOp func,
                             const std::string &filename) const {
  std::ofstream ofs(filename);
  ofs << dump(func);
}

std::string GraphDumper::getShapeStr(const Type &type) const {
  std::ostringstream oss;
  oss << "[";
  if (auto tensorTy = type.dyn_cast<RankedTensorType>()) {
    auto shape = tensorTy.getShape();
    for (unsigned i = 0; i < shape.size(); ++i) {
      if (i > 0)
        oss << ", ";
      oss << shape[i];
    }
  }
  oss << "]";
  return oss.str();
}

std::string GraphDumper::getUniqueId(Value value) const {
  std::ostringstream oss;
  oss << value.getImpl();
  return oss.str();
}

std::string GraphDumper::getUniqueId(Operation *op) const {
  std::ostringstream oss;
  oss << op;
  return oss.str();
}

std::string GraphDumper::emitNode(const std::string &id,
                                  const GraphDumper::NodeInfo info) const {
  std::ostringstream oss;
  oss << "\"" << id << "\" [";
  for (auto it = info.begin(); it != info.end(); ++it) {
    if (it != info.begin())
      oss << ", ";
    oss << it->first << " = \"" << it->second << "\"";
  }
  oss << "];";
  return oss.str();
}

std::string GraphDumper::emitEdge(const std::string &srcId,
                                  const std::string &destId) const {
  std::ostringstream oss;
  oss << "\"" << srcId << "\" -> \"" << destId << "\";";
  return oss.str();
}

std::string GraphDumper::emitValueNode(Value value) const {
  NodeInfo info = onValue(value);
  if (info.find("label") == info.end()) {
    std::string shapeStr = getShapeStr(value.getType());
    if (auto arg = value.dyn_cast<BlockArgument>())
      info["label"] =
          "BlockArg" + std::to_string(arg.getArgNumber()) + " " + shapeStr;
    else
      info["label"] = shapeStr;
  }
  return emitNode(getUniqueId(value), info);
}

std::string GraphDumper::emitOperationNode(Operation *op) const {
  NodeInfo info = onOperation(op);
  if (info.find("label") == info.end())
    info["label"] = op->getName().getStringRef().str();
  return emitNode(getUniqueId(op), info);
}

//===----------------------------------------------------------------------===//
// GraphLayoutMarker
//===----------------------------------------------------------------------===//

GraphDumper::NodeInfo GraphLayoutMarker::onValue(Value value) const {
  std::string color = getColor(value.getType());
  return {{"shape", "box"}, {"style", "filled"}, {"fillcolor", color}};
}

std::string GraphLayoutMarker::getColor(const Type &type) const {
  if (auto tensorTy = type.dyn_cast<RankedTensorType>()) {
    auto layout = tensorTy.getEncoding();
    if (layout.isa<triton::gpu::BlockedEncodingAttr>())
      return "green";
    else if (layout.isa<triton::gpu::SliceEncodingAttr>())
      return "yellow";
    else if (layout.isa<triton::gpu::MmaEncodingAttr>())
      return "lightslateblue";
    else if (layout.isa<triton::gpu::DotOperandEncodingAttr>())
      return "orange";
    else if (layout.isa<triton::gpu::SharedEncodingAttr>())
      return "orangered";
    else
      assert(0 && "Unrecognized layout");
  } else {
    return "white";
  }
}
// -------------------------------------------------------------------------- //

// TODO: Interface
LogicalResult invertEncoding(Attribute targetEncoding, Operation *op,
                             Attribute &ret) {
  ret = targetEncoding;
  if (auto expand_dims = dyn_cast<triton::ExpandDimsOp>(op)) {
    ret = triton::gpu::SliceEncodingAttr::get(
        op->getContext(), expand_dims.getAxis(), targetEncoding);
  }
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    auto sliceEncoding =
        targetEncoding.dyn_cast<triton::gpu::SliceEncodingAttr>();
    if (!sliceEncoding)
      return failure();
    if (sliceEncoding.getDim() != reduce.getAxis())
      return failure();
    ret = sliceEncoding.getParent();
  }
  if (isa<triton::ViewOp, triton::CatOp>(op)) {
    return failure();
  }
  return success();
}

bool isExpensiveLoadOrStore(Operation *op) {
  // Case 1: Pointer of tensor is always expensive
  auto operandType = op->getOperand(0).getType();
  if (triton::isTensorPointerType(operandType))
    return true;
  // Case 2a: A size 1 tensor is not expensive since all threads will load the
  // same
  if (isSingleValue(op->getOperand(0)))
    return false;
  // Case 2b: Tensor of pointers has more threads than elements
  // we can presume a high hit-rate that makes it cheap to load
  auto ptrType = op->getOperand(0).getType().cast<RankedTensorType>();
  auto mod = op->getParentOfType<ModuleOp>();
  int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  if (ptrType.getNumElements() < numWarps * threadsPerWarp)
    return false;
  return true;
}

bool isExpensiveToRemat(Operation *op, Attribute &targetEncoding) {
  if (!op)
    return true;
  if (isa<triton::LoadOp, triton::StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  if (isa<triton::CatOp>(op))
    return triton::gpu::isExpensiveCat(cast<triton::CatOp>(op), targetEncoding);
  if (isa<tensor::ExtractSliceOp, triton::gpu::AllocTensorOp,
          triton::gpu::InsertSliceAsyncOp, triton::AtomicRMWOp,
          triton::AtomicCASOp, triton::DotOp>(op))
    return true;
  if (isa<scf::YieldOp, scf::ForOp, scf::IfOp, scf::WhileOp, scf::ConditionOp>(
          op))
    return true;
  return false;
}

bool canFoldIntoConversion(Operation *op, Attribute targetEncoding) {
  if (isa<triton::CatOp>(op))
    return !triton::gpu::isExpensiveCat(cast<triton::CatOp>(op),
                                        targetEncoding);
  return isa<triton::gpu::ConvertLayoutOp, arith::ConstantOp,
             triton::MakeRangeOp, triton::SplatOp, triton::ViewOp>(op);
}

int simulateBackwardRematerialization(
    Operation *initOp, SetVector<Operation *> &processed,
    SetVector<Attribute> &layout, llvm::MapVector<Value, Attribute> &toConvert,
    Attribute targetEncoding) {
  // DFS
  std::vector<std::pair<Operation *, Attribute>> queue;
  queue.emplace_back(initOp, targetEncoding);
  // We want to see the effect of converting `initOp` to a new layout
  // so we initialize `numCvts = 1`.
  int numCvts = 1;
  while (!queue.empty()) {
    Operation *currOp;
    Attribute currLayout;
    std::tie(currOp, currLayout) = queue.back();
    queue.pop_back();
    // If the current operation is expensive to rematerialize,
    // we stop everything
    if (isExpensiveToRemat(currOp, currLayout))
      return INT_MAX;
    // A conversion will be removed here (i.e. transferred to operands)
    numCvts -= 1;
    // Done processing
    processed.insert(currOp);
    layout.insert(currLayout);
    // Add all operands to the queue
    for (Value argI : currOp->getOperands()) {
      Attribute newEncoding;
      // Cannot invert the current encoding for this operand
      // we stop everything
      if (failed(invertEncoding(currLayout, currOp, newEncoding)))
        return INT_MAX;
      if (toConvert.count(argI) && toConvert[argI] != newEncoding)
        return INT_MAX;
      if (auto ptrTy = argI.getType().dyn_cast<triton::PointerType>()) {
        if (ptrTy.getPointeeType().isa<RankedTensorType>()) {
          return INT_MAX;
        }
      }

      Operation *opArgI = argI.getDefiningOp();
      toConvert.insert({argI, newEncoding});
      // 1. Only convert RankedTensorType
      // 2. Skip if there's no defining op
      // 3. Skip if the defining op has already been processed
      // 4. Skip or the defining op is in a different block
      if (!argI.getType().isa<RankedTensorType>())
        continue;
      if (opArgI && (processed.contains(opArgI) ||
                     opArgI->getBlock() != currOp->getBlock()))
        continue;
      // If the conversion can be folded into opArgI then
      // we don't count this conversion as expensive
      if (opArgI && canFoldIntoConversion(opArgI, newEncoding))
        continue;

      // We add one expensive conversion for the current operand
      numCvts += 1;
      if (opArgI)
        queue.emplace_back(opArgI, newEncoding);
    }
  }
  // return net number of conversions
  return numCvts;
}

//

Operation *cloneWithInferType(mlir::OpBuilder &rewriter, Operation *op,
                              IRMapping &mapping) {
  Operation *newOp = rewriter.clone(*op, mapping);
  // if input types haven't changed, we're done
  bool preserveTypes =
      std::all_of(op->operand_begin(), op->operand_end(), [&](Value v) {
        return !mapping.contains(v) ||
               v.getType() == mapping.lookup(v).getType();
      });
  if (preserveTypes)
    return newOp;

  if (newOp->getNumResults() == 0)
    return newOp;
  auto origType = op->getResult(0).getType().dyn_cast<RankedTensorType>();
  auto argType = newOp->getOperand(0).getType().dyn_cast<RankedTensorType>();
  if (!origType || !argType)
    return newOp;
  auto newType = RankedTensorType::get(
      origType.getShape(), origType.getElementType(), argType.getEncoding());
  newOp->getResult(0).setType(newType);
  auto typeInfer = dyn_cast<InferTypeOpInterface>(newOp);
  if (typeInfer) {
    SmallVector<Type, 1> newTypes;
    auto success = typeInfer.inferReturnTypes(
        newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
        newOp->getAttrDictionary(), newOp->getPropertiesStorage(),
        newOp->getRegions(), newTypes);
    if (succeeded(success)) {
      for (size_t i = 0; i < newTypes.size(); i++)
        newOp->getResult(i).setType(newTypes[i]);
    }
  }
  return newOp;
}

void rematerializeConversionChain(
    const llvm::MapVector<Value, Attribute> &toConvert,
    mlir::PatternRewriter &rewriter, SetVector<Operation *> &processed,
    IRMapping &mapping) {
  SmallVector<Value, 4> sortedValues;
  SetVector<Operation *> tmp;
  for (auto &item : toConvert) {
    Value v = item.first;
    if (v.getDefiningOp())
      tmp.insert(v.getDefiningOp());
    else
      sortedValues.push_back(v);
  }
  tmp = mlir::multiRootTopologicalSort(tmp);
  for (Operation *op : tmp)
    sortedValues.push_back(op->getResult(0));

  for (Value currOperand : sortedValues) {
    Value origOperand = currOperand;
    // unpack information
    Attribute targetLayout = toConvert.lookup(currOperand);
    // rematerialize the operand if necessary
    Operation *currOperation = currOperand.getDefiningOp();
    if (processed.contains(currOperation)) {
      Operation *newOperation =
          cloneWithInferType(rewriter, currOperation, mapping);
      newOperation->moveAfter(currOperation);
      currOperation = newOperation;
      currOperand = currOperation->getResult(0);
    }
    // compute target type for the layout cast
    auto currType = currOperand.getType().cast<RankedTensorType>();
    auto newType = RankedTensorType::get(
        currType.getShape(), currType.getElementType(), targetLayout);
    auto newOperand = rewriter.create<triton::gpu::ConvertLayoutOp>(
        currOperand.getLoc(), newType, currOperand);
    if (currOperation)
      newOperand->moveAfter(currOperation);
    else {
      Block *block = currOperand.cast<BlockArgument>().getOwner();
      newOperand->moveBefore(block, block->begin());
    }
    mapping.map(origOperand, newOperand);
  }
}

bool getBackwardSliceSCF(Value root, SetVector<Value> &slice,
                         llvm::function_ref<bool(Value)> filter,
                         Attribute rootEncoding,
                         DenseMap<Value, Attribute>& layout) {
  SmallVector<std::pair<Value, Attribute>> queue = {{ root, rootEncoding }};
  while (!queue.empty()) {
    auto [currentValue, encoding] = queue.back();
    queue.pop_back();
    if (filter && !filter(currentValue))
      continue;
    // Skip propagating through for op results for now.
    // TODO: enable this based on needs.
    if(currentValue.getDefiningOp<scf::ForOp>())
      return false;
    slice.insert(currentValue);
    layout[currentValue] = encoding;
    if (auto *definingOp = currentValue.getDefiningOp()) {
      if (auto forOp = dyn_cast<scf::ForOp>(definingOp)) {
        auto result = currentValue.cast<OpResult>();
        OpOperand &initOperand = forOp.getOpOperandForResult(result);
        Value yieldOperand = forOp.getBody()->getTerminator()->getOperand(
            result.getResultNumber());
        queue.push_back({initOperand.get(), encoding});
        queue.push_back({yieldOperand, encoding});
        continue;
      }
      if (canFoldIntoConversion(definingOp, encoding))
        continue;
      for (Value operand : definingOp->getOperands()) {
        Attribute srcEncoding;
        if(invertEncoding(encoding, definingOp, srcEncoding).failed())
          return false;
        if (slice.count(operand) == 0)
          queue.push_back({operand, srcEncoding});
      }
      continue;
    }
    auto blockArg = cast<BlockArgument>(currentValue);
    Block *block = blockArg.getOwner();
    Operation *parentOp = block->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      OpOperand &initOperand = forOp.getOpOperandForRegionIterArg(blockArg);
      Value yieldOperand = forOp.getBody()->getTerminator()->getOperand(
          blockArg.getArgNumber() - forOp.getNumInductionVars());
      queue.push_back({initOperand.get(), encoding});
      queue.push_back({yieldOperand, encoding});
      continue;
    }
    // TODO: add support for WhileOp and other region types.
    return false;
  }
  return true;
}

// TODO(thomas): this is duplicated with what is in GPUToLLVM
//  Convert an \param index to a multi-dim coordinate given \param shape and
//  \param order.
SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = reorder(shape, order);
  auto reorderedMultiDim = delinearize(b, loc, linear, reordered);
  SmallVector<Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  if (rank == 1) {
    multiDim[0] = linear;
  } else {
    Value remained = linear;
    for (auto &&en : llvm::enumerate(shape.drop_back())) {
      auto dimSize = b.create<arith::ConstantIntOp>(loc, en.value(), 32);
      multiDim[en.index()] = b.create<arith::RemSIOp>(loc, remained, dimSize);
      remained = b.create<arith::DivSIOp>(loc, remained, dimSize);
    }
    multiDim[rank - 1] = remained;
  }
  return multiDim;
}

Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order) {
  return linearize(b, loc, reorder<Value>(multiDim, order),
                   reorder<unsigned>(shape, order));
}

Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape) {
  auto rank = multiDim.size();
  Value linear = b.create<arith::ConstantIntOp>(loc, 0, 32);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      Value dimSize = b.create<arith::ConstantIntOp>(loc, dimShape, 32);
      linear = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, linear, dimSize), dim);
    }
  }
  return linear;
}

std::optional<int> getWSAgentId(Operation *op) {
  int prevAgentId = -1;
  if (auto attr = op->getAttrOfType<DenseIntElementsAttr>("async_agent")) {
    for (auto agentId : attr.getValues<int>()) {
      assert(prevAgentId == -1 && "support at most one agent id");
      prevAgentId = agentId;
    }
  }
  if (prevAgentId == -1)
    return std::nullopt;
  return prevAgentId;
}

std::optional<int> getWSRoleId(Operation *op) {
  if (!op->hasAttr("agent.mutex_role"))
    return std::nullopt;
  return op->getAttrOfType<IntegerAttr>("agent.mutex_role").getInt();
}

void setRoleId(Operation *op, int roleId) {
  auto attr = IntegerAttr::get(IntegerType::get(op->getContext(), 32), roleId);
  op->setAttr("agent.mutex_role", attr);
}

} // namespace mlir
