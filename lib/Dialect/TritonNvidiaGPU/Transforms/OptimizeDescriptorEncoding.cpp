#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/PriorityWorklist.h"
#include <algorithm>
#include <unordered_set>

namespace ttg = mlir::triton::gpu;

namespace {

struct UseInfo {
  TypedValue<TensorDescType> descriptor;
  Operation *use;
  Attribute desiredSharedEncoding;
  SmallVector<int64_t> shape;
  ttg::CGAEncodingAttr cgaLayout;
  Type desiredElementType;
  bool forbidTF32ElementType = false;
  SmallVector<Operation *> tf32RoundUsers;
};

static bool isTMACompatibleEncoding(Attribute enc) {
  if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(enc)) {
    return !nvmma.getTransposed();
  }
  return false;
}

Attribute findLoadEncodingFromUsers(Operation *op) {
  // Ignore multiple users and just pick the first compatible layout
  for (auto use : op->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(use)) {
      auto enc = alloc.getType().getEncoding();
      if (isTMACompatibleEncoding(enc))
        return enc;
    } else if (auto store = dyn_cast<ttg::LocalStoreOp>(use)) {
      auto enc = store.getDst().getType().getEncoding();
      if (isTMACompatibleEncoding(enc))
        return enc;
    }
  }
  return {};
}

SmallVector<int64_t> expandToRank(ArrayRef<int64_t> shape, int rank) {
  SmallVector<int64_t> result(rank, 1);
  assert(shape.size() <= rank);
  auto rankDiff = rank - shape.size();
  std::copy(shape.begin(), shape.end(), result.begin() + rankDiff);
  return result;
}

static bool allUsersAreTF32Round(Operation *op,
                                 SmallVectorImpl<Operation *> &roundUsers) {
  for (Operation *user : op->getUsers()) {
    if (!isa<TF32RoundOp>(user))
      return false;
    roundUsers.push_back(user);
  }
  return !roundUsers.empty();
}

std::optional<UseInfo> getUseInfo(Operation *op,
                                  SmallVectorImpl<UseInfo> &tf32Candidates) {
  UseInfo info;
  info.use = op;
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    info.descriptor = load.getDesc();
    info.desiredSharedEncoding = findLoadEncodingFromUsers(op);
    auto encoding = info.desiredSharedEncoding ? info.desiredSharedEncoding
                                               : load.getType().getEncoding();
    info.cgaLayout = ttg::getCGALayout(encoding);
    auto shape = load.getResult().getType().getShape();
    auto rank = load.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    auto descElemTy = load.getDesc().getType().getBlockType().getElementType();
    auto resultElemTy = load.getType().getElementType();
    if (descElemTy.isF32() && resultElemTy.isF32() &&
        allUsersAreTF32Round(op, info.tf32RoundUsers)) {
      info.desiredElementType = FloatTF32Type::get(op->getContext());
      tf32Candidates.push_back(info);
    }
    return info;
  }
  if (auto gather = dyn_cast<DescriptorGatherOp>(op)) {
    info.descriptor = gather.getDesc();
    info.desiredSharedEncoding = findLoadEncodingFromUsers(op);
    auto encoding = info.desiredSharedEncoding ? info.desiredSharedEncoding
                                               : gather.getType().getEncoding();
    info.cgaLayout = ttg::getCGALayout(encoding);
    auto shape = gather.getResult().getType().getShape();
    auto rank = gather.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    auto descElemTy =
        gather.getDesc().getType().getBlockType().getElementType();
    auto resultElemTy = gather.getType().getElementType();
    if (descElemTy.isF32() && resultElemTy.isF32() &&
        allUsersAreTF32Round(op, info.tf32RoundUsers)) {
      info.desiredElementType = FloatTF32Type::get(op->getContext());
      tf32Candidates.push_back(info);
    }
    return info;
  }
  if (auto store = dyn_cast<DescriptorStoreLikeOpInterface>(op)) {
    info.descriptor = store.getDesc();
    auto encoding = store.getSrc().getType().getEncoding();
    info.cgaLayout = ttg::getCGALayout(encoding);
    auto shape = store.getSrc().getType().getShape();
    auto rank = store.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    if (store.getDesc().getType().getBlockType().getElementType().isF32())
      info.forbidTF32ElementType = true;
    return info;
  }
  return std::nullopt;
}

struct EncodingInfo {
  Attribute desiredEncoding;
  ttg::CGAEncodingAttr cgaLayout;
  // Shape may be different from the descriptor block shape for gather/scatter
  // use case
  SmallVector<int64_t> shape;
  bool forcedToDefault = false;
  Type desiredElementType;
  bool forbidTF32ElementType = false;

  bool operator==(const EncodingInfo &other) const {
    return desiredEncoding == other.desiredEncoding &&
           cgaLayout == other.cgaLayout &&
           forcedToDefault == other.forcedToDefault &&
           desiredElementType == other.desiredElementType &&
           forbidTF32ElementType == other.forbidTF32ElementType &&
           shape == other.shape;
  }
};

} // namespace

template <> struct std::hash<EncodingInfo> {
  size_t operator()(const EncodingInfo &einfo) const {
    return llvm::hash_combine(einfo.desiredEncoding, einfo.cgaLayout,
                              einfo.forcedToDefault, einfo.desiredElementType,
                              einfo.forbidTF32ElementType,
                              ArrayRef<int64_t>(einfo.shape));
  }
};

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEDESCRIPTORENCODINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

const EncodingInfo *internEncoding(std::unordered_set<EncodingInfo> &encodings,
                                   EncodingInfo info) {
  return &*encodings.insert(info).first;
}

EncodingInfo combineEncodings(const EncodingInfo &lhs, const EncodingInfo &rhs,
                              unsigned rank) {
  EncodingInfo result;
  // Always propagate forcedToDefault
  result.forcedToDefault = lhs.forcedToDefault || rhs.forcedToDefault;
  result.forbidTF32ElementType =
      lhs.forbidTF32ElementType || rhs.forbidTF32ElementType;

  if (result.forcedToDefault)
    return result;

  if (lhs.shape.empty() || lhs.shape == rhs.shape)
    result.shape = rhs.shape;
  else if (rhs.shape.empty())
    result.shape = lhs.shape;
  else {
    assert(lhs.shape.size() == rhs.shape.size());
    auto rank = lhs.shape.size();
    result.shape.reserve(rank);
    for (int i = 0; i < rank; ++i)
      result.shape.push_back(std::min(lhs.shape[i], rhs.shape[i]));
  }

  SetVector<ttg::CGAEncodingAttr> cgaLayouts;
  if (lhs.cgaLayout)
    cgaLayouts.insert(lhs.cgaLayout);
  if (rhs.cgaLayout)
    cgaLayouts.insert(rhs.cgaLayout);

  auto getDefaultLayout = [&](ttg::CGAEncodingAttr encoding) {
    // The default layout puts all the CTAs in the last dimension
    // We do this as this function needs to be commutative for all encodings
    // This heuristic could be improved if needed
    auto ctx = encoding.getContext();
    auto kBlock = StringAttr::get(ctx, "block");
    auto dims = triton::standardOutDimNames(ctx, rank);
    auto numCTAs = encoding.getLinearLayout().getInDimSize(kBlock);
    LinearLayout llDefault;
    for (int i = 0; i < rank - 1; ++i) {
      llDefault *= LinearLayout::identity1D(1, kBlock, dims[i]);
    }
    llDefault *= LinearLayout::identity1D(numCTAs, kBlock, dims.back());
    return ttg::CGAEncodingAttr::get(ctx, llDefault);
  };

  switch (cgaLayouts.size()) {
  case 2:
    // if we find clashing CGALayouts, fallback to default
    result.cgaLayout = getDefaultLayout(lhs.cgaLayout);
    break;
  case 1:
    result.cgaLayout = cgaLayouts[0];
    break;
  default:
    break;
  }

  SetVector<Attribute> desiredEncodings;
  if (lhs.desiredEncoding)
    desiredEncodings.insert(lhs.desiredEncoding);
  if (rhs.desiredEncoding)
    desiredEncodings.insert(rhs.desiredEncoding);

  switch (desiredEncodings.size()) {
  case 2:
    // if we find clashing encodings, fallback to default
    result.forcedToDefault = true;
    break;
  case 1:
    result.desiredEncoding = desiredEncodings[0];
    break;
  default:
    break;
  }

  if (!result.forbidTF32ElementType) {
    if (lhs.desiredElementType && rhs.desiredElementType &&
        lhs.desiredElementType != rhs.desiredElementType) {
      result.desiredElementType = {};
    } else if (lhs.desiredElementType) {
      result.desiredElementType = lhs.desiredElementType;
    } else {
      result.desiredElementType = rhs.desiredElementType;
    }
  }
  return result;
}

Attribute getFallbackSharedEncoding(RankedTensorType tensorType,
                                    ttg::CGAEncodingAttr cgaLayout,
                                    ArrayRef<int64_t> usageShape,
                                    unsigned numCTAs) {
  auto ctx = tensorType.getContext();
  SmallVector<unsigned> order;
  for (int i = tensorType.getRank() - 1; i >= 0; --i)
    order.push_back(i);

  ArrayRef<int64_t> shape =
      usageShape.empty() ? tensorType.getShape() : usageShape;
  if (!cgaLayout) {
    // Arbitrarily distribute along the last dim
    SmallVector<unsigned> ctasPerCGA(tensorType.getRank(), 1);
    ctasPerCGA.back() = numCTAs;
    cgaLayout = ttg::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA,
                                                      ctasPerCGA, order);
  } else if (cgaLayout.getRank() != tensorType.getRank())
    cgaLayout = updateCGALayoutForShape(cgaLayout, shape);

  return ttg::NVMMASharedEncodingAttr::get(ctx, shape, order, cgaLayout,
                                           tensorType.getElementType(),
                                           /*fp4Padded*/ false);
}

TensorDescType getTensorDescTypeWithEncoding(Operation *op,
                                             RankedTensorType existingTy,
                                             Attribute encoding,
                                             Type elementType = {}) {
  if (elementType)
    existingTy = RankedTensorType::get(existingTy.getShape(), elementType);
  auto sharedEnc = cast<triton::gpu::SharedEncodingTrait>(encoding);
  encoding = updateEncodingForShape(op, sharedEnc, existingTy);
  auto blockTy = RankedTensorType::get(existingTy.getShape(),
                                       existingTy.getElementType(), encoding);
  return TensorDescType::get(existingTy.getContext(), blockTy);
}

void assignMemoryLayouts(FuncOp &func) {
  std::unordered_set<EncodingInfo> encodings;
  llvm::MapVector<TypedValue<TensorDescType>, const EncodingInfo *>
      valueToEncodingInfo;
  llvm::PriorityWorklist<TypedValue<triton::TensorDescType>> worklist;
  SmallVector<UseInfo> tf32Candidates;

  auto updateEncoding = [&](ArrayRef<Value> descValues, EncodingInfo info) {
    for (auto value : descValues) {
      auto typedVal = cast<TypedValue<TensorDescType>>(value);
      auto itr = valueToEncodingInfo.find(typedVal);
      if (itr != valueToEncodingInfo.end())
        info = combineEncodings(*itr->second, info,
                                typedVal.getType().getBlockType().getRank());
    }

    auto einfo = internEncoding(encodings, info);
    for (auto value : descValues) {
      auto typedVal = cast<TypedValue<TensorDescType>>(value);
      auto res = valueToEncodingInfo.try_emplace(typedVal, einfo);
      if (res.second) {
        worklist.insert(typedVal);
      } else if (res.first->second != einfo) {
        res.first->second = einfo;
        worklist.insert(typedVal);
      }
    }
  };

  // 1. Set seed values from either TMA ops, or device function boundaries for
  // which we fallback to default encoding
  auto isKernel = triton::isKernel(func);
  for (auto blockArg : func.getBlocks().front().getArguments())
    if (auto desc = dyn_cast<TypedValue<TensorDescType>>(blockArg))
      updateEncoding({desc},
                     EncodingInfo{{}, {}, {}, /*forcedToDefault=*/!isKernel});

  func.walk([&](Operation *op) {
    if (auto info = getUseInfo(op, tf32Candidates)) {
      updateEncoding(
          info->descriptor,
          EncodingInfo{info->desiredSharedEncoding, info->cgaLayout,
                       info->shape, /*forcedToDefault=*/false,
                       info->desiredElementType, info->forbidTF32ElementType});
    } else {
      bool forcedToDefault = isa<CallOp, ReturnOp, ReinterpretTensorDescOp>(op);
      auto einfo =
          internEncoding(encodings, EncodingInfo{{}, {}, {}, forcedToDefault});

      auto setEncoding = [&](Value v) {
        auto typedVal = cast<TypedValue<TensorDescType>>(v);
        valueToEncodingInfo.try_emplace(typedVal, einfo);
        if (forcedToDefault)
          worklist.insert(typedVal);
      };
      for (auto result : op->getResults())
        if (auto desc = dyn_cast<TypedValue<TensorDescType>>(result))
          setEncoding(desc);

      for (auto arg : op->getOperands())
        if (auto desc = dyn_cast<TypedValue<TensorDescType>>(arg))
          setEncoding(desc);
    }
  });

  // 2. Propagate encoding info through the graph until fixed point
  while (!worklist.empty()) {
    auto desc = worklist.pop_back_val();

    // Propagate to users
    for (OpOperand &use : desc.getUses()) {
      auto op = use.getOwner();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        auto offset = 3 * isa<scf::ForOp>(op);
        auto vals = getTiedArgs(op, use.getOperandNumber() - offset);
        updateEncoding(vals, EncodingInfo{});
      } else if (isa<scf::YieldOp>(op)) {
        auto vals = getTiedArgs(op->getParentOp(), use.getOperandNumber());
        updateEncoding(vals, EncodingInfo{});
      }
    }

    // Propagate to defining ops
    if (auto opResult = dyn_cast<OpResult>(desc)) {
      auto definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto vals = getTiedArgs(definingOp, opResult.getResultNumber());
        updateEncoding(vals, EncodingInfo{});
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(desc)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        auto offset = isa<scf::ForOp>(parentOp);
        auto vals = getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
        updateEncoding(vals, EncodingInfo{});
      }
    }
  }

  // 3. Fold tf32_round(load(desc)) into load(desc_tf32) when the descriptor
  // was proven to be TF32-only.
  auto ctx = func.getContext();
  IRRewriter rewriter(ctx);
  for (const UseInfo &candidate : tf32Candidates) {
    auto it = valueToEncodingInfo.find(candidate.descriptor);
    if (it == valueToEncodingInfo.end())
      continue;
    const EncodingInfo *einfo = it->second;
    if (einfo->forbidTF32ElementType ||
        !isa_and_nonnull<FloatTF32Type>(einfo->desiredElementType))
      continue;
    for (Operation *roundOp : candidate.tf32RoundUsers)
      rewriter.replaceOp(roundOp, candidate.use->getResult(0));
  }

  // 4. Transfer propagated encodings into the graph
  auto numCTAs = gpu::lookupNumCTAs(func);
  for (auto &[desc, einfo] : valueToEncodingInfo) {
    auto existingTy = desc.getType().getBlockType();
    Type elementType = existingTy.getElementType();
    if (!einfo->forbidTF32ElementType && einfo->desiredElementType)
      elementType = einfo->desiredElementType;
    auto elementTy = RankedTensorType::get(existingTy.getShape(), elementType);
    Attribute newEncoding;
    if (einfo->desiredEncoding) {
      newEncoding = einfo->desiredEncoding;
    } else if (einfo->forcedToDefault) {
      newEncoding = getFallbackSharedEncoding(elementTy, {}, {}, numCTAs);
    } else {
      newEncoding = getFallbackSharedEncoding(elementTy, einfo->cgaLayout,
                                              einfo->shape, numCTAs);
    }
    desc.setType(getTensorDescTypeWithEncoding(desc.getDefiningOp(), elementTy,
                                               newEncoding, elementType));
  }

  SmallVector<Type> argTys(func.getBlocks().front().getArgumentTypes());
  SmallVector<Type> resultTys(func.getResultTypes());
  for (auto [i, resultTy] : llvm::enumerate(resultTys)) {
    if (auto descTy = dyn_cast<TensorDescType>(resultTy)) {
      auto encoding =
          getFallbackSharedEncoding(descTy.getBlockType(), {}, {}, numCTAs);
      resultTys[i] = getTensorDescTypeWithEncoding(
          nullptr, descTy.getBlockType(), encoding);
    }
  }
  func.setFunctionType(FunctionType::get(ctx, argTys, resultTys));
}

void assignMemoryLayouts(ModuleOp &mod) {
  for (auto &op : *mod.getBody()) {
    if (auto func = dyn_cast<FuncOp>(&op)) {
      assignMemoryLayouts(func);
    }
  }
}

} // anonymous namespace

class TritonNvidiaGPUOptimizeDescriptorEncodingPass
    : public impl::TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
          TritonNvidiaGPUOptimizeDescriptorEncodingPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
      TritonNvidiaGPUOptimizeDescriptorEncodingPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    assignMemoryLayouts(m);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
