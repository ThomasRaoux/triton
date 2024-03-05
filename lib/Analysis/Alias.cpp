#include "triton/Analysis/Alias.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

AliasInfo AliasInfo::join(const AliasInfo &lhs, const AliasInfo &rhs) {
  if (lhs == rhs)
    return lhs;
  AliasInfo ret;
  for (auto value : lhs.allocs) {
    ret.insert(value);
  }
  for (auto value : rhs.allocs) {
    ret.insert(value);
  }
  return ret;
}

void SharedMemoryAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
    ArrayRef<dataflow::Lattice<AliasInfo> *> results) {
  AliasInfo aliasInfo;
  bool pessimistic = true;
  // These ops may allocate a new shared memory buffer.
  auto result = op->getResult(0);

  // Only AllocOp creates a new buffer.
  if (isa<triton::gpu::AllocOp>(op)) {
    aliasInfo.insert(result);
    pessimistic = false;
  } else if (isa<triton::gpu::SubviewOp, triton::TransOp>(op)) {
    // extract_slice %src
    // trans %src
    aliasInfo = AliasInfo(operands[0]->getValue());
    pessimistic = false;
  } else if (result.getType().isa<triton::MemDescType>()) {
    for (int i = 0; i < op->getNumOperands(); i++) {
      if (op->getOperandTypes()[i].isa<triton::MemDescType>())
        aliasInfo.insert(op->getOperand(i));
    }
    pessimistic = false;
  }

  if (pessimistic) {
    return setAllToEntryStates(results);
  }
  // Join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(aliasInfo));
}

AliasResult SharedMemoryAliasAnalysis::alias(Value lhs, Value rhs) {
  // TODO: implement
  return AliasResult::MayAlias;
}

ModRefResult SharedMemoryAliasAnalysis::getModRef(Operation *op,
                                                  Value location) {
  // TODO: implement
  return ModRefResult::getModAndRef();
}

} // namespace mlir
