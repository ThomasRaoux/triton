#include "LLVMPasses.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

static bool processPhiStruct(PHINode *phiNode) {
  StructType *STy = dyn_cast<StructType>(phiNode->getType());
  if (!STy)
    return false;
  IRBuilder<> builder(phiNode);
  unsigned numOperands = phiNode->getNumIncomingValues();
  unsigned numScalarEl = STy->getNumElements();
  Value *newStruct = UndefValue::get(STy);
  for (unsigned i = 0; i < numScalarEl; i++) {
    builder.SetInsertPoint(phiNode);
    PHINode *newPhiNode =
        builder.CreatePHI(STy->getElementType(i), numOperands);
    for (unsigned j = 0; j < numOperands; ++j) {
      Value *operand = phiNode->getIncomingValue(j);
      builder.SetInsertPoint(phiNode->getIncomingBlock(j)->getTerminator());
      newPhiNode->addIncoming(builder.CreateExtractValue(operand, i),
                              phiNode->getIncomingBlock(j));
    }
    builder.SetInsertPoint(phiNode->getParent()->getFirstNonPHI());
    newStruct = builder.CreateInsertValue(newStruct, newPhiNode, i);
  }
  phiNode->replaceAllUsesWith(newStruct);
  return true;
}

static bool processPhiInsert(PHINode *phiNode) {
  InsertElementInst *insertElement = nullptr;
  for (User *U : phiNode->users()) {
    insertElement = dyn_cast<InsertElementInst>(U);
    if (insertElement)
      break;
  }
  if (!insertElement)
    return false;
  if (!isa<ConstantInt>(insertElement->getOperand(2)))
    return false;
  unsigned vecSize =
      cast<FixedVectorType>(insertElement->getType())->getNumElements();
  InsertElementInst *previousInsert = insertElement;
  SmallVector<InsertElementInst *> phisToCombine = {insertElement};
  while (phisToCombine.size() < vecSize) {
    previousInsert = dyn_cast<InsertElementInst>(previousInsert->getOperand(0));
    if (!previousInsert)
      return false;
    if (!isa<ConstantInt>(previousInsert->getOperand(2)))
      return false;
    PHINode *phi = dyn_cast<PHINode>(previousInsert->getOperand(1));
    if (!phi || phi->getParent() != phiNode->getParent())
      return false;
    phisToCombine.push_back(previousInsert);
  }
  unsigned numOperands = phiNode->getNumIncomingValues();
  IRBuilder<> builder(phiNode);
  PHINode *newPhiNode =
      builder.CreatePHI(insertElement->getType(), numOperands);

  for (unsigned j = 0; j < numOperands; ++j) {
    builder.SetInsertPoint(phiNode->getIncomingBlock(j)->getTerminator());
    Value *newVec = UndefValue::get(insertElement->getType());
    for (InsertElementInst *insert : phisToCombine) {
      PHINode *p = cast<PHINode>(insert->getOperand(1));
      newVec = builder.CreateInsertElement(newVec, p->getIncomingValue(j),
                                           insert->getOperand(2));
    }
    newPhiNode->addIncoming(newVec, phiNode->getIncomingBlock(j));
  }

  builder.SetInsertPoint(phiNode->getParent()->getFirstNonPHI());
  for (InsertElementInst *insert : phisToCombine) {
    Value *r = builder.CreateExtractElement(newPhiNode, insert->getOperand(2));
    insert->getOperand(1)->replaceAllUsesWith(r);
  }
  return true;
}

static bool runOnFunction(Function &F) {
  bool Changed = false;
  SmallVector<PHINode *> PhiNodes;
  for (BasicBlock &BB : F) {
    for (Instruction &inst : BB) {
      if (PHINode *phiNode = dyn_cast<PHINode>(&inst)) {
        if (processPhiStruct(phiNode)) {
          Changed = true;
          continue;
        }
       // Changed |= processPhiInsert(phiNode);
        continue;
      }
      break;
    }
  }
  return Changed;
}

PreservedAnalyses BreakStructPhiNodesPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {

  bool b = runOnFunction(F);
  llvm::errs() << F;
  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}