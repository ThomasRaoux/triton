#include "LLVMPasses.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

struct linkedValues {
  SmallVector<Value *> operands;
  Instruction* root;
};

bool isSameOp(Instruction *firstInst, Value *v) {
  if (Instruction *inst = dyn_cast<Instruction>(v)) {
    return inst->getOpcode() == firstInst->getOpcode();
  }
  return false;
}

bool supportVectorization(Instruction* inst) {
  switch(inst->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::FAdd:
    case Instruction::FSub:
      return true;
    default:
      break;
  }
  return false;
}

bool canVectorize(SmallVector<Value *> &operands) {
  Instruction* firstInst = dyn_cast<Instruction>(operands[0]);
  if(!firstInst)
    return false;
  if(!supportVectorization(firstInst))
    return false;
  for(Value *v : operands) {
    if(!isSameOp(firstInst, v))
      return false;
  }
  return true;
}

unsigned getNumOperands(Value *v) {
  if (Instruction *inst = dyn_cast<Instruction>(v)) {
    return inst->getNumOperands();
  }
  return 0;
}

Value* getOperand(Value *v, unsigned index) {
  if (Instruction *inst = dyn_cast<Instruction>(v)) {
    return inst->getOperand(index);
  }
  return nullptr;
}

Value *createInst(IRBuilder<>& builder, Instruction *inst, SmallVector<Value *> &srcs) {
  switch (inst->getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::FAdd:
  case Instruction::FSub:
    return builder.CreateBinOp(
        static_cast<Instruction::BinaryOps>(inst->getOpcode()), srcs[0],
        srcs[1]);
  }
  return nullptr;
}

Value *createVectorInst(SmallVector<Value *> &operands, Instruction *root,
                        std::vector<linkedValues> &worklist) {
  IRBuilder<> builder(root);
  Type *type = operands[0]->getType();
  unsigned vecSize = operands.size();
  Value *vec = UndefValue::get(FixedVectorType::get(type, vecSize));
  unsigned numOperands = getNumOperands(operands[0]);
  SmallVector<Value *> vecSrcs;
  for (unsigned i = 0; i < numOperands; i++) {
    linkedValues operandsValues;
    Value *firstEl = getOperand(operands[0], i);
    Value *vec = UndefValue::get(FixedVectorType::get(firstEl->getType(), vecSize));
    for (unsigned j = 0; j < vecSize; j++) {
      Value *operand = getOperand(operands[j], i);
      operandsValues.operands.push_back(operand);
      vec = builder.CreateInsertElement(vec, operand, j);
    }
    operandsValues.root = cast<Instruction>(vec);
    worklist.push_back(operandsValues);
    vecSrcs.push_back(vec);
  }
  Value *inst = createInst(builder, root, vecSrcs);
  return inst;
}

bool getBuildChain(InsertElementInst *root, linkedValues &values, DenseSet<Instruction*> &visited) {
  Value *element = root;
  unsigned vecSize = cast<FixedVectorType>(root->getType())->getNumElements();
  values.operands.resize(vecSize, nullptr);
  SmallVector<Instruction *> inserts;
  while (InsertElementInst *insertElement =
             dyn_cast<InsertElementInst>(element)) {
    if (visited.count(insertElement))
      return false;
    inserts.push_back(insertElement);
    ConstantInt *index = dyn_cast<ConstantInt>(insertElement->getOperand(2));
    if (!index)
      return false;
    unsigned indexCst = index->getZExtValue();
    if (indexCst >= vecSize)
      return false;
    values.operands[indexCst] = insertElement->getOperand(1);
  }
  for (Value *v : values.operands) {
    if (!v)
      return false;
  }
  visited.insert(inserts.begin(), inserts.end());
  values.root = root;
  return true;
}

static bool
runOnFunction(Function &F) {
  bool Changed = false;
  DenseSet<Instruction*> visited;
  std::vector<linkedValues> worklist;
  for (BasicBlock &BB : F) {
    for (Instruction &inst : BB) {
      linkedValues values;
      if (auto insertElement = dyn_cast<InsertElementInst>(&inst)) {
        if(getBuildChain(insertElement, values, visited)) {
          if(values.operands.size() == 2)
            worklist.push_back(values);
        }
      }
    }
  }

  while (!worklist.empty()) {
    linkedValues values = worklist.back();
    worklist.pop_back();
    if (canVectorize(values.operands)) {
      Value *vecInst = createVectorInst(values.operands, values.root, worklist);
      values.root->replaceAllUsesWith(vecInst);
      Changed = true;
    }
  }
  return Changed;
}

PreservedAnalyses VectorizeArith::run(Function &F,
                                               FunctionAnalysisManager &AM) {

  bool b = runOnFunction(F);
  llvm::errs() << F;
  return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
}